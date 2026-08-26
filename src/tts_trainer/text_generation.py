"""负责为 TTS 训练生成多语言语料文本：支持内置模板、CSV 导入与 OpenAI 兼容 LLM 三种来源，并提供缓存复用、断点续传与质量过滤。 / Generates multilingual TTS training texts via builtin templates, CSV import, or an OpenAI-compatible LLM, with corpus caching, resumable checkpoints, and quality filtering."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .experiments import prepare_experiment, resolve_experiment
from .frontend import frontend_from_config
from .logging_utils import configure_logging_from_config
from .text import normalize


logger = logging.getLogger(__name__)
# 语料名只允许安全字符，避免被拼进文件路径。 / Corpus names allow only safe chars so they can be embedded in paths.
CORPUS_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# API 密钥环境变量名校验，防止把密钥明文误写进配置。 / Validates an env-var name to catch API key values pasted by mistake.
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 各语料类别的默认采样权重，控制题材覆盖均衡。 / Default sampling weights per text category, balancing topic coverage.
DEFAULT_CATEGORIES = {
    "daily": 0.35,
    "question": 0.10,
    "number": 0.10,
    "date_time": 0.10,
    "money_unit": 0.10,
    "names_places": 0.10,
    "long_sentence": 0.10,
    "domain": 0.05,
}

# 这些确定性模板仅供流水线测试与覆盖度播种使用。产品级语料应使用人工审校文本或 LLM 生成。
# These deterministic templates are intended for pipeline tests and coverage
# seeding. Product corpora should use reviewed text or an LLM provider.
BUILTIN_TEMPLATES = {
    "zh": {
        "daily": "这是第{n}条日常语音提醒，请确认内容。",
        "question": "您能确认第{n}项请求是否已经完成吗？",
        "number": "当前统计数量是{number}，请记录这个结果。",
        "date_time": "会议安排在{month}月{day}日{hour}点{minute}分开始。",
        "money_unit": "编号{n}的商品价格是{amount}元。",
        "names_places": "{name}将在{place}的第{platform}号入口等候。",
        "long_sentence": "在处理第{n}项任务之前，请先核对网络状态、账户信息和通知设置，然后继续下一步操作。",
        "domain": "{keyword}服务的第{n}次状态更新已经准备完成。",
    },
    "en": {
        "daily": "This is daily voice reminder number {n}; please review the message.",
        "question": "Could you confirm whether request number {n} has been completed?",
        "number": "The current count is {number}; please record this result.",
        "date_time": "The meeting starts on {month}/{day} at {hour}:{minute:02d}.",
        "money_unit": "Item number {n} costs {amount} dollars.",
        "names_places": "{name} will wait at entrance {platform} in {place}.",
        "long_sentence": "Before processing task {n}, verify the network, account details, and notification settings, and then continue to the next step.",
        "domain": "Status update {n} for the {keyword} service is ready.",
    },
    "ja": {
        "daily": "これは{n}ばんめのおんせいあんないです。ないようをかくにんしてください。",
        "question": "{n}ばんのいらいがかんりょうしたか、かくにんできますか。",
        "number": "げんざいのすうちは{number}です。このけっかをきろくしてください。",
        "date_time": "よていは{month}がつ{day}にちの{hour}じ{minute}ふんにはじまります。",
        "money_unit": "{n}ばんのしょうひんは{amount}えんです。",
        "names_places": "{name}さんは{place}の{platform}ばんいりぐちでまっています。",
        "long_sentence": "{n}ばんのさぎょうをはじめるまえに、つうしんとせっていをかくにんしてから、つぎのてじゅんへすすんでください。",
        "domain": "{keyword}サービスの{n}かいめのじょうたいこうしんがじゅんびできました。",
    },
    "ko": {
        "daily": "이것은 {n}번째 일상 음성 알림입니다. 내용을 확인해 주세요.",
        "question": "{n}번 요청이 완료되었는지 확인해 주시겠습니까?",
        "number": "현재 집계된 수량은 {number}개입니다. 결과를 기록해 주세요.",
        "date_time": "회의는 {month}월 {day}일 {hour}시 {minute}분에 시작합니다.",
        "money_unit": "{n}번 상품의 가격은 {amount}원입니다.",
        "names_places": "{name} 님은 {place}의 {platform}번 입구에서 기다립니다.",
        "long_sentence": "{n}번 작업을 처리하기 전에 네트워크와 계정 정보, 알림 설정을 확인한 다음 다음 단계로 진행해 주세요.",
        "domain": "{keyword} 서비스의 {n}번째 상태 업데이트가 준비되었습니다.",
    },
    "de": {
        "daily": "Dies ist die tägliche Sprachmeldung Nummer {n}; bitte prüfen Sie den Inhalt.",
        "question": "Können Sie bestätigen, ob Anfrage Nummer {n} abgeschlossen ist?",
        "number": "Die aktuelle Anzahl beträgt {number}; bitte notieren Sie das Ergebnis.",
        "date_time": "Die Besprechung beginnt am {day}.{month}. um {hour}:{minute:02d} Uhr.",
        "money_unit": "Artikel Nummer {n} kostet {amount} Euro.",
        "names_places": "{name} wartet am Eingang {platform} in {place}.",
        "long_sentence": "Bevor Sie Aufgabe {n} bearbeiten, prüfen Sie Netzwerk, Kontodaten und Benachrichtigungen und fahren Sie dann mit dem nächsten Schritt fort.",
        "domain": "Statusaktualisierung {n} für den Dienst {keyword} ist verfügbar.",
    },
    "fr": {
        "daily": "Ceci est le rappel vocal quotidien numéro {n} ; veuillez vérifier le message.",
        "question": "Pouvez-vous confirmer que la demande numéro {n} est terminée ?",
        "number": "Le nombre actuel est {number} ; veuillez noter ce résultat.",
        "date_time": "La réunion commence le {day}/{month} à {hour} h {minute:02d}.",
        "money_unit": "L'article numéro {n} coûte {amount} euros.",
        "names_places": "{name} attendra à l'entrée {platform} de {place}.",
        "long_sentence": "Avant de traiter la tâche {n}, vérifiez le réseau, les informations du compte et les notifications, puis passez à l'étape suivante.",
        "domain": "La mise à jour numéro {n} du service {keyword} est prête.",
    },
    "ru": {
        "daily": "Это ежедневное голосовое напоминание номер {n}; проверьте сообщение.",
        "question": "Вы можете подтвердить, что запрос номер {n} выполнен?",
        "number": "Текущее количество равно {number}; запишите этот результат.",
        "date_time": "Встреча начнётся {day}.{month} в {hour}:{minute:02d}.",
        "money_unit": "Товар номер {n} стоит {amount} рублей.",
        "names_places": "{name} будет ждать у входа {platform} в городе {place}.",
        "long_sentence": "Перед выполнением задачи {n} проверьте сеть, данные учётной записи и настройки уведомлений, а затем переходите к следующему шагу.",
        "domain": "Обновление номер {n} для сервиса {keyword} готово.",
    },
    "pt": {
        "daily": "Este é o lembrete de voz diário número {n}; confira a mensagem.",
        "question": "Você pode confirmar se a solicitação número {n} foi concluída?",
        "number": "A contagem atual é {number}; registre este resultado.",
        "date_time": "A reunião começa em {day}/{month} às {hour}:{minute:02d}.",
        "money_unit": "O item número {n} custa {amount} reais.",
        "names_places": "{name} aguardará na entrada {platform} de {place}.",
        "long_sentence": "Antes de processar a tarefa {n}, confira a rede, os dados da conta e as notificações e depois avance para a próxima etapa.",
        "domain": "A atualização número {n} do serviço {keyword} está pronta.",
    },
    "es": {
        "daily": "Este es el recordatorio de voz diario número {n}; revisa el mensaje.",
        "question": "¿Puedes confirmar si la solicitud número {n} está terminada?",
        "number": "La cantidad actual es {number}; registra este resultado.",
        "date_time": "La reunión comienza el {day}/{month} a las {hour}:{minute:02d}.",
        "money_unit": "El artículo número {n} cuesta {amount} euros.",
        "names_places": "{name} esperará en la entrada {platform} de {place}.",
        "long_sentence": "Antes de procesar la tarea {n}, revisa la red, los datos de la cuenta y las notificaciones, y después continúa con el siguiente paso.",
        "domain": "La actualización número {n} del servicio {keyword} está lista.",
    },
    "it": {
        "daily": "Questo è il promemoria vocale quotidiano numero {n}; controlla il messaggio.",
        "question": "Puoi confermare se la richiesta numero {n} è stata completata?",
        "number": "Il conteggio attuale è {number}; registra questo risultato.",
        "date_time": "La riunione inizia il {day}/{month} alle {hour}:{minute:02d}.",
        "money_unit": "L'articolo numero {n} costa {amount} euro.",
        "names_places": "{name} aspetterà all'ingresso {platform} di {place}.",
        "long_sentence": "Prima di elaborare l'attività {n}, controlla la rete, i dati dell'account e le notifiche, quindi continua con il passaggio successivo.",
        "domain": "L'aggiornamento numero {n} del servizio {keyword} è pronto.",
    },
}

# 内置模板填充用的人名/地点素材与各语言领域关键词。 / Person/place fillers and per-language domain keywords for builtin templates.
NAMES = ["Alex", "Mina", "Luca", "Sofia", "Noah", "Yuna"]
PLACES = ["Central Station", "North Park", "City Hall", "Airport"]
DEFAULT_KEYWORDS = {
    "zh": ["通知", "导航", "日历", "天气", "消息"],
    "en": ["notification", "navigation", "calendar", "weather", "message"],
    "ja": ["つうち", "あんない", "よてい", "てんき", "めっせーじ"],
    "ko": ["알림", "내비게이션", "달력", "날씨", "메시지"],
    "de": ["Benachrichtigung", "Navigation", "Kalender", "Wetter", "Nachricht"],
    "fr": ["notification", "navigation", "calendrier", "météo", "message"],
    "ru": ["уведомление", "навигация", "календарь", "погода", "сообщение"],
    "pt": ["notificação", "navegação", "calendário", "clima", "mensagem"],
    "es": ["notificación", "navegación", "calendario", "clima", "mensaje"],
    "it": ["notifica", "navigazione", "calendario", "meteo", "messaggio"],
}


@dataclass(frozen=True)
class GeneratedText:
    """一条候选语料文本及其语言、类别与来源标签。 / One candidate corpus text with its language, category, and source tag."""

    text: str
    language: str
    category: str
    source: str


def _category_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """把总条数按权重分配到各类别，余数补给小数损失最大的类别。 / Split a total into per-category counts by weight, giving remainders to the largest fractional losses."""
    if total < 1:
        raise ValueError("text_generation.sentences_per_language must be at least 1")
    unknown = sorted(set(weights) - set(DEFAULT_CATEGORIES))
    if unknown:
        raise ValueError("unsupported text categories: " + ", ".join(unknown))
    positive = {key: float(value) for key, value in weights.items() if float(value) > 0}
    if not positive:
        raise ValueError("text_generation.categories must contain a positive weight")
    weight_sum = sum(positive.values())
    exact = {key: total * value / weight_sum for key, value in positive.items()}
    counts = {key: int(value) for key, value in exact.items()}
    for key in sorted(positive, key=lambda item: exact[item] - counts[item], reverse=True)[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def _keywords(config: dict, language: str) -> list[str]:
    """解析某语言的领域关键词，缺省时回退到内置列表。 / Resolve per-language domain keywords, falling back to builtin defaults."""
    value = config.get("domain", {}).get("keywords")
    if value is None:
        value = DEFAULT_KEYWORDS.get(language, DEFAULT_KEYWORDS["en"])
    if isinstance(value, dict):
        value = value.get(language) or value.get("default") or DEFAULT_KEYWORDS.get(language, DEFAULT_KEYWORDS["en"])
    result = [str(item).strip() for item in value if str(item).strip()]
    return result or DEFAULT_KEYWORDS.get(language, DEFAULT_KEYWORDS["en"])


def _context(serial: int, language: str, config: dict) -> dict:
    """由序号+语言确定性推导模板填充值，保证可复现。 / Deterministically derive template fill values from serial + language so output is reproducible."""
    # 语言名混入种子，使不同语言即使序号相同也得到不同人名/地点。 / Language mixes into the seed so equal serials diverge across languages.
    rng = random.Random(int(config.get("seed", 1337)) + serial * 7919 + sum(map(ord, language)))
    return {
        "n": serial,
        "number": 10 + (serial * 37) % 99990,
        "month": 1 + serial % 12,
        "day": 1 + serial % 28,
        "hour": 7 + serial % 15,
        "minute": (serial * 5) % 60,
        "amount": 5 + (serial * 13) % 995,
        "platform": 1 + serial % 20,
        "name": rng.choice(NAMES),
        "place": rng.choice(PLACES),
        "keyword": rng.choice(_keywords(config, language)),
    }


def _builtin_rows(language: str, total: int, config: dict) -> list[GeneratedText]:
    """用内置模板按类别配额批量生成确定性语料。 / Generate deterministic corpus rows from builtin templates by category quota."""
    if language not in BUILTIN_TEMPLATES:
        raise ValueError(
            f"builtin text generation has no templates for {language}; "
            "use provider=file/openai_compatible or contribute language templates"
        )
    weights = config.get("categories") or DEFAULT_CATEGORIES
    counts = _category_counts(total, weights)
    rows = []
    serial = 1
    for category, count in counts.items():
        template = BUILTIN_TEMPLATES[language][category]
        for _ in range(count):
            rows.append(GeneratedText(
                text=template.format(**_context(serial, language, config)),
                language=language,
                category=category,
                source="builtin",
            ))
            serial += 1
    return rows


def _file_rows(path: str | Path, supported_languages) -> list[GeneratedText]:
    """读取 CSV 语料并只保留受支持语言的行。 / Read a CSV corpus and keep only rows in supported languages."""
    source = Path(path)
    # utf-8-sig 兼容带 BOM 的 Excel 导出文件。 / utf-8-sig tolerates BOM from Excel exports.
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = {"text", "language"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError("text source CSV missing columns: " + ", ".join(sorted(missing)))
        return [
            GeneratedText(
                text=row["text"].strip(),
                language=row["language"].strip().lower(),
                category=(row.get("category") or "imported").strip(),
                source=(row.get("source") or "file").strip(),
            )
            for row in reader
            if row["language"].strip().lower() in supported_languages
        ]


def _openai_compatible_config(config: dict) -> dict:
    """合并并校验 openai_compatible 提供方配置（嵌套键优先，顶层键可覆盖）。 / Merge and validate openai_compatible provider config (nested keys first, top-level keys override)."""
    nested = config.get("openai_compatible") or {}
    if not isinstance(nested, dict):
        raise ValueError("text_generation.openai_compatible must be an object")
    # 允许把 endpoint/model 等键直接写在 text_generation 顶层以便快捷配置。 / Allow endpoint/model etc. at the text_generation top level for convenience.
    resolved = dict(nested)
    for key in ("endpoint", "base_url", "model", "api_key_env", "temperature",
                "timeout_seconds", "batch_size", "max_rounds", "max_retries",
                "retry_backoff_seconds", "retry_max_backoff_seconds"):
        if key in config:
            resolved[key] = config[key]
    endpoint = str(resolved.get("endpoint") or resolved.get("base_url") or "").rstrip("/")
    model = str(resolved.get("model") or "").strip()
    if not endpoint or not model:
        raise ValueError(
            "text_generation provider=openai_compatible requires endpoint and model; "
            "set text_generation.endpoint/model, or use provider=builtin for a smoke test"
        )
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("text_generation.endpoint must start with http:// or https://")
    key_value = resolved.get("api_key_env", "OPENAI_API_KEY")
    key_env = str(key_value) if key_value else None
    if key_env and not ENVIRONMENT_NAME.fullmatch(key_env):
        raise ValueError(
            "text_generation.api_key_env must be an environment variable name such as "
            "TEXT_LLM_API_KEY, not the API key value"
        )
    resolved.update({"endpoint": endpoint, "model": model, "api_key_env": key_env})
    return resolved


def validate_text_generation_config(config: dict) -> None:
    """入口前校验 text_generation 配置的取值合法性。 / Validate text_generation config values before use."""
    provider = str(config.get("provider", "builtin"))
    if provider not in {"builtin", "file", "openai_compatible"}:
        raise ValueError("text_generation.provider must be builtin, file, or openai_compatible")
    if provider == "file" and not config.get("input"):
        raise ValueError("text_generation provider=file requires input")
    if provider == "openai_compatible":
        resolved = _openai_compatible_config(config)
        if float(resolved.get("timeout_seconds", 180)) <= 0:
            raise ValueError("text_generation.timeout_seconds must be greater than zero")
        if int(resolved.get("max_retries", 4)) < 0:
            raise ValueError("text_generation.max_retries cannot be negative")
        if float(resolved.get("retry_backoff_seconds", 2)) < 0:
            raise ValueError("text_generation.retry_backoff_seconds cannot be negative")
        if float(resolved.get("retry_max_backoff_seconds", 30)) < 0:
            raise ValueError("text_generation.retry_max_backoff_seconds cannot be negative")


# 可安全重试的瞬时 HTTP 状态码（超时/限流/网关类）。 / HTTP statuses safe to retry (timeout / rate-limit / gateway class).
TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _retry_delay(resolved: dict, failed_attempt: int,
                 retry_after: str | None = None) -> float:
    """计算指数退避延迟，并尊重服务器 Retry-After 头。 / Compute exponential backoff delay, honouring the server's Retry-After header."""
    base = float(resolved.get("retry_backoff_seconds", 2))
    maximum = float(resolved.get("retry_max_backoff_seconds", 30))
    delay = base * (2 ** max(0, failed_attempt - 1))
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except ValueError:
            pass
    return min(maximum, delay) if maximum > 0 else 0.0


def _wait_before_retry(resolved: dict, *, failed_attempt: int,
                       total_attempts: int, reason: str,
                       retry_after: str | None = None) -> None:
    """记录瞬时失败日志并按退避延迟休眠。 / Log a transient failure and sleep for the backoff delay."""
    delay = _retry_delay(resolved, failed_attempt, retry_after)
    logger.warning(
        "text LLM request transient failure attempt=%d/%d reason=%s "
        "retry_in=%.1fs next_attempt=%d/%d",
        failed_attempt, total_attempts, reason, delay,
        failed_attempt + 1, total_attempts,
    )
    if delay:
        time.sleep(delay)


def _is_transient_url_error(exc: urllib.error.URLError) -> bool:
    """区分可重试的网络错误与证书配置错误（后者重试无意义）。 / Separate retryable network errors from certificate misconfiguration (retrying which never helps)."""
    reason = exc.reason
    if isinstance(reason, ssl.SSLCertVerificationError):
        return False
    return isinstance(reason, (TimeoutError, ConnectionError, OSError, ssl.SSLError))


def _openai_compatible_request(config: dict, prompt: str) -> str:
    """调用 OpenAI 兼容的 /chat/completions 接口，带限流退避重试。 / POST to an OpenAI-compatible /chat/completions endpoint with backoff retries."""
    resolved = _openai_compatible_config(config)
    endpoint = resolved["endpoint"]
    model = resolved["model"]
    key_env = resolved["api_key_env"]
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise RuntimeError(f"environment variable {key_env} is not set")
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON. Do not use Markdown fences."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(resolved.get("temperature", 0.9)),
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint + "/chat/completions", data=payload,
                                     headers=headers, method="POST")
    timeout = float(resolved.get("timeout_seconds", 180))
    max_retries = max(0, int(resolved.get("max_retries", 4)))
    total_attempts = max_retries + 1
    # 瞬时失败（限流/网关/超时）按指数退避重试，永久失败立即抛出并附诊断提示。
    # Transient failures (rate-limit / gateway / timeout) retry with backoff; permanent ones raise immediately with hints.
    for attempt in range(1, total_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            return str(raw["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            if exc.code in TRANSIENT_HTTP_STATUSES and attempt < total_attempts:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                _wait_before_retry(
                    resolved, failed_attempt=attempt, total_attempts=total_attempts,
                    reason=f"HTTP_{exc.code}", retry_after=retry_after,
                )
                continue
            try:
                detail = exc.read(2048).decode("utf-8", errors="replace").strip()
            except Exception:
                detail = ""
            suffix = f": {detail}" if detail else ""
            hint = " Check that the API key belongs to this endpoint and plan." \
                if exc.code in {401, 403} else ""
            raise RuntimeError(
                f"text LLM request failed with HTTP {exc.code}{suffix}.{hint}"
            ) from None
        except urllib.error.URLError as exc:
            if _is_transient_url_error(exc) and attempt < total_attempts:
                _wait_before_retry(
                    resolved, failed_attempt=attempt, total_attempts=total_attempts,
                    reason=type(exc.reason).__name__,
                )
                continue
            raise RuntimeError(
                "text LLM request failed before receiving an HTTP response after "
                f"{attempt} attempt(s): {exc.reason}. Check text_generation.endpoint "
                "and the server's HTTPS_PROXY/NO_PROXY settings. Saved text batches "
                "can be resumed by running the same command again."
            ) from None
        except ssl.SSLCertVerificationError as exc:
            raise RuntimeError(
                f"text LLM TLS certificate verification failed: {exc}"
            ) from None
        except (TimeoutError, ConnectionError, ssl.SSLError) as exc:
            if attempt < total_attempts:
                _wait_before_retry(
                    resolved, failed_attempt=attempt, total_attempts=total_attempts,
                    reason=type(exc).__name__,
                )
                continue
            raise RuntimeError(
                f"text LLM request failed after {attempt} attempt(s): {exc}. "
                "Saved text batches can be resumed by running the same command again."
            ) from None
    raise AssertionError("text LLM retry loop ended unexpectedly")


def _parse_llm_rows(content: str, language: str) -> list[GeneratedText]:
    """解析 LLM 返回的 JSON 数组（容忍 Markdown 围栏）。 / Parse the LLM's JSON array response, tolerating Markdown fences."""
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    raw = json.loads(content)
    if not isinstance(raw, list):
        raise ValueError("LLM text response must be a JSON array")
    result = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            raise ValueError(f"LLM text response item {index} has no text")
        result.append(GeneratedText(
            text=str(item["text"]).strip(), language=language,
            category=str(item.get("category", "llm")).strip() or "llm",
            source="openai_compatible",
        ))
    return result


def _request_batch_size(config: dict) -> int:
    """返回单次请求条数，且不参与语料指纹计算。 / Return the operational request size without changing corpus identity."""
    value = config.get("request_batch_size")
    if value is None:
        value = config.get("batch_size", 50)
    return max(1, int(value))


def _llm_rows(language: str, language_name: str, total: int, config: dict,
              requester, *, on_batch=None, round_offset: int = 0) -> list[GeneratedText]:
    """按批向 LLM 请求语料直到凑足条数，每批经 on_batch 持久化。 / Request corpus rows from the LLM in batches until the count is met, persisting each batch via on_batch."""
    batch_size = _request_batch_size(config)
    rows = []
    round_index = 0
    while len(rows) < total and round_index < int(config.get("max_rounds", 100)):
        count = min(batch_size, total - len(rows))
        prompt = (
            f"Create {count} unique, natural TTS training sentences directly in {language_name} "
            f"(language code {language}). Cover daily speech, questions, numbers, dates, times, "
            "money, names, places, and longer sentences. Avoid translations, personal data, "
            "unsafe content, and duplicated wording. Return a JSON array of objects with exactly "
            "two fields: text and category."
        )
        logger.info(
            "LLM text request language=%s round=%d count=%d",
            language, round_offset + round_index + 1, count,
        )
        batch = _parse_llm_rows(requester(config, prompt), language)
        rows.extend(batch)
        # 每批成功后立即回调，保证断点可续传。 / Callback right after each batch so progress is resumable.
        if on_batch is not None:
            on_batch(batch)
        round_index += 1
    return rows


def _script_matches(text: str, language: str) -> bool:
    """用 Unicode 区块检测文本是否为纯目标语言书写，用于拒绝混码文本。 / Check via Unicode blocks that text is written purely in the target script, to reject mixed-language text."""
    if language == "zh":
        return bool(re.search(r"[\u3400-\u9fff]", text))
    if language == "ja":
        return bool(re.search(r"[\u3040-\u30ff]", text))
    if language == "ko":
        return bool(re.search(r"[\uac00-\ud7af]", text))
    if language == "ru":
        return bool(re.search(r"[\u0400-\u04ff]", text))
    if language in {"en", "de", "fr", "pt", "es", "it"}:
        # 拉丁字母语言：需含拉丁字母且不含 CJK/西里尔字符。 / Latin-script languages: require Latin letters and exclude CJK/Cyrillic.
        return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text)) and not bool(
            re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]", text)
        )
    # 未知语言默认放行，交由 G2P 前端兜底校验。 / Unknown languages pass through; the G2P frontend acts as the last line of defence.
    return True


def _file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256，用于语料指纹中的输入溯源。 / Stream-compute a file's SHA-256 for input provenance in corpus fingerprints."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_payload(config: dict, layout, *, legacy: bool = False) -> dict:
    """构建语义语料身份载荷，剔除不影响语料内容的执行调度键。 / Build the semantic corpus identity payload, excluding execution-scheduling keys."""
    if legacy:
        # 精确复现 v1 哈希：request_batch_size 在 v1 不存在，不能阻碍旧断点迁移。
        # Reproduce the v1 hash exactly. request_batch_size did not exist in v1
        # and must not prevent migration of an older checkpoint.
        operational = {
            "enabled", "output", "root", "corpus_name", "reuse", "overwrite",
            "request_batch_size", "max_retries", "retry_backoff_seconds",
            "retry_max_backoff_seconds",
        }
    else:
        operational = {
            "enabled", "output", "root", "corpus_name", "reuse", "overwrite",
            "batch_size", "request_batch_size", "timeout_seconds", "max_rounds",
            "refill_rounds", "allow_fewer", "api_key_env", "max_retries",
            "retry_backoff_seconds", "retry_max_backoff_seconds",
        }

    # 递归剥离运行参数与 _comment 注释键，只留下影响语料语义的配置。 / Recursively strip operational and _comment keys, keeping only semantics-affecting config.
    def strip_operational(value):
        if isinstance(value, dict):
            return {
                key: strip_operational(item)
                for key, item in value.items()
                if not str(key).startswith("_comment") and key not in operational
            }
        if isinstance(value, list):
            return [strip_operational(item) for item in value]
        return value

    generation_config = strip_operational(config)
    input_value = generation_config.get("input")
    if input_value:
        # file 输入以「绝对路径 + 内容哈希」参与指纹，内容不变即可复用。 / file inputs enter the fingerprint as (absolute path, content hash) so unchanged content stays reusable.
        input_path = Path(input_value).expanduser().resolve()
        generation_config["input"] = {
            "path": str(input_path),
            "sha256": _file_sha256(input_path) if input_path.is_file() else None,
        }
    return {
        "format": 1 if legacy else 2,
        "languages": sorted(layout.languages),
        "generation": generation_config,
    }


def _identity_from_payload(config: dict, layout, payload: dict) -> tuple[str, str]:
    """由身份载荷算出语料 ID 与指纹（未命名时用指纹片段生成 ID）。 / Derive the corpus ID and fingerprint from the identity payload (auto-naming from a fingerprint slice)."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    configured_name = config.get("corpus_name")
    if configured_name:
        corpus_id = str(configured_name).strip()
        if not CORPUS_NAME.fullmatch(corpus_id):
            raise ValueError(
                "text_generation.corpus_name must contain only letters, numbers, '.', '_' "
                "and '-', and cannot start with punctuation"
            )
    else:
        provider = str(config.get("provider", "builtin"))
        language_slug = "-".join(sorted(layout.languages))
        corpus_id = f"{provider}-{language_slug}-{fingerprint[:12]}"
    return corpus_id, fingerprint


def _payload_fingerprint(payload: dict) -> str:
    """对载荷做规范化 JSON 编码后取 SHA-256。 / Canonical-JSON encode the payload and hash it with SHA-256."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _corpus_identity(config: dict, layout) -> tuple[str, str]:
    """返回与模型无关的语义语料 ID 与指纹。 / Return a model-independent semantic corpus ID and fingerprint."""
    return _identity_from_payload(config, layout, _identity_payload(config, layout))


def _legacy_corpus_identity(config: dict, layout) -> tuple[str, str]:
    """按 v1 规则计算旧身份，用于定位并迁移旧缓存。 / Compute the v1 identity to locate and migrate legacy caches."""
    return _identity_from_payload(
        config, layout, _identity_payload(config, layout, legacy=True),
    )


def _corpus_family_fingerprint(config: dict, layout) -> str:
    """识别兼容文本池，忽略语言与条数选择差异。 / Identify compatible text pools while ignoring language/count selection."""
    payload = _identity_payload(config, layout)
    payload["format"] = "text-family-v1"
    payload.pop("languages", None)
    payload["generation"].pop("sentences_per_language", None)
    return _payload_fingerprint(payload)


def _selection_fingerprint(config: dict, languages: list[str], target: int) -> str:
    """按当前选择重建 v2 语料指纹，用于安全发现旧语料池。 / Recreate a v2 corpus fingerprint for safe legacy-pool discovery."""
    selected = deepcopy(config)
    selected["sentences_per_language"] = target
    layout = SimpleNamespace(languages=tuple(languages))
    return _payload_fingerprint(_identity_payload(selected, layout))


def _corpus_paths(config: dict, layout, corpus_id: str) -> tuple[Path, Path]:
    """解析语料 CSV 与报告 JSON 的输出路径。 / Resolve output paths for the corpus CSV and report JSON."""
    output_value = config.get("output")
    if output_value:
        output = Path(output_value)
    else:
        root = Path(config.get("root") or layout.dataset_dir.parent / "text_corpora")
        output = root / corpus_id / "texts.csv"
    return output, output.with_suffix(".report.json")


def _partial_corpus_path(output: Path) -> Path:
    """断点续传检查点文件（JSONL）路径。 / Path of the resumable checkpoint file (JSONL)."""
    return output.with_suffix(".partial.jsonl")


def _compatible_corpus_rows(config: dict, layout, output: Path, target: int,
                            family_fingerprint: str) -> list[GeneratedText]:
    """从兼容的更大规模/多语言语料中复用已采纳的行。 / Reuse accepted rows from compatible larger/multilingual corpora."""
    if config.get("output"):
        roots = [output.parent]
    else:
        roots = [Path(config.get("root") or layout.dataset_dir.parent / "text_corpora")]
    compatible = []
    for root in roots:
        for report_path in root.glob("*/texts.report.json"):
            candidate_output = report_path.with_name("texts.csv")
            if candidate_output.resolve() == output.resolve() or not candidate_output.is_file():
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                languages = [str(value) for value in report.get("languages", [])]
                candidate_target = int(report.get("target_per_language", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if report.get("provider") != config.get("provider", "builtin"):
                continue
            # 兼容判定：族指纹一致，或旧 v2 语料的选择指纹精确匹配。 / Compatible iff the family fingerprint matches, or a legacy v2 corpus's selection fingerprint matches exactly.
            family_matches = report.get("family_fingerprint") == family_fingerprint
            legacy_matches = (
                report.get("identity_format") == 2
                and report.get("fingerprint")
                == _selection_fingerprint(config, languages, candidate_target)
            )
            if not family_matches and not legacy_matches:
                continue
            accepted = report.get("accepted", {})
            coverage = sum(
                min(int(accepted.get(language, 0)), target)
                for language in layout.languages
            )
            if coverage:
                compatible.append((coverage, candidate_output))
    # 覆盖率高的候选优先，减少后续 LLM 补齐量。 / Prefer higher-coverage candidates to shrink the LLM top-up.
    compatible.sort(key=lambda item: item[0], reverse=True)
    rows = []
    counts = Counter()
    seen = set()
    sources = []
    for _, candidate_output in compatible:
        used = 0
        for row in _file_rows(candidate_output, layout.language_specs):
            # 归一化 + casefold 后再去重，跨候选语料防止重复文本。 / Dedupe after normalize + casefold across candidate corpora.
            key = (row.language, normalize(row.text, row.language).casefold())
            if counts[row.language] >= target or key in seen:
                continue
            seen.add(key)
            counts[row.language] += 1
            rows.append(row)
            used += 1
        if used:
            sources.append(str(candidate_output))
        if all(counts[language] >= target for language in layout.languages):
            break
    if rows:
        logger.info(
            "TEXT REUSE | reused=%d | counts=%s | compatible_corpora=%d | sources=%s",
            len(rows), dict(counts), len(sources), ";".join(sources),
            extra={"tts_style": "success"},
        )
    return rows


def _rewrite_partial_fingerprint(path: Path, old_fingerprint: str,
                                 new_fingerprint: str) -> bool:
    """就地改写断点头中的指纹，使旧检查点能在新身份下续传。 / Rewrite the checkpoint header fingerprint in place so an old checkpoint resumes under the new identity."""
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        return False
    header = json.loads(lines[0])
    checkpoint = header.get("_checkpoint", {}) if isinstance(header, dict) else {}
    if checkpoint.get("fingerprint") != old_fingerprint:
        return False
    checkpoint["fingerprint"] = new_fingerprint
    checkpoint["identity_format"] = 2
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(header, ensure_ascii=False) + "\n" + "".join(lines[1:]),
        encoding="utf-8",
    )
    temporary.replace(path)
    return True


def _migrate_legacy_corpus(config: dict, layout, *, corpus_id: str,
                           fingerprint: str, legacy_id: str,
                           legacy_fingerprint: str) -> bool:
    """把 v1 缓存无损迁移到语义 v2 身份。 / Move a v1 cache to the semantic v2 identity without losing progress."""
    if fingerprint == legacy_fingerprint:
        return False
    output, report_path = _corpus_paths(config, layout, corpus_id)
    partial_path = _partial_corpus_path(output)
    old_output, old_report = _corpus_paths(config, layout, legacy_id)
    old_partial = _partial_corpus_path(old_output)
    same_paths = old_output == output
    old_paths = (old_output, old_report, old_partial)
    new_paths = (output, report_path, partial_path)
    if not any(path.is_file() for path in old_paths):
        return False
    if not same_paths and any(path.is_file() for path in new_paths):
        logger.warning(
            "legacy corpus migration skipped because target already exists old=%s new=%s",
            old_output.parent, output.parent,
        )
        return False
    if not same_paths:
        output.parent.mkdir(parents=True, exist_ok=True)
        for source, destination in zip(old_paths, new_paths):
            if source.is_file():
                source.replace(destination)
    migrated = _rewrite_partial_fingerprint(
        partial_path, legacy_fingerprint, fingerprint,
    )
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("fingerprint") == legacy_fingerprint:
            report.update({
                "corpus_id": corpus_id,
                "fingerprint": fingerprint,
                "identity_format": 2,
                "migrated_from": legacy_id,
                "output": str(output.resolve()),
            })
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            temporary.replace(report_path)
            migrated = True
    if migrated and not same_paths:
        try:
            old_output.parent.rmdir()
        except OSError:
            pass
    if migrated:
        logger.info(
            "text corpus identity migrated format=1->2 old=%s new=%s",
            legacy_id, corpus_id,
        )
    return migrated


def _load_partial_rows(path: Path, fingerprint: str) -> list[GeneratedText]:
    """加载被中断的 LLM 生成留下的请求级进度。 / Load request-level progress left by an interrupted LLM generation."""
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
        try:
            header = json.loads(first)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid text generation checkpoint header: {path}") from exc
        checkpoint = header.get("_checkpoint", {}) if isinstance(header, dict) else {}
        if checkpoint.get("format") != 1 or checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"text generation checkpoint {path} belongs to different settings; "
                "delete it or set text_generation.overwrite=true"
            )
        for line_number, line in enumerate(stream, 2):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            # 最后一行可能是进程被杀时的半截写入，容忍并终止加载。 / The final line may be a torn write from a killed process; tolerate and stop.
            except json.JSONDecodeError:
                logger.warning(
                    "ignoring incomplete final text checkpoint row path=%s line=%d",
                    path, line_number,
                )
                break
            rows.append(GeneratedText(
                text=str(item["text"]), language=str(item["language"]),
                category=str(item.get("category", "llm")),
                source=str(item.get("source", "openai_compatible")),
            ))
    return rows


def _append_partial_rows(path: Path, fingerprint: str,
                         rows: list[GeneratedText]) -> None:
    """在下一次请求前把一批成功的 LLM 响应持久化追加。 / Durably append one successful LLM response before the next request."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # 首次写入用临时文件落盘头部，避免崩溃留下无头 JSONL。 / Write the header via a temp file first so a crash never leaves a headless JSONL.
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "_checkpoint": {
                "format": 1, "identity_format": 2, "fingerprint": fingerprint,
            },
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({
                "text": row.text, "language": row.language,
                "category": row.category, "source": row.source,
            }, ensure_ascii=False) + "\n")
        # fsync 确保断电后已确认的批次不丢失。 / fsync so acknowledged batches survive power loss.
        stream.flush()
        os.fsync(stream.fileno())


def text_corpus_path(config: dict, layout, *, voice_id: str | None = None) -> Path:
    """只解析共享语料路径，不触发生成或写入。 / Resolve the shared corpus path without generating or mutating it."""
    if voice_id and not any(config.get(key) for key in ("output", "root", "corpus_name")):
        return layout.dataset_dir.parent / "voices" / voice_id / "texts.csv"
    corpus_id, _ = _corpus_identity(config, layout)
    output, _ = _corpus_paths(config, layout, corpus_id)
    return output


def _cached_corpus_status(output: Path, report_path: Path, fingerprint: str,
                          languages: tuple[str, ...], target: int) -> str:
    """检查已缓存语料状态：missing / partial / complete。 / Classify a cached corpus as missing / partial / complete."""
    if not output.is_file() and not report_path.is_file():
        return "missing"
    if not output.is_file() or not report_path.is_file():
        raise RuntimeError(
            f"shared text corpus cache is incomplete at {output.parent}; "
            "delete it or set text_generation.overwrite=true"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"shared text corpus {output} was created with different settings; "
            "change text_generation.corpus_name or set text_generation.overwrite=true"
        )
    accepted = report.get("accepted", {})
    missing = [language for language in languages if int(accepted.get(language, 0)) < target]
    if missing:
        return "partial"
    return "complete"


def _write_text_pool(path: Path, rows: list[GeneratedText]) -> None:
    """以临时文件原子写出文本池 CSV。 / Atomically write the text pool CSV via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["id", "text", "language", "category", "source"],
        )
        writer.writeheader()
        serials = Counter()
        for row in rows:
            serials[row.language] += 1
            writer.writerow({
                "id": f"{row.language}_{serials[row.language]:07d}",
                "text": row.text,
                "language": row.language,
                "category": row.category,
                "source": row.source,
            })
    temporary.replace(path)


def _voice_text_pool(
    raw: dict, layout, config: dict, voice_id: str, requester,
) -> Path:
    """维护以公开音色 ID 命名的、只增不减的多语言文本池。 / Grow one append-only multilingual text pool owned by a public voice ID."""
    provider = str(config.get("provider", "builtin"))
    target = int(config.get("sentences_per_language", 100))
    reuse = bool(config.get("reuse", True))
    overwrite = bool(config.get("overwrite", False))
    output = layout.dataset_dir.parent / "voices" / voice_id / "texts.csv"
    report_path = output.with_suffix(".report.json")
    family_fingerprint = _corpus_family_fingerprint(config, layout)
    partial_path = output.with_suffix(".partial.jsonl")

    existing = []
    if reuse and not overwrite and output.is_file():
        existing = _file_rows(output, layout.language_registry)
    elif overwrite or not reuse:
        partial_path.unlink(missing_ok=True)
    reusable = []
    if reuse and not overwrite:
        reusable = _compatible_corpus_rows(
            config, layout, output, target, family_fingerprint,
        )

    # 保留该音色已拥有的全部语言；只有本次配置的语言子集参与生成。
    # Preserve every language already owned by this voice. Only the configured
    # subset participates in this run and may receive new rows.
    pool = []
    seen = set()
    pool_counts = Counter()
    for row in (*existing, *reusable):
        text = normalize(row.text, row.language)
        key = (row.language, text.casefold())
        if not text or key in seen:
            continue
        seen.add(key)
        pool.append(GeneratedText(text, row.language, row.category, row.source))
        pool_counts[row.language] += 1
    missing = {
        language: max(target - pool_counts[language], 0)
        for language in layout.languages
    }
    logger.info(
        "VOICE TEXT PLAN | voice_id=%s | target=%d/language | existing=%s | missing=%s | output=%s",
        voice_id, target, dict(sorted(pool_counts.items())),
        {key: value for key, value in missing.items() if value}, output,
    )
    imported = len(pool) > len(existing)
    if not any(missing.values()) and not imported:
        partial_path.unlink(missing_ok=True)
        logger.info(
            "VOICE TEXT CACHE HIT | voice_id=%s | selected=%d | pool=%d",
            voice_id, target * len(layout.languages), len(pool),
            extra={"tts_style": "success"},
        )
        return output

    filters = config.get("filters", {})
    min_chars = int(filters.get("min_characters", 5))
    max_chars = int(filters.get("max_characters", 180))
    reject_mixed = bool(filters.get("reject_mixed_language", True))
    require_g2p = bool(filters.get("require_g2p_pass", False))
    frontend = frontend_from_config(
        raw.get("frontend"), languages=layout.languages,
        language_registry=raw.get("language_registry"),
    ) if require_g2p else None
    rejected = Counter()
    rejected_examples = []

    def accept(rows: list[GeneratedText]) -> None:
        """把候选行过滤后并入池：长度、去重、书写系统、可选 G2P 校验。 / Filter candidate rows into the pool: length, dedupe, script, and optional G2P check."""
        for row in rows:
            if row.language not in layout.language_specs \
                    or pool_counts[row.language] >= target:
                continue
            text = normalize(row.text, row.language)
            key = (row.language, text.casefold())
            # 拒因优先级：长度 → 重复 → 混码 → G2P 失败。 / Rejection precedence: length → duplicate → mixed script → G2P failure.
            reason = None
            if not text or len(text) < min_chars or len(text) > max_chars:
                reason = "length"
            elif bool(filters.get("deduplicate", True)) and key in seen:
                reason = "duplicate"
            elif reject_mixed and not _script_matches(text, row.language):
                reason = "script"
            elif frontend is not None:
                try:
                    frontend.phonemize(text, row.language)
                except Exception as exc:
                    reason = "g2p"
                    if len(rejected_examples) < 20:
                        rejected_examples.append({
                            "reason": reason, "language": row.language,
                            "text": text, "error": str(exc),
                        })
            if reason:
                rejected[reason] += 1
                if len(rejected_examples) < 20 and reason != "g2p":
                    rejected_examples.append({
                        "reason": reason, "language": row.language, "text": text,
                    })
                continue
            seen.add(key)
            pool.append(GeneratedText(text, row.language, row.category, row.source))
            pool_counts[row.language] += 1

    if provider == "builtin":
        for language in layout.languages:
            if missing[language]:
                accept(_builtin_rows(language, target, config))
    elif provider == "file":
        input_path = config.get("input")
        if not input_path:
            raise ValueError("text_generation provider=file requires input")
        accept(_file_rows(input_path, layout.language_specs))
    elif provider == "openai_compatible":
        request_config = _openai_compatible_config(config)
        batch_size = _request_batch_size(config)
        logger.info(
            "LLM request plan batch_size=%d timeout_seconds=%g max_retries=%d",
            batch_size, float(request_config.get("timeout_seconds", 180)),
            int(request_config.get("max_retries", 4)),
        )
        try:
            pending = _load_partial_rows(partial_path, family_fingerprint) if reuse else []
        # 配置变化导致检查点不匹配时重置而非报错，音色池可继续生成。 / Reset rather than fail when settings changed; the voice pool can regenerate.
        except RuntimeError:
            logger.warning(
                "VOICE TEXT CHECKPOINT RESET | voice_id=%s | reason=settings_changed",
                voice_id,
            )
            partial_path.unlink(missing_ok=True)
            pending = []
        accept(pending)

        def save_request_batch(batch: list[GeneratedText]) -> None:
            _append_partial_rows(partial_path, family_fingerprint, batch)

        for language, spec in layout.language_specs.items():
            count = max(target - pool_counts[language], 0)
            if count:
                # round_offset 延续日志轮次编号，使续传日志连续。 / round_offset keeps log round numbering continuous across resumes.
                accept(_llm_rows(
                    language, spec.name, count, config, requester,
                    on_batch=save_request_batch,
                    round_offset=pool_counts[language] // batch_size,
                ))
        for refill_round in range(max(0, int(config.get("refill_rounds", 5)))):
            remaining = {
                language: target - pool_counts[language]
                for language in layout.languages if pool_counts[language] < target
            }
            if not remaining:
                break
            for language, count in remaining.items():
                # 过滤会淘汰部分候选，按缺口的 2 倍超额请求以提高一次命中率。 / Filtering rejects some rows; over-request ~2x the gap to fill in one pass.
                request_count = max(count, min(batch_size, max(4, count * 2)))
                logger.info(
                    "LLM text refill language=%s round=%d missing=%d requesting=%d",
                    language, refill_round + 1, count, request_count,
                )
                accept(_llm_rows(
                    language, layout.language_specs[language].name,
                    request_count, config, requester,
                    on_batch=save_request_batch,
                ))
    else:
        raise ValueError("text_generation.provider must be builtin, file, or openai_compatible")

    _write_text_pool(output, pool)
    missing = {
        language: target - pool_counts[language]
        for language in layout.languages if pool_counts[language] < target
    }
    report = {
        "format": 2,
        "storage": "voice-id",
        "voice_id": voice_id,
        "provider": provider,
        "output": str(output.resolve()),
        "requested_languages": list(layout.languages),
        "target_per_language": target,
        "accepted": dict(sorted(pool_counts.items())),
        "rejected": dict(sorted(rejected.items())),
        "rejected_examples": rejected_examples,
        "generation_family": family_fingerprint,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    partial_path.unlink(missing_ok=True)
    logger.info(
        "VOICE TEXT DONE | voice_id=%s | pool=%d | counts=%s | added=%d",
        voice_id, len(pool), dict(sorted(pool_counts.items())), len(pool) - len(existing),
        extra={"tts_style": "success"},
    )
    if missing and not bool(config.get("allow_fewer", False)):
        raise RuntimeError(
            "text generation did not reach target counts: "
            + ", ".join(f"{language} missing {count}" for language, count in missing.items())
            + f"; inspect {report_path} or set dataset.text.allow_fewer=true"
        )
    return output


def generate_texts(config_path: str | Path, *, requester=_openai_compatible_request) -> Path:
    """主入口：按实验配置生成多语言语料并返回 texts.csv 路径，支持缓存命中与断点续传。 / Entry point: generate the multilingual corpus per experiment config and return the texts.csv path, with cache hits and resumable checkpoints."""
    raw, layout = resolve_experiment(config_path)
    configure_logging_from_config(raw)
    prepare_experiment(layout, raw, config_path)
    config = raw.get("text_generation", {})
    if not config.get("enabled", False):
        raise ValueError("text generation is disabled in this config")
    validate_text_generation_config(config)
    voice_id = str(
        (raw.get("generation", {}).get("voice") or {}).get("id") or ""
    ).strip()
    if voice_id and not CORPUS_NAME.fullmatch(voice_id):
        raise ValueError(
            "dataset.voice.id must contain only letters, numbers, '.', '_' and '-', "
            "and cannot start with punctuation"
        )
    uses_default_storage = not any(
        config.get(key) for key in ("output", "root", "corpus_name")
    )
    # 指定音色 ID 且未自定义存储时，走只增不减的音色文本池。 / With a voice ID and default storage, use the append-only voice text pool.
    if voice_id and uses_default_storage:
        return _voice_text_pool(raw, layout, config, voice_id, requester)
    provider = str(config.get("provider", "builtin"))
    total = int(config.get("sentences_per_language", 100))
    corpus_id, fingerprint = _corpus_identity(config, layout)
    family_fingerprint = _corpus_family_fingerprint(config, layout)
    legacy_id, legacy_fingerprint = _legacy_corpus_identity(config, layout)
    reuse = bool(config.get("reuse", True))
    overwrite = bool(config.get("overwrite", False))
    if reuse and not overwrite:
        _migrate_legacy_corpus(
            config, layout, corpus_id=corpus_id, fingerprint=fingerprint,
            legacy_id=legacy_id, legacy_fingerprint=legacy_fingerprint,
        )
    output, report_path = _corpus_paths(config, layout, corpus_id)
    partial_path = _partial_corpus_path(output)
    logger.info(
        "text corpus requested corpus=%s model=%s provider=%s languages=%s target_per_language=%d",
        corpus_id, layout.name, provider, ",".join(layout.languages), total,
    )
    if provider == "openai_compatible":
        request_batch_size = _request_batch_size(config)
        request_config = _openai_compatible_config(config)
        estimated_requests = len(layout.languages) * (
            (total + request_batch_size - 1) // request_batch_size
        )
        logger.info(
            "LLM request plan batch_size=%d estimated_min_requests=%d "
            "target_sentences=%d timeout_seconds=%g max_retries=%d "
            "retry_backoff_seconds=%g retry_max_backoff_seconds=%g",
            request_batch_size, estimated_requests, total * len(layout.languages),
            float(request_config.get("timeout_seconds", 180)),
            int(request_config.get("max_retries", 4)),
            float(request_config.get("retry_backoff_seconds", 2)),
            float(request_config.get("retry_max_backoff_seconds", 30)),
        )
    if overwrite or not reuse:
        partial_path.unlink(missing_ok=True)
    cache_status = "missing"
    if reuse and not overwrite:
        cache_status = _cached_corpus_status(
            output, report_path, fingerprint, layout.languages, total,
        )
    if cache_status == "complete":
        partial_path.unlink(missing_ok=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("family_fingerprint") != family_fingerprint:
            report["family_fingerprint"] = family_fingerprint
            temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            temporary_report.replace(report_path)
        logger.info("text corpus cache hit corpus=%s output=%s", corpus_id, output)
        return output
    # 非确定性 provider 无法续写部分缓存，只能要求整体重做。 / Non-deterministic providers cannot top up a partial cache; require a full redo.
    if cache_status == "partial" and provider != "openai_compatible":
        raise RuntimeError(
            f"shared text corpus {output} is incomplete; set "
            "text_generation.overwrite=true after fixing the source"
        )
    partial_counts = Counter()
    reusable_candidates = []
    if reuse and not overwrite and cache_status == "missing" \
            and provider == "openai_compatible":
        reusable_candidates = _compatible_corpus_rows(
            config, layout, output, total, family_fingerprint,
        )

    def save_request_batch(batch: list[GeneratedText]) -> None:
        """持久化一批 LLM 响应并记录检查点进度。 / Persist one LLM batch and log checkpoint progress."""
        _append_partial_rows(partial_path, fingerprint, batch)
        partial_counts.update(row.language for row in batch)
        logger.info(
            "text corpus request checkpoint saved corpus=%s batch=%d persisted=%d counts=%s",
            corpus_id, len(batch), sum(partial_counts.values()), dict(partial_counts),
        )

    if cache_status == "partial":
        candidates = _file_rows(output, layout.language_specs)
        logger.info(
            "text corpus partial cache resume corpus=%s accepted=%d",
            corpus_id, len(candidates),
        )
        if provider == "openai_compatible" and reuse:
            pending = _load_partial_rows(partial_path, fingerprint)
            candidates.extend(pending)
            partial_counts.update(row.language for row in pending)
            if pending:
                logger.info(
                    "text corpus request checkpoint resume corpus=%s persisted=%d counts=%s path=%s",
                    corpus_id, len(pending), dict(partial_counts), partial_path,
                )
    elif provider == "builtin":
        candidates = []
        for language in layout.languages:
            logger.info("builtin text generation language=%s count=%d", language, total)
            candidates.extend(_builtin_rows(language, total, config))
    elif provider == "file":
        input_path = config.get("input")
        if not input_path:
            raise ValueError("text_generation provider=file requires input")
        candidates = _file_rows(input_path, layout.language_specs)
        logger.info("file text import input=%s selected=%d", input_path, len(candidates))
    elif provider == "openai_compatible":
        # 先并入兼容语料的复用行与检查点进度，再按缺口补齐。 / Seed with reusable compatible-corpus rows plus checkpoint progress, then top up the gap.
        checkpoint_candidates = _load_partial_rows(partial_path, fingerprint) if reuse else []
        candidates = [*reusable_candidates, *checkpoint_candidates]
        partial_counts.update(row.language for row in checkpoint_candidates)
        if checkpoint_candidates:
            logger.info(
                "text corpus request checkpoint resume corpus=%s persisted=%d counts=%s path=%s",
                corpus_id, len(checkpoint_candidates), dict(partial_counts), partial_path,
            )

        configured_batch = _request_batch_size(config)
        available_counts = Counter(row.language for row in candidates)
        for language, spec in layout.language_specs.items():
            existing = available_counts[language]
            missing_raw = max(total - existing, 0)
            if not missing_raw:
                continue
            candidates.extend(_llm_rows(
                language, spec.name, missing_raw, config, requester,
                on_batch=save_request_batch,
                round_offset=existing // configured_batch,
            ))
    else:
        raise ValueError("text_generation.provider must be builtin, file, or openai_compatible")

    filters = config.get("filters", {})
    min_chars = int(filters.get("min_characters", 5))
    max_chars = int(filters.get("max_characters", 180))
    reject_mixed = bool(filters.get("reject_mixed_language", True))
    require_g2p = bool(filters.get("require_g2p_pass", False))
    frontend = frontend_from_config(
        raw.get("frontend"), languages=layout.languages,
        language_registry=raw.get("language_registry"),
    ) if require_g2p else None
    accepted = []
    seen = set()
    previous_report = json.loads(report_path.read_text(encoding="utf-8")) \
        if cache_status == "partial" else {}
    rejected = Counter(previous_report.get("rejected", {}))
    rejected_examples = list(previous_report.get("rejected_examples", []))[:20]
    per_language = Counter()

    def filter_candidates(rows: list[GeneratedText]) -> None:
        """按语言/长度/去重/书写系统/G2P 过滤候选并计入采纳统计。 / Filter candidates by language/length/dedupe/script/G2P and record acceptance stats."""
        for row in rows:
            if row.language not in layout.language_specs:
                rejected["language"] += 1
                if len(rejected_examples) < 20:
                    rejected_examples.append({
                        "reason": "language", "language": row.language, "text": row.text,
                    })
                continue
            text = normalize(row.text, row.language)
            key = (row.language, text.casefold())
            if not text or len(text) < min_chars or len(text) > max_chars:
                rejected["length"] += 1
                reason = "length"
            elif bool(filters.get("deduplicate", True)) and key in seen:
                rejected["duplicate"] += 1
                reason = "duplicate"
            elif reject_mixed and not _script_matches(text, row.language):
                rejected["script"] += 1
                reason = "script"
            else:
                reason = None
                if frontend is not None:
                    try:
                        frontend.phonemize(text, row.language)
                    except Exception as exc:
                        rejected["g2p"] += 1
                        reason = "g2p"
                        if len(rejected_examples) < 20:
                            rejected_examples.append({
                                "reason": reason, "language": row.language,
                                "text": text, "error": str(exc),
                            })
                        continue
                seen.add(key)
                # 计入 seen 即使超出配额，防止 refill 阶段重复请求同文本。 / Track in seen even past quota so refill rounds never re-request the same text.
                if per_language[row.language] < total:
                    accepted.append(GeneratedText(
                        text, row.language, row.category, row.source,
                    ))
                    per_language[row.language] += 1
            if reason and len(rejected_examples) < 20:
                rejected_examples.append({
                    "reason": reason, "language": row.language, "text": text,
                })

    filter_candidates(candidates)
    if provider == "openai_compatible":
        for refill_round in range(max(0, int(config.get("refill_rounds", 5)))):
            missing = {
                language: total - per_language[language]
                for language in layout.languages if per_language[language] < total
            }
            if not missing:
                break
            # refill：过滤淘汰后仍有缺口时超额补请求，最多 refill_rounds 轮。 / Refill: over-request when filtering left gaps, bounded by refill_rounds.
            refill = []
            configured_batch = _request_batch_size(config)
            for language, count in missing.items():
                # 同音色池：按缺口 2 倍超额请求以提高命中。 / Same as the voice pool: over-request ~2x the gap.
                request_count = max(count, min(configured_batch, max(4, count * 2)))
                logger.info(
                    "LLM text refill language=%s round=%d missing=%d requesting=%d",
                    language, refill_round + 1, count, request_count,
                )
                refill.extend(_llm_rows(
                    language, layout.language_specs[language].name,
                    request_count, config, requester,
                    on_batch=save_request_batch,
                ))
            filter_candidates(refill)

    # 临时文件写满后原子替换，读方永远看到完整 CSV。 / Write to a temp file then atomically replace so readers never see a partial CSV.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "text", "language", "category", "source"])
        writer.writeheader()
        language_serials = Counter()
        for row in accepted:
            language_serials[row.language] += 1
            writer.writerow({
                "id": f"{row.language}_{language_serials[row.language]:07d}",
                "text": row.text,
                "language": row.language,
                "category": row.category,
                "source": row.source,
            })
    temporary.replace(output)
    report = {
        "format": 1,
        "identity_format": 2,
        "corpus_id": corpus_id,
        "fingerprint": fingerprint,
        "family_fingerprint": family_fingerprint,
        "provider": provider,
        "output": str(output.resolve()),
        "languages": list(layout.languages),
        "target_per_language": total,
        "accepted": dict(sorted(per_language.items())),
        "rejected": dict(sorted(rejected.items())),
        "rejected_examples": rejected_examples,
        "seed": int(config.get("seed", 1337)),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    logger.info("text generation completed output=%s accepted=%s rejected=%s", output, dict(per_language), dict(rejected))
    # 未达标且未开启 allow_fewer 时视为失败。 / Below target without allow_fewer is a hard failure.
    missing = {language: total - per_language[language] for language in layout.languages if per_language[language] < total}
    if missing and not bool(config.get("allow_fewer", False)):
        raise RuntimeError(
            "text generation did not reach target counts: "
            + ", ".join(f"{language} missing {count}" for language, count in missing.items())
            + f"; inspect {report_path} or set text_generation.allow_fewer=true"
        )
    return output
