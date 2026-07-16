from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .experiments import prepare_experiment, resolve_experiment
from .frontend import frontend_from_config
from .logging_utils import configure_logging
from .text import normalize


logger = logging.getLogger(__name__)
CORPUS_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

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
    text: str
    language: str
    category: str
    source: str


def _category_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
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
    value = config.get("domain", {}).get("keywords")
    if value is None:
        value = DEFAULT_KEYWORDS.get(language, DEFAULT_KEYWORDS["en"])
    if isinstance(value, dict):
        value = value.get(language) or value.get("default") or DEFAULT_KEYWORDS.get(language, DEFAULT_KEYWORDS["en"])
    result = [str(item).strip() for item in value if str(item).strip()]
    return result or DEFAULT_KEYWORDS.get(language, DEFAULT_KEYWORDS["en"])


def _context(serial: int, language: str, config: dict) -> dict:
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
    source = Path(path)
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
    nested = config.get("openai_compatible") or {}
    if not isinstance(nested, dict):
        raise ValueError("text_generation.openai_compatible must be an object")
    resolved = dict(nested)
    for key in ("endpoint", "base_url", "model", "api_key_env", "temperature",
                "timeout_seconds", "batch_size", "max_rounds"):
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
    provider = str(config.get("provider", "builtin"))
    if provider not in {"builtin", "file", "openai_compatible"}:
        raise ValueError("text_generation.provider must be builtin, file, or openai_compatible")
    if provider == "file" and not config.get("input"):
        raise ValueError("text_generation provider=file requires input")
    if provider == "openai_compatible":
        _openai_compatible_config(config)


def _openai_compatible_request(config: dict, prompt: str) -> str:
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
    try:
        with urllib.request.urlopen(
            request, timeout=float(resolved.get("timeout_seconds", 120)),
        ) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
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
        raise RuntimeError(
            "text LLM request failed before receiving an HTTP response: "
            f"{exc.reason}. Check text_generation.endpoint and the server's "
            "HTTPS_PROXY/NO_PROXY settings."
        ) from None
    return str(raw["choices"][0]["message"]["content"])


def _parse_llm_rows(content: str, language: str) -> list[GeneratedText]:
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


def _llm_rows(language: str, language_name: str, total: int, config: dict,
              requester) -> list[GeneratedText]:
    batch_size = max(1, int(config.get("batch_size", 50)))
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
        logger.info("LLM text request language=%s round=%d count=%d", language, round_index + 1, count)
        rows.extend(_parse_llm_rows(requester(config, prompt), language))
        round_index += 1
    return rows


def _script_matches(text: str, language: str) -> bool:
    if language == "zh":
        return bool(re.search(r"[\u3400-\u9fff]", text))
    if language == "ja":
        return bool(re.search(r"[\u3040-\u30ff]", text))
    if language == "ko":
        return bool(re.search(r"[\uac00-\ud7af]", text))
    if language == "ru":
        return bool(re.search(r"[\u0400-\u04ff]", text))
    if language in {"en", "de", "fr", "pt", "es", "it"}:
        return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text)) and not bool(
            re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]", text)
        )
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_comments(value):
    if isinstance(value, dict):
        return {
            key: _without_comments(item)
            for key, item in value.items()
            if not str(key).startswith("_comment")
        }
    if isinstance(value, list):
        return [_without_comments(item) for item in value]
    return value


def _corpus_identity(config: dict, layout) -> tuple[str, str]:
    """Return a model-independent corpus ID and its reproducibility fingerprint."""
    operational = {"enabled", "output", "root", "corpus_name", "reuse", "overwrite"}
    generation_config = _without_comments({
        key: value for key, value in config.items() if key not in operational
    })
    input_value = generation_config.get("input")
    input_identity = None
    if input_value:
        input_path = Path(input_value).expanduser().resolve()
        input_identity = {
            "path": str(input_path),
            "sha256": _file_sha256(input_path) if input_path.is_file() else None,
        }
        generation_config["input"] = input_identity
    payload = {
        "format": 1,
        "languages": sorted(layout.languages),
        "generation": generation_config,
    }
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


def _corpus_paths(config: dict, layout, corpus_id: str) -> tuple[Path, Path]:
    output_value = config.get("output")
    if output_value:
        output = Path(output_value)
    else:
        root = Path(config.get("root") or layout.dataset_dir.parent / "text_corpora")
        output = root / corpus_id / "texts.csv"
    return output, output.with_suffix(".report.json")


def text_corpus_path(config: dict, layout) -> Path:
    """Resolve the shared corpus path without generating or mutating it."""
    corpus_id, _ = _corpus_identity(config, layout)
    output, _ = _corpus_paths(config, layout, corpus_id)
    return output


def _reuse_cached_corpus(output: Path, report_path: Path, fingerprint: str,
                         languages: tuple[str, ...], target: int) -> bool:
    if not output.is_file() and not report_path.is_file():
        return False
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
        raise RuntimeError(
            f"shared text corpus {output} is incomplete for: {', '.join(missing)}; "
            "set text_generation.overwrite=true after fixing the source"
        )
    return True


def generate_texts(config_path: str | Path, *, requester=_openai_compatible_request) -> Path:
    raw, layout = resolve_experiment(config_path)
    configure_logging(raw.get("logging", {}).get("level", "INFO"))
    prepare_experiment(layout, raw, config_path)
    config = raw.get("text_generation", {})
    if not config.get("enabled", False):
        raise ValueError("text generation is disabled in this config")
    validate_text_generation_config(config)
    provider = str(config.get("provider", "builtin"))
    total = int(config.get("sentences_per_language", 100))
    corpus_id, fingerprint = _corpus_identity(config, layout)
    output, report_path = _corpus_paths(config, layout, corpus_id)
    logger.info(
        "text corpus requested corpus=%s model=%s provider=%s languages=%s target_per_language=%d",
        corpus_id, layout.name, provider, ",".join(layout.languages), total,
    )
    reuse = bool(config.get("reuse", True))
    overwrite = bool(config.get("overwrite", False))
    if reuse and not overwrite and _reuse_cached_corpus(
        output, report_path, fingerprint, layout.languages, total,
    ):
        logger.info("text corpus cache hit corpus=%s output=%s", corpus_id, output)
        return output
    if provider == "builtin":
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
        candidates = [
            row for language, spec in layout.language_specs.items()
            for row in _llm_rows(language, spec.name, total, config, requester)
        ]
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
    rejected = Counter()
    rejected_examples = []
    per_language = Counter()
    for row in candidates:
        if row.language not in layout.language_specs:
            rejected["language"] += 1
            if len(rejected_examples) < 20:
                rejected_examples.append({"reason": "language", "language": row.language, "text": row.text})
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
            if per_language[row.language] < total:
                accepted.append(GeneratedText(text, row.language, row.category, row.source))
                per_language[row.language] += 1
        if reason and len(rejected_examples) < 20:
            rejected_examples.append({"reason": reason, "language": row.language, "text": text})

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
        "corpus_id": corpus_id,
        "fingerprint": fingerprint,
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
    logger.info("text generation completed output=%s accepted=%s rejected=%s", output, dict(per_language), dict(rejected))
    missing = {language: total - per_language[language] for language in layout.languages if per_language[language] < total}
    if missing and not bool(config.get("allow_fewer", False)):
        raise RuntimeError(
            "text generation did not reach target counts: "
            + ", ".join(f"{language} missing {count}" for language, count in missing.items())
            + f"; inspect {report_path} or set text_generation.allow_fewer=true"
        )
    return output
