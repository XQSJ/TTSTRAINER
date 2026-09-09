# 定制词定音方案 / Custom-word pronunciation design

> 目标：让训练者为官方/产品/品牌/人名等特殊词指定正确读音，全程零音素负担、
> 可人工确认、不配置时零影响。 / Goal: let trainers assign exact pronunciations to
> special words (brands, product names, people) with no phoneme burden, a human
> confirmation gate, and zero cost when unconfigured.

---

## 0. 核心事实 / Core constraint

VITS 的输入是**音素序列**，不是音频。所以特殊词必须有音素表示才能被模型读出。
问题不是"要不要音素"，而是**音素由谁定、怎么让人不用懂音素就能定对**。

/ VITS consumes a phoneme sequence, not audio. A special word therefore needs a
phoneme representation to be readable by the model. The question is not whether
phonemes are needed, but who sets them and how a user can set them without
knowing phonemes.

---

## 1. 目标原则 / Principles

1. **零音素负担**：用户从头到尾在"听、选、念"，不碰音素字符串。 / The user only
   listens, picks and speaks — never touches phoneme strings.
2. **强制确认闸门**：所有定制词必须过用户的"确认"才进入训练/导出，不允许自动但
   读错的词漏进生产。 / A mandatory confirmation gate: every customized word must
   pass user approval before training/export; a silently-wrong reading must never slip
   into production.
3. **可选、零侵入**：未配置 `custom_words` 时完全走原流程，不打断任何人。 /
   Optional and non-intrusive: with no `custom_words` configured the pipeline runs
   exactly as before.
4. **全语言覆盖**：机制语言无关，同一套闸门适用于 en/zh/ja/ko/fr/es/pt。 /
   Language-agnostic; the same gate serves every supported language.
5. **跨语言共享词**：英文/品牌词在各语言语境读相似音，靠 code-switching 统一，
   不逐语言单独配。 / Cross-language shared words read similarly across languages
   via code-switching, never configured per language.

---

## 2. 词的两类 / Two word classes

### 2.1 `native_words`（原生词）/ native words

每个语言自己的词，落音到**该语言词典**。 / A language's own words; readings land
in that language's dictionary.

```json
"custom_words": {
  "en": {
    "native_words": {
      "fosi": {},
      "zorp": {}
    }
  }
}
```

### 2.2 `shared_words`（跨语言共享词）/ shared words

写拉丁/英文、各语言混排都读相似音的词（品牌/产品/人名，如 fosi、Google）。
落音到该"基准语言"词典 **+ 自动分发到 code-switching 词表**，覆盖各语言混排语境。
/ Words written in Latin/English that read similarly in every language's mixed text
(brands, product names, people). Their reading lands in the base-language dictionary
**and** is distributed into the code-switching table so every language's mixed
context is covered.

```json
"custom_words": {
  "shared_words": {
    "fosi":      {},
    "kubernetes": {}
  }
}
```

---

## 3. 读音来源三机制 / Three reading sources

| 机制 | 读音怎么定 | 训练者做什么 |
|---|---|---|
| **A. 自动候选** | 基准语言前端（en→g2p）预测读音 | 听，直接判定对错 |
| **B. QwenTTS 试听** | QwenTTS 按当前音色把候选念出来 | 试听，选中符合期望的念法 |
| **C. 录音对齐** | 训练者自己念一遍 → 系统对齐成音素 | 用嘴念/给近似词 |

三个机制都收敛到**确定的音素序列**，进后续闸门。 / All three converge to a
definite phoneme sequence that flows into the gate.

---

## 4. 完整流程 / Flow

```
训练前
  │
  ├─ 配置了 custom_words ？ ── 否 ──▶ 原流程照常（零影响）
  │        │ 是
  │        ▼
  │   阶段① 候选生成：每词生成候选读音 + 参考音频（A/B）
  │        ▼
  │   阶段② 确认闸门：逐词过用户
  │         ✓ ok → 定稿
  │         ✗ 不对 → 调整：B 重生成试听 / C 录音对齐 → 再确认
  │             （可逐词接受"用自动候选"跳过）
  │        ▼ 全绿
  │   阶段③ 分发落点：
  │         native_words → 写该语言词典
  │         shared_words → 写基准语言词典 + 同步 code-switching 词表
  │        ▼
  ├─▶ 训练（可选：把已定稿音素词写进语料，让模型学得更自然）
       ▼
   导出   已定稿音素写进 cmudict_data.json / code-switching 表
       ▼
   部署   词典命中 → 模型读正确音素（兜底不参与）
```

**阶段② 是强制人审闸门**——不全部绿，训练/导出不放行下一段。
/ Stage 2 is the mandatory human gate; nothing downstream starts until it is green.

---

## 5. 配置形态 / Config shape

```json
// 训练配置 train.json 内嵌 custom_words 块（随 preset/extends 合并）
{
  ...,
  "custom_words": {
    "en": {
      "native_words": {
        "fosi": {},
        "zorp": {}
      }
    },
    "ja": {
      "native_words": { "my_product_name": {} }
    },
    "shared_words": {
      "kubernetes": {},
      "google": {}
    }
  }
}
```

`{}` 表示读音待确认（auto 即可）；也可显式给参考：

```json
{ "fosi": { "reference": "foe see" } }        // 发音基准：写一个能读的近似拼写
{ "fosi": { "audio": "refs/fosi.wav" } }       // 录音基准：念一遍
```

产出物：`custom_words_verified.json`（每词已确认音素），训练/导出读取它。

---

## 6. 部署侧 / Deployment side

- 复用现有**词典注入链路**（custom-wordlist / cmudict），不改 AAR。
- `native_words` → 对应语言词典; `shared_words` → 基准语言查读(code-switching)。
- 部署侧 custom-wordlist.json 变成**导出自动产物**，不再是人手填。

---

## 7. 范围与不做的事 / Scope & non-goals

- **做**：custom_words 配置解析、三机制（A/B/C）、确认闸门、分发落点、文档。
- **不做**：不重训模型才生效的"纯部署定制"作为主流程（那是应急，不是正解）；不
  为共享词做逐语言手写转写（用 code-switching 自动分发）。
