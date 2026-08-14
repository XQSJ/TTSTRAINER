# TTSTRAINER

用 Qwen3-TTS 生成训练数据，训练多语言、多音色 VITS，并导出移动端可用的
ONNX 资源。

> **闭源商业 App 推荐：**使用 `"preset": "mobile_commercial"`，七语全部走
> MIT Piper Plus G2P，不把 GPL eSpeak 带入部署链路。旧模型继续使用原来的
> `quality`、`mobile_routed` 或 `mobile`；前端契约不同的 checkpoint 不能 resume。

## 五分钟开始训练

### 1. 安装

```bash
git clone https://github.com/XQSJ/TTSTRAINER.git
cd TTSTRAINER
python3 -m venv .venv
.venv/bin/pip install -U pip setuptools wheel
.venv/bin/pip install -e '.[qwen,export,commercial]'
```

### 2. 复制并修改配置

```bash
cp training_configs/train1.json training_configs/my_model.json
```

普通用户优先只改这些字段：

```json
{
  "task": "train",
  "preset": "mobile_commercial",
  "experiment": {
    "name": "my_model",
    "languages": ["zh", "en", "ja", "ko"],
    "device": "cuda:0"
  },
  "dataset": {
    "sentences_per_language": 2000
  },
  "training": {
    "batch_size": 4,
    "epochs": 200
  }
}
```

音色和文本来源的完整写法见下方“生成音色并直接训练”。

`mobile_commercial` 支持 `zh/en/ja/ko/fr/es/pt`，不会调用 eSpeak。日语由
Piper Plus 内部的 OpenJTalk 后端处理，导出语言包会携带对应字典。若更重视质量而
不在意模型尺寸，将 preset 改为 `quality_commercial`。

已有 eSpeak/OpenJTalk 路由模型不能只改 preset 后续训。请使用新的
`experiment.name` 从头训练；已生成的 WAV 数据仍可通过公共 voice ID 复用。

### 3. 预检并运行

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer language-check \
  --config training_configs/my_model.json

PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/my_model.json
```

同一命令中断后可以重新执行：文本和音频按 `voice_id` 复用，已有内容不会重复生成。
训练输出位于 `runs/my_model/`，移动部署资源位于 `artifacts/my_model/`。

## 完整流程

```text
文本生成/导入 → Qwen 生成或克隆公共音色 WAV → 多语言 G2P
       → VITS 训练与验证 → ONNX 导出 → Android 可组合部署包
```

| 阶段 | 主要目录 |
|---|---|
| 公共音色与训练 WAV | `datasets/voices/<voice_id>/` |
| 当前模型 metadata | `datasets/<model_name>/` |
| checkpoint 与训练日志 | `runs/<model_name>/` |
| ONNX 与移动部署资源 | `artifacts/<model_name>/` |

## 三个配置概念

| 名称 | 属于哪里 | 作用 |
|---|---|---|
| `task` | 自动流水线 | `prepare` 只准备音色数据；`train` 准备数据并训练、导出 |
| `dataset.voices` | 公共音色数据 | 声明一个或多个需要生成/补齐的 `voice_id` 及生成方式 |
| `dataset.speakers` | 某个 VITS 模型 | 把模型内 speaker 名称映射到公共 `voice_id` |

程序不会再通过“配置中有哪些字段”猜测用户想做什么，`task` 是自动入口
`run-pipeline` 的明确执行模式。直接子命令（如 `generate-samples`、`train-vits`）本身
已有明确动作，不依赖推断。旧配置未写 `task` 时默认按 `train` 处理。

`resume`、`warm_start`、`refine_text_prior`、`expand_speakers` 不是第三种 task，它们只描述
`task: "train"` 应如何初始化模型：

```json
"task": "train",
"experiment": {
  "initialization": {
    "mode": "resume",
    "checkpoint": "runs/old_model/checkpoints/last"
  }
}
```

公共音色本身没有 speaker。同一个 `voice_id` 可以在不同模型中分配不同 speaker：

```text
datasets/voices/voice_xiaoling_a/       # 公共音色与 WAV
                  │
                  ├─ 模型一：speaker=gentle
                  └─ 模型二：speaker=assistant
```

项目不会把公共 WAV 复制到每个模型。它只在
`datasets/<model_name>/metadata.csv` 中写入引用路径和模型内 speaker。

按目标选择：

| 目标 | 阅读 |
|---|---|
| 一个配置生成新音色并直接训练 | 用法一 |
| 一次只准备一个或多个公共音色 | 用法二 |
| 使用已有音色训练模型 | 用法三 |

## 安装详解

```bash
git clone https://github.com/XQSJ/TTSTRAINER.git
cd TTSTRAINER

python3 -m venv .venv
.venv/bin/pip install -U pip setuptools wheel
.venv/bin/pip install -e '.[qwen,export,japanese,asian]'
```

不需要执行 `.venv/bin/activate`。如果希望激活虚拟环境，应使用：

```bash
source .venv/bin/activate
```

网络不稳定时可以临时指定镜像：

```bash
.venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple \
  -e '.[qwen,export,japanese,asian]'
```

`flash-attn` 不是必需依赖。没有安装时会自动使用 PyTorch SDPA。

## 用法一：生成音色并直接训练

这是最简单的用法。`voices` 负责生成，`speakers` 负责分配模型内名称：

```bash
cp training_configs/train1.json training_configs/my_model.json
```

```json
{
  "task": "train",
  "preset": "quality",
  "experiment": {
    "name": "my_model",
    "languages": ["zh", "en"],
    "device": "cuda:0"
  },
  "dataset": {
    "sentences_per_language": 2000,
    "speakers": {
      "xiaoling": "voice_xiaoling"
    },
    "text": {
      "provider": "builtin"
    },
    "voices": {
      "voice_xiaoling": {
        "mode": "design",
        "reference_strategy": "cascade",
        "prompt": "A warm, natural young adult female voice with conversational pacing.",
        "reference_text": "你好，这是一段用于创建统一音色的参考录音。",
        "reference_language": "zh"
      }
    }
  },
  "training": {
    "batch_size": 4,
    "epochs": 200
  }
}
```

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer language-check \
  --config training_configs/my_model.json

PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/my_model.json
```

`run-pipeline` 会自动完成文本、Qwen 音频、模型 metadata、音素化、质检、训练和 ONNX
导出。首次验证可以限制为 10 step：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/my_model.json \
  --max-steps 10
```

再次运行同一命令会复用已经完成的文本、音频和模型资源。

示例中的 `builtin` 文本只适合跑通流程；正式模型请换成 OpenAI-compatible 服务或经过
审核的 CSV，见“文本来源”。

## 用法二：一次准备多个公共音色

复制 [prepare-voices.example.json](training_configs/prepare-voices.example.json)。设置
`task: "prepare"`，在 `voices` 中可以同时声明任意数量的新音色；不要填写
`speakers`，因为此时还没有组装 VITS 模型：

```json
{
  "_comment": "一次准备多个公共音色 / Prepare multiple shared voices",
  "task": "prepare",
  "preset": "quality",
  "experiment": {
    "name": "prepare_xiaoling_a",
    "languages": ["zh", "en", "ja", "ko", "fr", "es", "pt"],
    "device": "cuda:1"
  },
  "dataset": {
    "sentences_per_language": 2000,
    "text": {
      "provider": "openai_compatible",
      "endpoint": "https://example.com/v1",
      "model": "your-model",
      "api_key_env": "TEXT_LLM_API_KEY"
    },
    "voices": {
      "voice_xiaoling_a": {
        "mode": "design",
        "reference_strategy": "cascade",
        "prompt": "A warm natural young adult female voice.",
        "reference_text": "你好，这是一段用于创建统一音色的参考录音。",
        "reference_language": "zh"
      },
      "voice_xiaoling_b": {
        "mode": "clone",
        "reference_audio": "voices/xiaoling_b.wav",
        "reference_text": "与参考录音逐字一致的文本",
        "reference_language": "zh"
      }
    }
  }
}
```

一条命令会自动生成或复用文本，然后生成或复用音频：

```bash
export TEXT_LLM_API_KEY='你的实际密钥'

PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/prepare-voices.example.json
```

结果位于：

```text
datasets/voices/
├── voice_xiaoling_a/
│   ├── voice.json
│   ├── texts.csv
│   ├── manifest.csv
│   ├── references/
│   └── wavs/
└── voice_xiaoling_b/
    └── ...

runs/prepare_public_voices/prepared-voices.json
```

`voices` 对象的键就是公共 `voice_id`，不再重复填写 `id`。不同声音必须使用不同键。

`task: "prepare"` 不会创建模型目录 `datasets/<experiment.name>/`、checkpoint、训练
日志或 `artifacts/<experiment.name>/`。`runs/<experiment.name>/` 只保存本次数据准备的
配置、流水线报告和多音色子任务记录，不包含 VITS 模型。真正可复用的数据始终只在
`datasets/voices/<voice_id>/`。

## 用法三：用已有公共音色训练模型

先确保各个 `voice_id` 已准备好，然后复制：

```bash
cp training_configs/multi-speaker.example.json \
  training_configs/my_multi_voice_model.json
```

核心配置只有映射关系：

```json
{
  "task": "train",
  "preset": "quality",
  "experiment": {
    "name": "my_multi_voice_model",
    "languages": ["zh", "en", "ja", "ko", "fr", "es", "pt"],
    "device": "cuda:1"
  },
  "dataset": {
    "sentences_per_language": 2000,
    "speakers": {
      "gentle": "voice_xiaoling_a",
      "bright": "voice_xiaoling_b"
    }
  },
  "training": {
    "batch_size": 4,
    "epochs": 300
  }
}
```

映射方向始终是：

```text
"模型内 speaker 名称": "公共 voice_id"
```

启动：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/my_multi_voice_model.json
```

因为没有 `voices`，这个配置不会调用文本 LLM 或 Qwen。程序会检查两个公共音色是否
满足所选语言和数量，
然后生成当前模型的 metadata 并开始训练。音频仍在公共目录，不会复制。

也可以把“尚未生成的音色”与“已有音色”放在同一份训练配置里：

```json
"dataset": {
  "sentences_per_language": 2000,
  "text": {"provider": "builtin"},
  "voices": {
    "voice_new": {
      "mode": "design",
      "reference_strategy": "cascade",
      "prompt": "A bright and friendly adult voice.",
      "reference_text": "你好，这是新音色的参考录音。",
      "reference_language": "zh"
    }
  },
  "speakers": {
    "gentle": "voice_existing",
    "bright": "voice_new"
  }
}
```

程序只生成/补齐 `voice_new`，直接复用 `voice_existing`，最后把两者组装进当前模型。

## 最常调整的配置

| 字段 | 作用 |
|---|---|
| `task` | `prepare` 只准备数据；`train` 完整训练并导出 |
| `experiment.name` | 模型任务名称，决定 `datasets/runs/artifacts` 目录 |
| `experiment.languages` | 本次使用的语言；顺序决定 language ID |
| `experiment.device` | VITS 和 Qwen 默认设备，例如 `cuda:0` |
| `dataset.sentences_per_language` | 每个音色、每种语言选用的样本数 |
| `dataset.text` | 文本来源，仅生成新音色时需要 |
| `dataset.voices` | 要生成或补齐的公共音色集合，键是 `voice_id` |
| `dataset.speakers` | 当前模型的 `speaker → voice_id` 映射 |
| `dataset.voices.<id>.regenerate` | 可选：按语言重生成参考和/或训练 WAV |
| `dataset.voices.<id>.reference_strategy` | `cascade`（推荐）、`shared` 或 `per_language` |
| `training.batch_size` | 显存不足时优先调小 |
| `training.epochs` | 总训练轮数 |
| `training.stage` | `auto` 自动两阶段；`standard` 始终训练完整声学主链 |
| `training.mixed_precision` | 默认 `fp32`；显式设为 `bf16` 才启用省显存训练 |

移动端需要中日韩正确发音时选择 `preset: "mobile_routed"`：它使用 mobile 的轻量
时长模型，但按语言路由 Piper Plus、OpenJTalk 和 eSpeak。不要再用统一 eSpeak
直接处理日语汉字；预检检测到 `Chinese letter`/`Japanese letter` 回退读法会立即失败。

## 数据复用规则

每个公开 `voice_id` 只有一个目录：

```text
datasets/voices/voice_xiaoling/
├── voice.json
├── texts.csv
├── texts.report.json
├── manifest.csv              # 无 speaker 的公共音频清单
├── references/
│   └── designed.wav
└── wavs/
    ├── zh/
    ├── en/
    └── ...
```

程序启动时先比较“需要什么”和“已经有什么”：

- 已有中英日韩各 2,000 条，本次只训练中文 500 条：不生成，只选择中文前 500 条。
- 已有中文 500 条，本次需要中文 2,000 条：只补 1,500 条。
- 已有中文，本次增加英文：只向同一个目录增加英文文本和 WAV。
- 另一个模型使用相同 `voice_id`：直接复用，不复制数据。
- 减少语言或数量不会删除旧数据，以后仍可恢复使用。

`voice.json` 锁定音色身份，包括 prompt 或参考录音校验值、Qwen 模型、采样率和关键
生成参数。同一个 ID 不能混入另一种音色；要更换声音，请使用新的 `voice_id`。

文本数量和语言不属于音色身份，可以随时增减。文本池会保留曾经生成过的所有语言；
当前模型的 `metadata.csv` 只引用本次配置需要的子集。

老版本的 `<voice_id>/<revision>/` 会在首次使用时迁移到公开 `voice_id` 根目录，并保留
旧 metadata 可继续访问的兼容路径。旧音色目录如果没有无 speaker 的 `manifest.csv`，
首次用于新模型时会从已有 metadata 自动建立，不会重新生成 WAV。

## 文本来源

### 内置模板

```json
"text": {
  "provider": "builtin"
}
```

无需 API，适合烟雾测试和验证训练流程。它的句式变化有限，不建议作为高质量模型的唯一
产品语料。

### OpenAI-compatible 文本接口

```json
"text": {
  "provider": "openai_compatible",
  "endpoint": "https://example.com/v1",
  "model": "your-model",
  "api_key_env": "TEXT_LLM_API_KEY"
}
```

设置密钥：

```bash
export TEXT_LLM_API_KEY='你的实际密钥'
```

`api_key_env` 填的是环境变量名称，不是密钥本身。默认会自动重试网络超时，并在每批请求
完成后保存断点。

### 导入自己的 CSV

```json
"text": {
  "provider": "file",
  "input": "datasets/my_texts.csv"
}
```

CSV 至少包含：

```csv
text,language
你好，欢迎使用语音系统。,zh
Hello, welcome to the speech system.,en
```

无论文本来自哪种方式，产品训练前都应进行版权检查、母语者抽检以及数字、日期、金额和
业务词汇覆盖检查。

## 音色来源

### Prompt 设计音色

```json
"voices": {
  "voice_01": {
    "mode": "design",
    "reference_strategy": "cascade",
    "prompt": "A warm adult voice with natural conversational pacing.",
    "reference_text": "你好，这是一段用于创建统一音色的参考录音。",
    "reference_language": "zh"
  }
}
```

多语言模型推荐 `reference_strategy: "cascade"`。它不是独立设计七次音色，而是：

```text
Prompt + reference_text
  → VoiceDesign 生成唯一主参考 references/designed.wav
  → 用主参考克隆各语言参考 references/localized-<lang>.wav
  → 各语言参考分别生成对应语言训练 WAV
```

这样每个本地化参考都继承同一条主参考的音色，又能使用正确的 Qwen 语言条件，通常比
`per_language` 独立设计多次更能保持音色一致，也比 `shared` 直接跨语言生成更不容易
迁移主参考语言的口音。

三种策略的取舍：

| 策略 | 做法 | 适用情况 |
|---|---|---|
| `cascade` | 一条主参考克隆出各语言参考，再生成训练数据 | 多语言默认推荐，平衡音色一致与本地发音 |
| `shared` | 所有语言直接共用一条参考 | 音色最统一，但其他语言可能带主语言口音 |
| `per_language` | VoiceDesign 为每种语言独立设计参考 | 语言条件独立，但不同参考可能出现明显音色漂移 |

`cascade` 和 `per_language` 默认使用该语言训练文本的第一句作为本地参考文本，也可以
人工指定：

```json
"reference_texts": {
  "fr": "Bonjour, ceci est une référence vocale en français.",
  "es": "Hola, esta es una referencia de voz en español."
}
```

`cascade` 的 `reference_text` 和 `reference_language` 定义唯一主参考；例如主参考为
英文时，英文会直接复用主参考，其他语言由它克隆得到。生成的参考保存在：

```text
datasets/voices/<voice_id>/references/
├── designed.wav
├── localized-zh.wav
├── localized-zh.txt
├── localized-en.wav
├── localized-en.txt
└── ...
```

本地化参考会被缓存，后续增加样本数量时不会重新生成。同一个 `voice_id` 的 Prompt、
主参考和参考策略都属于不可变音色身份；已有音色切换策略时请使用新的 `voice_id`，
避免把不同参考链路混进同一个公共数据集。

### 上传参考录音

复制 [clone.example.json](training_configs/clone.example.json)。关键字段：

```json
"voices": {
  "my_voice": {
    "mode": "clone",
    "reference_audio": "datasets/references/my_voice.wav",
    "reference_text": "与参考录音逐字一致的文本",
    "x_vector_only_mode": false
  }
}
```

推荐使用 5～15 秒、单人、无音乐、低混响的干净录音。

### 按语言重新生成参考或训练 WAV

默认不写 `regenerate`，程序会复用全部缓存并只补缺失内容。`prepare` 和 `train` 使用
同一套重生成规则。

保留所有参考，仅重生成全部语言的训练 WAV：

```json
"regenerate": {
  "audio": true,
  "references": false,
  "languages": "all"
}
```

只重生成法语的本地化参考和法语训练 WAV：

```json
"regenerate": {
  "audio": true,
  "references": true,
  "languages": ["fr"]
}
```

同时重生成中文、日语：

```json
"regenerate": {
  "audio": true,
  "references": true,
  "languages": ["zh", "ja"]
}
```

字段含义：

| 字段 | 作用 |
|---|---|
| `audio` | 重生成选中语言的训练 WAV |
| `references` | 重生成选中语言的派生参考；必须同时设置 `audio: true` |
| `languages` | `"all"` 或 `experiment.languages` 中的语言数组 |

安全规则：

- `cascade` 重生成的是 `localized-<lang>.wav`，唯一主参考 `designed.wav` 始终保留，
  因而不会让同一个 `voice_id` 突然变成另一种音色。
- `per_language` 重生成对应的 `designed-<lang>.wav`。
- `shared` 只有一条主参考，不能单独重生成参考；可以保留参考并重生成 WAV。要更换主
  参考必须创建新的 `voice_id`。
- 上传录音的 `clone` 音色同样不能在原 `voice_id` 下更换参考；请创建新 ID。
- 完成后删除 `regenerate`，否则下一次运行仍会再次重生成。
- 旧配置 `regenerate_audio: true` 仍兼容，等价于保留参考并重生成全部语言 WAV，但
  不要和新 `regenerate` 同时配置。

训练配置中如果某个已有音色只出现在 `speakers`，它只会被复用。要重做它，还需在
`voices` 中写回该音色原来的完整生成设置，并增加 `regenerate`。

### 生成批次很久没有新日志

每个批次开始前都会显示：

```text
AUDIO BATCH START | language=fr | teacher_language=French | items=4
```

这样可以直接确认传给 Qwen 的目标语言。Qwen 单次生成是同步调用，复杂句子可能明显慢于
其他批次；超过 60 秒会持续显示 `AUDIO BATCH STILL RUNNING` 心跳，项目也通过
`max_new_tokens` 限制生成上限。需要停止时按 `Ctrl+C`；终端中的 `^[[A` 是方向键
“上”，不是中断信号。

## 设备选择

只配置外层设备即可：

```json
"experiment": {
  "device": "cuda:1"
}
```

VITS 训练和 Qwen 样本生成都会使用 `cuda:1`。可选值包括：

```text
auto
cuda
cuda:0
cuda:1
cpu
mps
```

专家可以在内部配置中通过 `generation.runtime.device` 单独覆盖 Qwen 设备。文本 LLM
通常是远程 HTTP 服务，不使用本机 CUDA。ASR/声纹质检默认关闭；启用后可分别配置设备。

## 支持的语言

内置 Qwen Teacher 与 G2P 路由：

| 代码 | 语言 | 文本前端 |
|---|---|---|
| `zh` | 中文 | Piper-plus Mandarin IPA |
| `en` | 英语 | eSpeak NG |
| `ja` | 日语 | Open JTalk |
| `ko` | 韩语 | Piper-plus Korean IPA |
| `de` | 德语 | eSpeak NG |
| `fr` | 法语 | eSpeak NG |
| `ru` | 俄语 | eSpeak NG |
| `es` | 西班牙语 | eSpeak NG |
| `pt` | 葡萄牙语 | eSpeak NG |
| `it` | 意大利语 | eSpeak NG |

查看全部语言状态：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer languages \
  --config training_configs/my_model.json
```

查看最终前端路由：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer frontend-info \
  --config training_configs/my_model.json
```

增加 Qwen 十语之外的语言，需要同时提供：

1. Teacher 对该语言的生成能力；
2. 文本规范化规则；
3. G2P/音素前端；
4. `configs/system/language_registry.json` 注册项；
5. 母语者评测。

增加语言前端不是训练 G2P，而是接入或实现该语言的文本规则。TTS 模型本身需要使用新增
语言数据重新训练；已经导出的 ONNX 不会自动学会新语言。

## 继续训练

使用 [resume.example.json](training_configs/resume.example.json)：

```json
"initialization": {
  "mode": "resume",
  "checkpoint": "runs/model_1/checkpoints/last"
}
```

数据阶段不必关闭。需要检查/补齐音色时，保持原来的 `dataset.voices` 和
`dataset.speakers`；所有音色已准备好时，只保留 `dataset.speakers` 即可。
由多个公共音色组装的模型则保持原来的 `dataset.speakers` 映射。程序会重建当前模型
metadata，然后继续训练。如果 checkpoint 原本包含多个音色，程序也会保留其他音色，
不需要手写 metadata 路径。

`resume` 要求语言及顺序、speaker 集合和模型结构与 checkpoint 一致。

`last` 和 `best` 都是可续训的完整 checkpoint。续训到新的 `experiment.name` 时，
系统会把来源实验的历史 `best` 一并复制到新目录；即使后续指标没有刷新，也不会再出现
新目录只有 `last`、没有 `best` 的情况。中断后无缝继续通常选择 `last`；希望从验证指标
最优的一轮重新训练则选择 `best`。

时长预测器也遵循 checkpoint 兼容规则：本次升级前训练的 format-4 模型没有
`duration_predictor_type` 字段，系统会将其识别为原有的
`stochastic_lognormal`。即使最新 preset 默认值已经升级，`resume`、
`refine_text_prior` 和 `expand_speakers` 仍会自动使用 checkpoint 中的旧结构，
不会因更新代码而突然无法继续训练。

要把旧模型主干迁移到新的 Flow SDP，复制
[sdp-warm-start.example.json](training_configs/sdp-warm-start.example.json)，使用新的
实验名称和 `initialization.mode: "warm_start"`。Text Encoder、音色/语言 embedding、
声学 Flow 和 Decoder 会复用，`duration_predictor` 会重新初始化；这不是普通 resume。

### 声音已学会，但文字推理仍不稳定

`quality` 和 `mobile` 预设会在同一次训练中自动执行两个阶段：

1. 标准阶段训练完整 VITS，先把音色、声码器和 posterior 重建学稳定；
2. 至少完成 30,000 step，并且每个“语言×音色”的分段 Posterior Mel、完整均值
   Posterior Mel 和完整采样 Posterior Mel 连续三次验证全部达标后，才冻结音色条件、
   Posterior Encoder、Decoder 和判别器，只强化 Text Encoder、Flow 与 Duration
   Predictor。总体平均值不会再掩盖某一种语言的电音。

日志出现 `TEXT PRIOR REFINEMENT ACTIVATED` 表示已经安全进入第二阶段。阶段和独立
优化器都保存在 checkpoint 中，`resume` 后会从同一阶段继续。最终仍只有一个
checkpoint 和一个 ONNX，不会产生需要组合的两个模型。

需要修复声学主链或不希望自动冻结时，在普通配置中写：

```json
"training": {
  "stage": "standard"
}
```

`standard` 会忽略自动 refinement 开关。已经处于 `text_prior_refinement` 的 checkpoint
不能靠普通 `resume` 解冻；必须使用 `warm_start` 创建新实验。显式开始文本先验强化仍使用
`initialization.mode: "refine_text_prior"`，避免误把从零训练的模型直接冻结。

混合精度默认保持 `fp32`，因此升级项目不会悄悄改变已有音质基线。4090 等支持 BF16 的
CUDA 显卡可显式设置 `"mixed_precision": "bf16"`，模型权重、优化器主权重以及
Mel/KL 汇总仍保持 FP32；但数值不可能与纯 FP32 逐位一致，正式长训前应先做短程 A/B。
FP16 只在显式配置时启用，并将 GradScaler 状态随 checkpoint 保存和恢复。

如果已有 **format 4** checkpoint 的 posterior 音频正常，但
`aligned-text-prior.wav`、`text-only-*.wav` 较差，可复制
[refine-text-prior.example.json](training_configs/refine-text-prior.example.json)，设置：

```json
"initialization": {
  "mode": "refine_text_prior",
  "checkpoint": "runs/old_model/checkpoints/last"
}
```

这会创建一个新实验目录并从旧权重开始；音色条件和声码器权重不会更新，显著降低
强化文本先验时破坏已有声音的风险。
语言顺序、speaker 映射、模型架构和前端必须与原 checkpoint 一致。format 1/2 的训练
语义有误，format 3 的时长结构不同，不能使用此模式；它们仍需按错误提示迁移或重训。
原模型是 `mobile` 就继续使用 `preset: "mobile"`，原模型是 `quality` 就继续使用
`preset: "quality"`，不能借 refinement 切换前端。

训练是否真正改善，应同时观察：posterior Mel 不应明显恶化、`prior_mel` 应下降、
deterministic duration ratio 应逐渐接近 `1.0`，并试听 `text-only-deterministic.wav`。
不要只按 epoch 数量判断。

## 增加音色

使用 [add-speaker.example.json](training_configs/add-speaker.example.json)：

```json
"initialization": {
  "mode": "expand_speakers",
  "checkpoint": "runs/model_1/checkpoints/last"
},
"dataset": {
  "speakers": {
    "voice_02": "voice_02"
  },
  "sentences_per_language": 2000,
  "text": {"provider": "builtin"},
  "voices": {
    "voice_02": {
      "mode": "design",
      "reference_strategy": "cascade"
    }
  }
}
```

程序会从 checkpoint 旁的 `run-layout.json` 自动找到旧模型数据，把旧音色合并进新
训练集，避免只训练新音色导致遗忘。新音色有自己的
`datasets/voices/voice_02/`，旧音色的数据不会复制。

如果必须从零一次性合并外部数据，专家仍可用 `dataset.include` 提供已有清单；普通
用户更建议先训练第一个音色，再用 `expand_speakers` 逐个增加，配置更简单且可恢复。

## 同时训练多个模型

每份配置有独立模型名：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer train-many \
  training_configs/train1.json \
  training_configs/train2.json \
  --max-parallel 2
```

不同模型分别写入：

```text
datasets/<model_name>/
runs/<model_name>/
artifacts/<model_name>/
```

相同 `voice_id` 仍引用同一份公共数据。并行任务最好使用不同 GPU；不要让两个进程同时
补写同一个尚未完成的 `voice_id`。

## 输出在哪里

```text
datasets/
├── voices/<voice_id>/          # 公共文本、参考音色和 WAV
└── <model_name>/
    ├── metadata.csv            # 当前训练选择的原始清单
    ├── metadata.phonemes.csv   # 已冻结音素的清单
    └── dataset.json

runs/<model_name>/
├── resolved-config.json
├── pipeline-report.json
├── quality/
├── validation-audio/
│   └── epoch-XXXX/
│       ├── target.wav
│       ├── posterior-reconstruction.wav
│       ├── posterior-sampled-reconstruction.wav
│       ├── aligned-text-prior.wav
│       ├── text-only-inference.wav
│       ├── text-only-deterministic.wav
│       ├── diagnostics.json
│       └── profiles/<language>/<speaker>/
│           └── 同一套六个 WAV 与 diagnostics.json
└── checkpoints/
    ├── best/
    └── last/

artifacts/<model_name>/
├── model.onnx
├── model.onnx.json
├── tokens.txt
├── tokens.json
├── frontend.json
├── frontend.conformance.json
├── frontend-packs/
│   ├── manifest.json           # 共享核心与语言包索引
│   ├── _shared/espeak-ng/      # 多个拉丁语言共享，存在时只保存一份
│   └── <language>/
│       ├── manifest.json       # provider、版本和运行时资源要求
│       └── conformance.json    # 该语言的文本→音素→token 校验向量
├── composable/
│   ├── catalog.json            # 可托管包目录和 SHA256
│   ├── core/
│   │   ├── model.onnx          # 不含 language/speaker embedding 表
│   │   ├── manifest.json
│   │   └── tokens.json
│   ├── languages/<language>/
│   │   ├── embedding.f32       # 模型专属语言向量
│   │   ├── manifest.json       # 前端契约及兼容 core hash
│   │   ├── conformance.json
│   │   └── runtime/             # 该语言需要的词典/前端数据（如需要）
│   ├── voices/<speaker>/
│   │   ├── embedding.f32       # 模型专属音色向量
│   │   └── manifest.json
│   └── packages/               # Android/iOS 按需下载的 ZIP
│       ├── language-zh.zip
│       └── voice-voice_01.zip
└── android_text/               # 仅 preset=mobile
    ├── espeak-ng-data/
    ├── model.weights            # 所有语言共享的一份权重
    ├── model-zh.onnx
    ├── model-en.onnx
    └── ...
```

务必保留 `runs/<model_name>/checkpoints/`。ONNX 用于推理，checkpoint 用于续训、增加
音色、迁移结构和后续压缩。

### 主模型内置、语言和音色按需下载

`composable/` 是完整的移动部署目录。App 使用者无需知道每种语言采用 eSpeak、
OpenJTalk 还是 Piper Plus：语言包的 `manifest.json` 已声明 provider，所需可分发数据也
已放入该语言包的 `runtime/`。Android Demo 会读取目录自行选择并安装。

最简单的本地集成方式就是把整个目录复制到 Demo：

```bash
rm -rf /path/to/TTSDemo/app/src/main/assets/tts/*
cp -R artifacts/<model_name>/composable \
  /path/to/TTSDemo/app/src/main/assets/tts/composable
```

若只想内置部分语言/音色并在线下载其余包，再使用 Demo 提供的安装脚本。

`composable/core/model.onnx` 接收 `language_embedding` 和
`speaker_embedding` 两个外部输入。语言包与音色包可以任意组合，但两个包的
`compatible_core_sha256` 必须和 App 内置核心完全一致：

```text
APK 内置：composable/core + catalog.json
运行时下载：language-<code>.zip + voice-<id>.zip
最终组合：core + 一个语言包 + 一个音色包
```

向量空间会随核心训练变化，所以一个模型导出的音色包不能默认用于另一个模型。增加
新音色时应从保留的 checkpoint 训练/扩展后重新导出核心和目录；若以后实现“冻结核心、
只训练新 speaker embedding”的模式，才可以保持旧 core ID 不变。新增语言还受核心
词表和训练覆盖限制，不能只上传一个语言向量就让没学过的语言自动可用。

验证音频按链路逐级排错：

1. `target.wav` 是数据集原音频。
2. `posterior-reconstruction.wav` 使用稳定的 posterior 均值，检查
   WAV→频谱→Posterior Encoder→Decoder，不经过文本前端；该文件名保持跨版本兼容。
3. `posterior-sampled-reconstruction.wav` 使用训练时的随机采样；它与均值版差异很大时，应查看
   `diagnostics.json` 的 `posterior_scale_*`。
4. `aligned-text-prior.wav` 加入文本编码和 MAS 真值对齐，但不测试时长预测。
5. 两个 `text-only-*` 才是完整 TTS 推理链路。

标准阶段的 `aligned-text-prior.wav` 只用于验证；进入 text-prior refinement 后，固定
Decoder 并使用对应的 aligned Mel、KL 和 Duration 目标更新 Text Encoder、Flow 与
Duration Predictor，因此不会反向改变已经通过声学门槛的 Decoder。

每次验证的 `metrics.validation.profiles` 会分别记录每个语言×音色的指标。自动阶段切换
要求所有 profile 连续通过，不再只看总体 `posterior_mel`。试听文件同时写入
`validation-audio/epoch-XXXX/profiles/<language>/<speaker>/`；顶层文件继续保留，兼容
已有排错脚本。

从零训练的 `epoch-0001` 是冷启动诊断，不是可发布音质。以单音色、单语言、
2,000 条样本、`batch_size=4` 为例，扣除 5% 验证集后第一轮约只有 475 step；
39M Generator 和判别器此时仍可能只产生噪音。先看稳定的
`posterior-reconstruction.wav` 是否随 5,000～10,000 step 持续改善，再判断主链；
`aligned-text-prior.wav` 和两个 `text-only-*` 在 epoch 1 是噪音并不表示链路损坏。

2026-07-30 训练回归的完整时间线、原因和迁移建议见
[中英文回归说明](docs/regression_2026-07-30.md)。

旧配置中的单数 `dataset.voice` 仍可读取，以避免已有训练任务失效；新配置统一使用
`dataset.voices`。

## 测试导出的 ONNX

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer synthesize-onnx \
  --model-dir artifacts/my_model \
  --text "你好，欢迎使用。" \
  --language zh \
  --speaker xiaoling \
  --output output.wav
```

普通文本推理默认启用自动分句。运行时先按各语言标点寻找边界，再用模型实际使用的
G2P 计算音素数量；超过 90 个音素 token 的句子会继续拆分，分别合成后自动加入停顿。
因此长段落不会再作为一个超长序列直接送进 ONNX。日志中的 `TEXT CHUNKS` 会显示
分段数量及每段 token 数。

通常不需要调整；如果模型只在更短的训练句上稳定，可降低上限：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer synthesize-onnx \
  --model-dir artifacts/my_model \
  --text "今天早上出门的时候，天空还是晴朗的。下班以后请帮我拿一下雨伞。" \
  --language zh \
  --speaker xiaoling \
  --output output.wav \
  --max-phoneme-tokens 70 \
  --sentence-pause-ms 180 \
  --clause-pause-ms 100 \
  --chunk-pause-ms 60
```

`--no-auto-chunk` 仅用于排错和对比，不建议用于长文本。自动分句解决的是长序列失稳；
如果拆分后的短句仍不自然，需要继续检查 G2P、数据和韵律建模。

验证移动端前端一致性：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer verify-frontend \
  --model-dir artifacts/my_model
```

最终发布资源不只有 `model.onnx`。移动端还必须携带 tokens、语言/音色映射和
`frontend.json` 指定的语言前端资源。

### Android 直接输入普通文本

只需要 eSpeak 可正确覆盖的语言时可以选择 `mobile`。包含中文、日文或韩文的产品模型
推荐选择 `mobile_routed`：

```json
{
  "task": "train",
  "preset": "mobile_routed",
  "experiment": {
    "name": "my_mobile_tts",
    "languages": ["zh", "en", "ja", "ko", "fr", "es", "pt"],
    "device": "cuda:0"
  },
  "dataset": {
    "sentences_per_language": 2000,
    "speakers": {
      "xiaoling": "voice_xiaoling"
    }
  },
  "training": {
    "batch_size": 4,
    "epochs": 200
  }
}
```

导出时会自动检查前端契约。`mobile_routed` 保留 mobile 的 2 层轻量 Flow SDP，
但中文走 Piper Plus、日文走 OpenJTalk、韩文走 Piper Plus，拉丁语言走 eSpeak。
导出目录的 `frontend-packs/` 可按语言独立交付，所有语言仍共享一份 `model.onnx`。

原有 `mobile` 使用统一 eSpeak，VITS 训练核心采用更容易
稳定对齐的紧凑序列 `BOS,(phoneme)*,EOS`。sherpa 传入的 Piper PAD 只属于部署传输
格式，会由 ONNX 输入适配器删除。`model.onnx.json` 中会出现
`"text_input": {"supported": true}`，并生成 `android_text/`。Java Demo 会检查
这个字段，不会把不兼容模型静默读错。

`mobile` 不再推荐用于日语汉字。若 eSpeak 输出 Unicode 字符名称，训练预检会拒绝
继续。`mobile_routed` 的 Android/iOS App 仍需安装语言包声明的原生前端；不能只复制
一个 ONNX 文件解决文本规范化和 G2P。

### 接入 Android Demo

配套 Java Demo 位于 [XQSJ/TTS-demo-android](https://github.com/XQSJ/TTS-demo-android)。
训练与导出完成后，在 Demo 仓库执行：

```bash
git clone https://github.com/XQSJ/TTS-demo-android.git
cd TTS-demo-android

./scripts/install_composable_model.sh \
  /path/to/TTSTRAINER/artifacts/my_mobile_tts \
  --language zh \
  --voice xiaoling

./gradlew clean :app:assembleDebug
```

如果需要语言包和音色包在 App 中按需下载，把
`artifacts/my_mobile_tts/composable/` 原样上传到 HTTPS 静态服务器，并在安装脚本中
增加：

```bash
--base-url https://cdn.example.com/tts/my_mobile_tts/composable/
```

Demo 会校验 core、语言 embedding、音色 embedding、前端 provider、token 和冻结
conformance 数据，防止把不属于同一个模型的资源组合起来。日语还需要按 Demo README
安装共享 OpenJTalk 词典；中文和韩文使用 Piper Plus Android G2P。

## 模型和前端资源

查看或下载项目本地 Qwen 模型：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer models status
PYTHONPATH=src .venv/bin/python -m tts_trainer models ensure voice-design-1.7b
PYTHONPATH=src .venv/bin/python -m tts_trainer models ensure base-1.7b
```

模型会放在项目的 `models/qwen/`，后续始终从该目录使用。

前端资源：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer frontends status openjtalk
PYTHONPATH=src .venv/bin/python -m tts_trainer frontends ensure openjtalk
PYTHONPATH=src .venv/bin/python -m tts_trainer frontends ensure korean
```

## 离线部署包

在与目标服务器相同的操作系统、CPU 架构、Python 和 CUDA 环境中构建：

```bash
.venv/bin/python deployment/build_bundle.py \
  --output dist/tts-trainer-offline
```

默认打包源码、Python wheelhouse 和中日韩前端资源，但不下载 GB 级 Qwen 权重。需要把
权重也放进包时增加 `--download-models`，生成单个归档时增加 `--archive`。

目标机器安装：

```bash
python3 verify_bundle.py
./install_offline.sh /path/to/venv
```

`wheelhouse` 是当前平台所需 Python 安装包的离线目录，不能把 macOS 构建的
wheelhouse 拿到 Linux/CUDA 服务器使用。完整说明见
[deployment/README.md](deployment/README.md)。

## 质检

基础信号质检默认开启，不需要额外模型，包括：

- 时长；
- 响度；
- 削波；
- DC 偏移；
- 首尾静音；
- 文本长度与语速异常。

ASR 回识别和声纹相似度属于可选的语义质检：

```bash
.venv/bin/pip install -e '.[quality]'
PYTHONPATH=src .venv/bin/python -m tts_trainer quality-models ensure asr-small
PYTHONPATH=src .venv/bin/python -m tts_trainer quality-models ensure speaker-ecapa
```

相关权重存放在 `models/quality/`。专家可在内部配置中启用。

## 普通配置与专家配置

普通用户只编辑 `training_configs/` 中的配置。复杂参数集中在：

```text
configs/internal/pipeline_defaults.json
configs/internal/quality_pipeline_defaults.json
configs/system/vits_mobile_architecture.json
configs/system/language_registry.json
configs/models.json
```

默认优先级：

```text
用户配置 > quality/compact preset > 系统架构默认值
```

可用 preset：

- `quality_commercial`：闭源商业应用的质量优先模式。`zh/en/ja/ko/fr/es/pt`
  全部使用 MIT `piper-plus-g2p`，不包含 eSpeak；必须安装 `.[commercial]`。
- `mobile_commercial`：闭源商业应用的移动轻量模式；前端与
  `quality_commercial` 相同，使用 64 channels、2 层 Flow SDP。
- `quality`：默认推荐，约 39M Generator，质量优先；使用 128 channels、6 层
  Flow SDP，时长模块 FP32 约 4.4 MB；
- `compact`：小模型和流程验证，音质上限较低。
- `mobile`：独立的移动部署链路。模型尺寸仍采用质量架构，但文本前端改为统一
  eSpeak。训练核心使用 `BOS,(phoneme)*,EOS`，导出图通过
  `strip-piper-pads-v1` 将 sherpa/Piper 传输序列规范化后再进入 VITS，可在
  Android 直接接收普通文本；使用 64 channels、2 层轻量 Flow SDP，时长模块
  FP32 约 0.54 MB。必须从头训练。
- `mobile_routed`：同样使用 64 channels、2 层轻量 Flow SDP，但恢复逐语言专用
  前端。导出共享核心和 `frontend-packs/<language>`，适合语言包按需安装。包含
  中日韩的移动产品优先选择它。

三个时长模式都保留在 `model.duration_predictor_type`，这是专家参数，普通配置不用写：

| 类型 | 用途 | preset 默认值 |
|---|---|---|
| `stochastic_lognormal` | 本次升级前的兼容实现，最简单 | `compact` 与旧 checkpoint |
| `stochastic_mobile` | 轻量条件 Normalizing Flow | `mobile`、`mobile_routed` |
| `stochastic_quality` | 更深的条件 Normalizing Flow | `quality` |

Flow SDP 训练的是音素时长分布，只增加节奏表达能力，不会改变 speaker/language
embedding、G2P 或 Decoder。最终仍导出单个 ONNX；`duration_noise_scale=0` 为确定性
节奏，推荐试听范围为 `0.15～0.45`。

`quality_commercial`、`mobile_commercial`、`quality`、`mobile_routed` 和
`mobile` 的前端契约彼此隔离：选择 `mobile` 不会修改
`quality` 的中文 Piper Plus、日语 Open JTalk、韩语 Piper Plus 路由。
不要在不同前端契约之间 `resume` 或 `warm_start`，音频数据可以复用，但模型必须
分别训练。

不要只因为训练能运行就使用 `compact` 发布产品模型。

## 常用命令

```bash
# 只生成/补齐文本
PYTHONPATH=src .venv/bin/python -m tts_trainer generate-texts \
  --config training_configs/my_model.json

# 自动生成/补齐文本，再生成/补齐 Qwen 音频并写 metadata
PYTHONPATH=src .venv/bin/python -m tts_trainer generate-samples \
  --config training_configs/my_model.json

# 只训练 VITS
PYTHONPATH=src .venv/bin/python -m tts_trainer train-vits \
  --config training_configs/my_model.json

# 重新导出 checkpoint
PYTHONPATH=src .venv/bin/python -m tts_trainer export-vits \
  --config training_configs/my_model.json \
  --validate-runtime
```

按 `Ctrl+C` 会安全停止。已经生成的文本批次、WAV 和 checkpoint 会保留；重新运行同一
命令即可继续。

## 当前限制

- Qwen Teacher 的公开语言范围决定自动生成音频的语言范围。
- 七语同音色质量仍取决于 Teacher 数据、文本覆盖和各语言发音评测。
- VITS 音色相似不代表韵律一定自然；应试听 `runs/<name>/validation-audio/`。
  `best` 默认按 `combined_mel = posterior_mel + prior_mel` 保存，用来同时约束声学
  重建和文本先验，但仍不能代替可懂度试听或 ASR。流水线默认导出 `last`；
  `prior_mel` 和时长比用于观察文字先验及韵律是否继续改善。
- 旧版 `preset=mobile` 曾直接使用带 blank 的 Piper v1/v2 序列训练。当前 v3 将
  Piper wire PAD 与 VITS 训练序列解耦；请保留原始 WAV/文本并使用新的
  `experiment.name` 从零训练。公共音频会复用，`quality` checkpoint 和专用 G2P
  数据不受影响。
- 新增语言需要新前端和重新训练。
- 结构变化通常不能直接 `resume`；高级迁移参考
  [sdp-warm-start.example.json](training_configs/sdp-warm-start.example.json)。

## 开发测试

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q
```

## 合规与许可证

请确认训练文本版权、参考声音授权、商业分发权、声纹保存政策和用户撤回机制。模型/代码
许可证不会自动授予真人声音或第三方文本的使用权。

本项目使用 Apache-2.0，第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
