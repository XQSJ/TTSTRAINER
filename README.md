# TTSTRAINER

用 Qwen3-TTS 生成训练数据，训练多语言、多音色 VITS，并导出移动端可用的
ONNX 资源。

## 三个配置概念

| 名称 | 属于哪里 | 作用 |
|---|---|---|
| `task` | 自动流水线 | `prepare` 只准备音色数据；`train` 准备数据并训练、导出 |
| `dataset.voices` | 公共音色数据 | 声明一个或多个需要生成/补齐的 `voice_id` 及生成方式 |
| `dataset.speakers` | 某个 VITS 模型 | 把模型内 speaker 名称映射到公共 `voice_id` |

程序不会再通过“配置中有哪些字段”猜测用户想做什么，`task` 是自动入口
`run-pipeline` 的明确执行模式。直接子命令（如 `generate-samples`、`train-vits`）本身
已有明确动作，不依赖推断。旧配置未写 `task` 时默认按 `train` 处理。

`resume`、`warm_start`、`expand_speakers` 不是第三种 task，它们只描述
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

## 安装

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
        "reference_strategy": "per_language",
        "regenerate_audio": false,
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
        "reference_strategy": "per_language",
        "regenerate_audio": false,
        "prompt": "A warm natural young adult female voice.",
        "reference_text": "你好，这是一段用于创建统一音色的参考录音。",
        "reference_language": "zh"
      },
      "voice_xiaoling_b": {
        "mode": "clone",
        "regenerate_audio": false,
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
      "reference_strategy": "per_language",
      "regenerate_audio": false,
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
| `dataset.voices.<id>.regenerate_audio` | 是否强制重写该音色本次选中的训练 WAV |
| `dataset.voices.<id>.reference_strategy` | `shared` 或 `per_language` 设计参考策略 |
| `training.batch_size` | 显存不足时优先调小 |
| `training.epochs` | 总训练轮数 |

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
    "reference_strategy": "per_language",
    "regenerate_audio": false,
    "prompt": "A warm adult voice with natural conversational pacing.",
    "reference_text": "你好，这是一段用于创建统一音色的参考录音。",
    "reference_language": "zh"
  }
}
```

多语言模型推荐 `reference_strategy: "per_language"`。程序会用正确的 Qwen 语言名为
每种语言生成一条设计参考，再分别创建 clone prompt：

```text
fr → French → references/designed-fr.wav
es → Spanish → references/designed-es.wav
```

这能降低“中文参考音频让法语、西语带中文口音”的风险。`shared` 则让全部语言共用一条
参考音频，跨语言音色通常更统一，但参考语言口音更容易迁移到其他语言。

`per_language` 默认使用该语言训练文本的第一句作为参考，也可以人工指定：

```json
"reference_texts": {
  "fr": "Bonjour, ceci est une référence vocale en français.",
  "es": "Hola, esta es una referencia de voz en español."
}
```

同一个 `voice_id` 的参考策略属于音色身份。已有 `shared` 音色要切换为
`per_language`，请使用新的 `voice_id`，避免把两套参考混进同一个公共数据集。

### 上传参考录音

复制 [clone.example.json](training_configs/clone.example.json)。关键字段：

```json
"voices": {
  "my_voice": {
    "mode": "clone",
    "regenerate_audio": false,
    "reference_audio": "datasets/references/my_voice.wav",
    "reference_text": "与参考录音逐字一致的文本",
    "x_vector_only_mode": false
  }
}
```

推荐使用 5～15 秒、单人、无音乐、低混响的干净录音。

### 强制重新生成某个音色的音频

`prepare` 和 `train` 使用同一规则。在需要重做的音色中设置：

```json
"voices": {
  "voice_a": {
    "mode": "design",
    "reference_strategy": "per_language",
    "regenerate_audio": true,
    "prompt": "A warm natural adult voice."
  },
  "voice_b": {
    "mode": "design",
    "reference_strategy": "per_language",
    "regenerate_audio": false,
    "prompt": "A bright friendly adult voice."
  }
}
```

- `false`：默认行为，已有 WAV 直接复用，只补缺失语言或数量。
- `true`：重新生成该音色当前语言和数量范围内的全部 WAV；其他音色不受影响。
- 成功完成一次强制重生成后，应改回 `false`，否则下次运行还会再次生成。
- `regenerate_audio` 只重做训练 WAV，不更换已锁定的参考音色、Prompt 或参考录音。

训练配置中如果某个已有音色只出现在 `speakers`，它只会被复用。要强制重做它，还需在
`voices` 中写回该音色原来的完整生成设置，并设置 `regenerate_audio: true`。

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
      "reference_strategy": "per_language",
      "regenerate_audio": false
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
└── checkpoints/
    ├── best/
    └── last/

artifacts/<model_name>/
├── model.onnx
├── model.json
├── tokens.txt
├── tokens.json
├── voices.json
├── frontend.json
└── frontend.conformance.json
```

务必保留 `runs/<model_name>/checkpoints/`。ONNX 用于推理，checkpoint 用于续训、增加
音色、迁移结构和后续压缩。

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

验证移动端前端一致性：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer verify-frontend \
  --model-dir artifacts/my_model
```

最终发布资源不只有 `model.onnx`。移动端还必须携带 tokens、语言/音色映射和
`frontend.json` 指定的语言前端资源。

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

- `quality`：默认推荐，约 39M Generator，质量优先；
- `compact`：小模型和流程验证，音质上限较低。

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
- VITS 音色相似不代表韵律一定自然；应试听 `runs/<name>/validation-audio/` 并选择
  best checkpoint。
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
