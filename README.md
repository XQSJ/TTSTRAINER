# tts-trainer

这是一个面向移动端离线部署的多语言、多音色 VITS 训练工程。即使你没有语音
模型训练经验，也可以从一份带中英双语说明的 JSON 配置开始，完成样本生成、
VITS 训练、checkpoint 保存和 ONNX 导出。

整个过程可以理解为：

```text
准备七语文本
  → Qwen 设计/克隆音色并生成 WAV（可跳过）
  → 冻结音素
  → 训练 Multilingual VITS
  → 保存可恢复 checkpoint
  → 导出动态 ONNX
  → ONNX Runtime 验证
  → 集成 Android / iOS
```

当前支持中文、英语、日语、韩语、法语、西班牙语和葡萄牙语。模型内部将
`language embedding` 和 `speaker embedding` 分开，所以既可以训练一个固定
音色，也可以在同一个模型中继续增加音色。

> 项目目前处于工程验证阶段。训练、恢复、扩音色和 ONNX 导出已经跑通；产品级
> 七语 G2P、自动音质评测、蒸馏和 INT8 仍在完善。正式投入大规模训练前，请先
> 阅读“已知限制”。

## 第一次使用：按这 5 步做

### 第 1 步：选择一份配置

所有给普通用户修改的配置都在 `training_configs/`，一份 JSON 就代表一个训练
任务。先根据目标选择：

| 你的目标 | 从哪个文件开始 |
|---|---|
| 用 Prompt 设计一个新音色并从零训练 | `training_configs/train1.json` |
| 再训练一个互不影响的模型 | `training_configs/train2.json` |
| 上传一段录音克隆音色 | `training_configs/clone.example.json` |
| 从旧 checkpoint 继续训练 | `training_configs/resume.example.json` |
| 给旧模型增加新音色 | `training_configs/add-speaker.example.json` |

通常不要直接改示例文件，复制一份更容易管理：

```bash
cp training_configs/train1.json training_configs/my_reader.json
```

### 第 2 步：只改四类内容

打开刚复制的 JSON，第一次训练只需要关心：

1. `experiment.name`：模型名称，例如 `my_reader`。
2. `experiment.languages`：需要训练的语言，例如 `["zh", "en"]`。
3. `generation.voice`：使用文字设计音色，或者上传参考录音。
4. `training`：显存不足时先减小 `batch_size`；其他参数可以先保留默认值。

配置中的 `_comment`、`_comment_languages` 是“中文 / English”双语说明，程序会
自动忽略。标准 JSON 不支持 `//` 注释，所以这里特意使用合法的 `_comment`
字段；可以保留或删除，不要把真正的参数名改成中文。

每个 `name` 都有完全独立的目录：

```text
my_reader
  ├── datasets/my_reader/    # 音频和 metadata
  ├── runs/my_reader/        # 可恢复训练的中间态，务必保留
  └── artifacts/my_reader/   # 最终 ONNX 发布资源
```

### 第 3 步：准备要朗读的文本

编辑 `datasets/texts.example.csv`，或者新建 CSV 并把路径填入
`generation.text_manifest`：

```csv
text,language
你好，欢迎使用多语言语音系统。,zh
Hello, welcome to the multilingual speech system.,en
```

CSV 中至少要有配置里每种语言的一条文本。示例 CSV 只能验证流程，不能训练出
可发布的声音；正式训练需要数量充足、文本覆盖合理并经过检查的数据。

### 第 4 步：安装并一键运行

环境要求：

- Python 3.10 或更高版本
- macOS、Linux 或 Windows
- 训练建议使用 NVIDIA CUDA GPU；Apple Silicon MPS 和 CPU 可用于烟雾测试
- 使用内置音素化命令时需要 `espeak-ng`

macOS：

```bash
brew install espeak-ng
```

Ubuntu/Debian：

```bash
sudo apt-get install espeak-ng
```

在线安装：

```bash
git clone https://github.com/XQSJ/TTSTRAINER.git
cd TTSTRAINER

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[qwen,export,dev]'
```

项目不复制 Qwen3-TTS 源码。`qwen` extra 会安装固定版本的官方 `qwen-tts`
Python 包。只使用自己准备的 WAV、不需要 Qwen 生成样本时，可以改为：

```bash
.venv/bin/pip install -e '.[export,dev]'
```

检查 Qwen Python 运行时不会下载模型：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer qwen-runtime
```

Windows 将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。

运行一份配置：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline --config training_configs/train1.json
```

程序会按配置自动执行：

```text
创建命名目录
  → 检查/按需下载项目内 Qwen 权重
  → 生成 PCM16 训练 WAV 和 metadata.csv
  → 冻结音素
  → 数据校验
  → VITS 训练
  → ONNX 导出与运行时验证
```

Qwen 权重不会在安装或读取配置时下载。只有真正执行样本生成，并且
`generation.auto_download_models=true` 时，缺少的权重才会下载到项目自己的
`models/qwen/`，后续始终复用这里的文件。

### 第 5 步：找到结果

运行成功后，重点看两个目录：

```text
runs/<name>/checkpoints/last/training-state.pt  # 训练中间态
artifacts/<name>/model.onnx                     # 移动端推理模型
```

不要只保存 ONNX。`runs/<name>/` 包含 Generator、Discriminator、Optimizer、
speaker/language 映射等完整状态，以后续训、增加音色、蒸馏、剪枝和 QAT 都要
从它开始。

## 安装、测试与离线依赖

### 构建离线依赖包

构建器默认下载当前平台的 Python wheel，但不会下载 Qwen 权重：

```bash
.venv/bin/python deployment/build_bundle.py \
  --output dist/tts-trainer-offline
```

复制 `dist/tts-trainer-offline/` 到目标机器后：

```bash
./install_offline.sh /path/to/new/venv
```

wheelhouse 与操作系统、CPU 架构和 Python 版本绑定。macOS wheelhouse 不能用于
Linux/CUDA 训练服务器。详细说明见 [deployment/README.md](deployment/README.md)。

### 验证安装

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m tts_trainer --help
```

仓库当前测试覆盖数据校验、音素词表、模型下载检测、VITS 前向/反向、一步 GAN
训练、checkpoint 恢复、动态 ONNX 导出和 ONNX Runtime 推理。

## 功能状态

| 功能 | 状态 |
|---|---|
| WAV/metadata 校验 | 已完成 |
| 单音色及多音色数据格式 | 已完成 |
| 七语 language embedding | 已完成 |
| 每个模型独立选择语言及动态 language map | 已完成 |
| speaker embedding | 已完成 |
| MAS、Flow、Duration、Waveform Decoder | 已完成 |
| Multi-Period/Scale Discriminator | 已完成 |
| 完整 checkpoint 保存 | 已完成 |
| 动态文本长度 ONNX | 已完成 |
| ONNX Runtime 参考合成 | 已完成 |
| Qwen 模型项目内检测和按需下载 | 已完成 |
| 命名实验与目录隔离 | 已完成 |
| 多配置批量/并行训练 | 已完成 |
| checkpoint 续训与新音色扩展 | 已完成 |
| Qwen VoiceDesign/上传音色批量生成 | 已完成 |
| 单配置自动生成→训练→导出 | 已完成 |
| 七语 eSpeak-ng 路由与前端版本契约 | 已完成 |
| 中日韩产品级专用 G2P | 进行中 |
| 验证集、best checkpoint、自动音质评测 | 待完成 |
| ASR、声纹和音质自动质检 | 待完成 |
| 蒸馏、剪枝、INT8、真机基准 | 待完成 |

## 项目目录

```text
tts_trainer/
├── training_configs/           # 用户的所有训练任务配置
│   ├── train1.json             # 训练 model_1
│   ├── train2.json             # 训练 model_2
│   ├── clone.example.json      # 上传参考音色
│   ├── resume.example.json     # 恢复完整训练状态
│   └── add-speaker.example.json # 扩展新音色
├── configs/
│   ├── internal/             # Qwen 生成细节和自动流水线默认值
│   ├── system/               # VITS 模型结构契约
│   └── models.json           # Qwen 模型注册
├── datasets/
│   ├── texts.example.csv      # 给 Qwen 生成的文本清单
│   └── metadata.example.csv
├── scripts/
│   ├── generate_samples.py
│   └── run_pipeline.py
├── docs/
│   ├── architecture.md
│   └── mobile_runtime.md
├── models/qwen/              # 按需下载的 Qwen 权重，不提交 Git
├── runs/<model-name>/        # 训练中间态、checkpoint 和日志
├── artifacts/<model-name>/   # ONNX 发布资源
├── src/tts_trainer/
└── tests/
```

## 准备训练数据

### 音频要求

默认配置要求：

- 单声道 WAV
- 16-bit PCM
- 22050 Hz
- 单句建议 2～12 秒
- 文本必须和声音严格对应
- 尽量没有背景音乐、混响、削波和降噪伪影

改变采样率时，配置和全部 WAV 必须一起改变。

### 原始 metadata

CSV 必须有表头。`audio` 相对于 CSV 文件所在目录：

```csv
audio,text,language,speaker
wavs/zh_000001.wav,你好世界,zh,voice_01
wavs/en_000001.wav,Hello world,en,voice_01
wavs/ja_000001.wav,こんにちは世界,ja,voice_01
```

支持的 language：

```text
zh en ja ko fr es pt
```

训练多个音色时，为不同配音员使用不同 speaker 名称：

```csv
audio,text,language,speaker
wavs/a_en_001.wav,Hello world,en,voice_01
wavs/b_en_001.wav,Hello world,en,voice_02
```

不要用 `speaker` 表示语言。同一个 speaker 最好覆盖多种语言，否则模型容易把
音色和语言绑定在一起。

### 校验数据

先为配置中的模型名创建数据目录，放入 WAV 和原始 metadata：

```bash
mkdir -p datasets/model_1/wavs
cp datasets/metadata.example.csv datasets/model_1/metadata.csv
```

单音色：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer validate \
  datasets/model_1/metadata.csv \
  --sample-rate 22050
```

多音色：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer validate \
  datasets/model_1/metadata.csv \
  --sample-rate 22050 \
  --multi-speaker
```

### 冻结音素

正式 VITS 配置要求 metadata 含 `phonemes` 列，从而确保训练端和移动端使用相同
的 token 语义。

建立 eSpeak 基线：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer phonemize \
  datasets/model_1/metadata.csv \
  datasets/model_1/metadata.phonemes.csv \
  --config training_configs/train1.json
```

输出类似：

```csv
audio,text,language,speaker,phonemes
wavs/en_000001.wav,Hello world,en,voice_01,h ə l ˈ o ʊ <space> w ˈ ɜ ː l d
```

音素以 UTF-8 codepoint 保存，`<space>` 表示 Piper 空格 token。命令还会在同一
目录生成 `frontend.lock.json`，记录 eSpeak-ng 版本、七语 voice、规范化规则和
token 格式。训练 checkpoint 和 ONNX 导出会继续携带这份契约。

查看当前机器最终生效的七语前端，不生成音频或下载模型：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer frontend-info \
  --config training_configs/train1.json
```

正式 metadata 可以再次校验：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer validate \
  datasets/model_1/metadata.phonemes.csv \
  --sample-rate 22050 \
  --require-phonemes
```

多音色数据再增加 `--multi-speaker`。

### 七语 G2P 使用什么

当前默认直接调用现有的 eSpeak-ng，而不是训练一套新的 G2P。它为每种语言切换
独立 voice：

```text
zh=cmn  en=en-us  ja=ja  ko=ko  fr=fr-fr  es=es  pt=pt-br
```

eSpeak-ng 有 C API、体积较小并支持 Android，因此比只存在于 Python 的前端更
容易让训练端和手机端保持一致。中文 `pypinyin`、日语 `pyopenjtalk`、韩语
`g2pK` 都是可以进一步接入的现有项目，但如果训练时改用它们，手机端也必须
复现完全相同的输出；不能只在 Python 训练端替换。

可在专家配置覆盖口音，例如欧葡：

```json
"frontend": {
  "provider": "espeak-ng",
  "voices": {"pt": "pt-pt"}
}
```

修改 provider、voice、规范化规则或 token 语义后，不应继续复用不兼容的旧
checkpoint。程序会比较 `frontend.lock.json` 和 checkpoint 契约并拒绝静默混用。

## 配置怎么改：从常用参数到专家参数

普通用户只需在 `training_configs/` 中新建或复制 JSON。
例如 [train1.json](training_configs/train1.json) 和 [train2.json](training_configs/train2.json)。
每个文件对应一个独立模型，并通过 `extends`
自动继承内部默认值：

```text
configs/system/vits_mobile_architecture.json       # 专家结构参数
                    ↓
configs/internal/pipeline_defaults.json           # Qwen/流水线细节
                    ↓
training_configs/train1.json                       # model_1 的用户参数
training_configs/train2.json                       # model_2 的用户参数
                    ↓
最终合并配置
```

相对继承路径以当前配置文件所在目录为基准，因此复制项目到其他目录仍然有效。

### 第一层：每个模型都要确认

这些参数决定“训练哪个模型、使用哪些语言、声音是什么”：

| 参数 | 要做什么 |
|---|---|
| `experiment.name` | 换成唯一模型名，它也是输出目录名 |
| `experiment.languages` | 只保留这个模型真正需要的语言 |
| `generation.text_manifest` | 指向待生成的文本 CSV |
| `generation.voice` | 选择 Prompt 设计或上传参考音频 |

### 第二层：根据机器和训练效果微调

这些参数都在外层配置的 `training` 中，可以为 `train1.json`、`train2.json`
分别设置：

| 参数 | 用途 | 推荐调整方式 |
|---|---|---|
| `batch_size` | 每批样本数 | OOM 时优先减半 |
| `epochs` | 最大训练轮数 | 小数据先用 10～100 验证 |
| `checkpoint_every_steps` | 周期 checkpoint | 调试 100～500，正式 5000 |
| `num_workers` | 数据读取进程 | macOS 先用 0，Linux 可试 2～8 |
| `seed` | 随机种子 | 复现实验时保持不变 |
| 两个 learning rate | GAN 学习速度 | 首轮保持默认，GAN 失衡后再调 |

### 一份完整的普通用户配置

`training_configs/*.json` 仅保留必填项和经常微调的项：

```json
{
  "_comment": "一个配置训练一个独立模型 / One config trains one independent model",
  "extends": "../configs/internal/pipeline_defaults.json",
  "experiment": {
    "_comment": "name 自动决定输出目录 / name automatically determines output directories",
    "name": "model_1",
    "_comment_languages": "顺序决定 language ID / Order defines language IDs",
    "languages": ["zh", "en", "ja", "ko", "fr", "es", "pt"],
    "device": "auto",
    "initialization": {
      "mode": "scratch",
      "checkpoint": null
    }
  },
  "generation": {
    "_comment": "enabled=false 时使用自备音频 / Use your own audio when enabled=false",
    "enabled": true,
    "qwen_runtime": "installed",
    "text_manifest": "datasets/texts.example.csv",
    "voice": {
      "_comment": "design 根据 prompt 设计音色 / design creates a voice from the prompt",
      "mode": "design",
      "speaker": "voice_01",
      "prompt": "A warm and natural adult voice with clear pronunciation.",
      "reference_text": "Hello, this is a reusable reference voice.",
      "reference_language": "en"
    }
  },
  "training": {
    "_comment": "显存不足时先降低 batch_size / Reduce batch_size first on OOM",
    "batch_size": 8,
    "learning_rate_generator": 0.0002,
    "learning_rate_discriminator": 0.0002,
    "epochs": 1000,
    "checkpoint_every_steps": 5000,
    "num_workers": 0,
    "seed": 1337
  }
}
```

以 `_comment` 开头的字段只是说明，不参与训练。真实示例文件已经全部加入中英
双语说明，可以直接复制后修改。

需要上传参考音频时，复制
[clone.example.json](training_configs/clone.example.json)。

`experiment.name` 是模型/实验的唯一名称，会同时决定训练与导出目录。
名称只能包含字母、数字、`.`、`_`和 `-`，不能包含路径分隔符。

| 实验参数 | 用途 |
|---|---|
| `name` | 模型名称，也是输出目录名 |
| `languages` | 该模型实际训练和导出的语言列表，顺序决定 language ID |
| `device` | `auto`、`cuda`、`cuda:0`、`mps` 或 `cpu` |
| `initialization.mode` | `scratch`、`resume` 或 `expand_speakers` |
| `initialization.checkpoint` | 续训/扩展时的旧 checkpoint 目录 |

默认路径全部由 `name` 推导：

```text
datasets/<name>/metadata.phonemes.csv
runs/<name>/
artifacts/<name>/
```

只有要共享数据或改变存储根目录时，才需要在专家配置中覆盖
`dataset_root`、`metadata`、`run_root` 或 `artifact_root`。

### 语言选项

支持的语言代码：

```text
zh  中文
en  英语
ja  日语
ko  韩语
fr  法语
es  西班牙语
pt  葡萄牙语
```

七语模型：

```json
"languages": ["zh", "en", "ja", "ko", "fr", "es", "pt"]
```

只训练英、法、西、葡：

```json
"languages": ["en", "fr", "es", "pt"]
```

Qwen 样本生成会从文本 CSV 中只选取这些语言。VITS 会根据列表创建
对应数量的 language embedding，checkpoint 和 ONNX 也只包含这些语言配置。
列表不能为空、不能重复。续训或扩音色时必须保持与旧 checkpoint 完全相同的
语言及顺序。

显存参考起点：

| 环境 | `batch_size` 建议 |
|---|---:|
| CPU 烟雾测试 | 1 |
| Apple Silicon MPS | 2～8 |
| 12～16 GB CUDA GPU | 4～8 |
| 24 GB CUDA GPU | 8～16 |

### 第三层：普通用户尽量不要调整

复杂参数放在 `configs/internal/`和 `configs/system/`。普通用户尽量不要调整：

- `hidden_channels`
- `latent_channels`
- `conditioning_channels`
- Text Encoder 层数和 heads
- Flow 层数
- Decoder channels
- upsample rates 和 kernels
- sample rate、FFT、hop length
- language/speaker embedding 维度
- phoneme 数据契约
- frontend provider、七语 eSpeak voice 和严格 fallback 检查
- Qwen `max_new_tokens`、top-k/top-p、temperature 和 subtalker 采样
- Qwen dtype、attention backend、生成 batch size
- 自动流水线的各阶段开关

这些参数会改变模型结构、checkpoint 兼容性、ONNX 大小或移动端前端协议。
除非准备重新训练全部模型和重新做真机基准，否则不要修改。

必须满足的结构关系：

```text
hidden_channels % text_encoder_heads == 0
product(upsample_rates) == audio.hop_length
spec_channels == n_fft / 2 + 1
```

其中 `vocab_size`、`num_languages`、`num_speakers` 和 `spec_channels` 会在训练启动时
根据实际词表、`experiment.languages`、metadata 和 FFT 配置自动覆盖，
不需要在专家配置中手动计算。

专家可复制 `configs/internal/pipeline_defaults.json` 和系统架构文件，
然后让外层配置继承新文件。如确实要设计新架构：

```bash
cp configs/system/vits_mobile_architecture.json \
   configs/system/vits_custom_architecture.json
```

再创建单独的用户配置继承它，不要直接修改公共预设。

## 用 Qwen 生成训练样本

生成文本 CSV 只需两列：

```csv
text,language
你好，欢迎使用多语言语音系统。,zh
Hello, welcome to the multilingual speech system.,en
```

支持 `zh en ja ko fr es pt`。使用一个参考音色生成全部语言，生成器会写入：

```text
datasets/<name>/
├── references/              # 设计或上传的参考音色副本
├── wavs/<speaker>/         # 已重采样的 PCM16 单声道 WAV
└── metadata.csv
```

### 方式一：用 Prompt 设计音色

```json
"voice": {
  "mode": "design",
  "speaker": "voice_01",
  "prompt": "A warm, natural adult voice with clear pronunciation and calm delivery.",
  "reference_text": "Hello, this is the reusable reference voice.",
  "reference_language": "en"
}
```

这不是每句直接重新设计音色。程序依照 Qwen 官方 README 的
Voice Design then Clone 流程：

```text
VoiceDesign.generate_voice_design()
  → 生成一条 canonical reference
  → Base.create_voice_clone_prompt()
  → Base.generate_voice_clone() 在全部文本中复用同一 prompt
```

### 方式二：上传参考音色

```json
"voice": {
  "mode": "clone",
  "speaker": "voice_01",
  "reference_audio": "datasets/references/voice_01.wav",
  "reference_text": "这里必须填录音中完全对应的文字。",
  "x_vector_only_mode": false
}
```

`reference_text` 应与录音完全对应。只有设置 `x_vector_only_mode=true` 时可以省略，
但 Qwen 官方提示这可能降低克隆质量。

### Qwen 运行时与下载开关

| 配置 | 效果 |
|---|---|
| `generation.enabled=true` | 流水线生成 Qwen 样本 |
| `generation.enabled=false` | 跳过生成，使用自己准备的 metadata/WAV |
| `qwen_runtime="installed"` | 默认，使用官方 `qwen-tts==0.1.1` Python 包 |
| `qwen_runtime="source"` | 专家模式，使用 `qwen_source_path` 指向的本地源码 |
| `auto_download_models=true` | 缺少权重时下载到项目 `models/qwen/` |
| `auto_download_models=false` | 缺少权重时立即报错，不访问网络 |

普通用户不需要手动 clone Qwen。安装并检查运行时：

```bash
.venv/bin/pip install -e '.[qwen]'
PYTHONPATH=src .venv/bin/python -m tts_trainer qwen-runtime
```

只有要修改或调试 Qwen 源码时才使用专家覆盖：

```json
"generation": {
  "qwen_runtime": "source",
  "qwen_source_path": "/absolute/path/to/Qwen3-TTS"
}
```

程序只切换 Python 运行时来源；模型权重仍统一检测和保存到项目自己的
`models/qwen/`。

只生成样本，不开始训练：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer generate-samples \
  --config training_configs/train1.json
```

也可直接执行脚本：

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_samples.py \
  --config training_configs/train1.json
```

## 按配置自动执行

一键入口：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/train1.json
```

等价脚本：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py \
  --config training_configs/train1.json
```

可在专家配置或外层配置覆盖阶段：

```json
"pipeline": {
  "generate_samples": true,
  "phonemize": true,
  "validate": true,
  "train": true,
  "export": true,
  "validate_onnx": true
}
```

任何阶段失败都会立即停止，不会用错误数据继续训练。成功后记录写入：

```text
runs/<name>/pipeline-report.json
```

调试整条流水线时可限制 VITS 步数：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/train1.json \
  --max-steps 10
```

## 开始训练

### 模型命名与目录隔离

一个用户配置对应一个命名模型。例如：

```text
training_configs/train1.json  -> experiment.name = model_1
training_configs/train2.json  -> experiment.name = model_2
```

分别运行就会得到两个完全隔离的模型：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/train1.json

PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/train2.json
```

```text
train1.json → datasets/model_1/ → runs/model_1/ → artifacts/model_1/
train2.json → datasets/model_2/ → runs/model_2/ → artifacts/model_2/
```

可以先初始化目录，不会开始训练：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer init-experiment \
  --config training_configs/train1.json
```

`train-vits` 启动时也会自动执行这一步。目录结构为：

```text
datasets/model_1/
└── metadata.phonemes.csv      # 未显式配置 metadata 时的默认位置

runs/model_1/                  # 中间态，必须保留
├── resolved-config.json      # 本次实际生效的完整配置
├── run-layout.json           # 路径和初始化信息
├── vocab.json
├── checkpoints/
│   ├── last/
│   └── step-000005000/
└── logs/                     # 预留给训练日志/TensorBoard

artifacts/model_1/             # 可发布的 ONNX 资源
```

`runs/<name>/` 是后续续训、增加音色、蒸馏、剪枝和 QAT 的源文件，
不能只保留 ONNX。

### 先做烟雾测试

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer train-vits \
  --config training_configs/train1.json \
  --output runs/smoke \
  --device cpu \
  --max-steps 10
```

确认 `runs/smoke/checkpoints/last/` 能正常生成后，再开始正式训练。
命令行的 `--metadata`、`--output`和 `--device` 只用于临时覆盖配置。

### 正式训练

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer train-vits \
  --config training_configs/train1.json
```

`--device auto` 的选择顺序为 CUDA、Apple MPS、CPU。也可以明确传入：

```text
--device cuda
--device mps
--device cpu
```

### 同时管理多个模型

复制用户配置，为每个模型设置不同的 `experiment.name`、`languages`、
音色和设备：

```bash
cp training_configs/train1.json training_configs/my_reader.json
```

顺序训练多个配置（最稳妥）：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer train-many \
  training_configs/train1.json \
  training_configs/train2.json
```

`train-many` 只执行 VITS 训练阶段，适合已经准备好
`metadata.phonemes.csv` 的任务。如果还需要 Qwen 生成、音素化和 ONNX 导出，
请像上面那样分别对 `train1.json`、`train2.json` 执行 `run-pipeline`。

并行训练：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer train-many \
  training_configs/train1.json \
  training_configs/train2.json \
  --max-parallel 2
```

默认 `--max-parallel 1`。单张 GPU 通常不建议并行；多张 GPU 可在各配置中
分别设置 `cuda:0`、`cuda:1`。程序会拒绝同一批任务中重复的模型名。

### 从 checkpoint 完整续训

参考 [resume.example.json](training_configs/resume.example.json)。
保持模型结构、tokens 和 speakers 不变，然后设置：

```json
{
  "experiment": {
    "name": "model_1_resume",
    "languages": ["zh", "en", "ja", "ko", "fr", "es", "pt"],
    "initialization": {
      "mode": "resume",
      "checkpoint": "runs/model_1/checkpoints/last"
    }
  },
  "training": {
    "epochs": 1500
  }
}
```

`resume` 恢复 Generator、Discriminator、两个 Optimizer、epoch 和 step。
`training.epochs` 是总目标 epoch，必须大于 checkpoint 中已完成的 epoch。

### 基于旧模型增加新音色

参考 [add-speaker.example.json](training_configs/add-speaker.example.json)：

```json
{
  "experiment": {
    "name": "model_1_voice02",
    "languages": ["zh", "en", "ja", "ko", "fr", "es", "pt"],
    "initialization": {
      "mode": "expand_speakers",
      "checkpoint": "runs/model_1/checkpoints/last"
    }
  },
  "generation": {
    "include_metadata": ["datasets/model_1/metadata.csv"],
    "voice": {
      "speaker": "voice_02"
    }
  }
}
```

`expand_speakers` 会：

- 保留旧 token 和 speaker ID，新 token/音色只追加到末尾。
- 复制所有尺寸兼容的旧 Generator 权重。
- 扩展 token embedding 和 speaker embedding，保留旧行并初始化新行。
- 不恢复旧 GAN Optimizer，使用新配置的小学习率微调。

`generation.include_metadata` 会把旧 metadata 中的样本合并到新数据集，
再加入 Qwen 生成的新音色。如果只用新音色，旧音色可能灾难性遗忘，
训练器会给出警告。扩展时结构参数必须与旧模型兼容。

## 训练输出

```text
runs/model_1/
├── vocab.json
├── resolved-config.json
├── run-layout.json
└── checkpoints/
    ├── last/
    │   ├── training-state.pt
    │   └── metadata.json
    └── step-000005000/
        ├── training-state.pt
        └── metadata.json
```

`training-state.pt` 保存 Generator、Discriminator、两个 Optimizer、epoch 和 step，
是后续继续训练、蒸馏、剪枝和 QAT 的源文件。

`metadata.json` 保存模型结构、tokens、language map、speaker map 和指标。
它也保存训练时使用的 frontend contract；不要在发布时只复制 `model.onnx`。

`last/` 每个 epoch 更新；`step-xxxxxxxxx/` 是不会覆盖的周期快照。
当前尚未自动选择 `best/`，请保留全部重要 checkpoint。

## 导出 ONNX

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer export-vits \
  --config training_configs/train1.json \
  --validate-runtime
```

`--config` 会自动使用 `runs/<name>/checkpoints/last`、配置中的采样率以及
`artifacts/<name>/`。也可以用 `--checkpoint`、`--output`和 `--sample-rate`
手动导出某个历史快照。

输出：

```text
artifacts/model_1/
├── model.onnx
├── model.onnx.json
├── frontend.json
├── tokens.json
└── tokens.txt
```

- `model.onnx`：移动端推理图
- `model.onnx.json`：采样率、默认 scales 和 voice profile 映射
- `frontend.json`：训练时的规范化、G2P provider、引擎版本和逐语言 voice
- `tokens.json`：App/Python 使用的完整词表
- `tokens.txt`：Piper/sherpa 风格词表

部署接口：

```text
input, input_lengths, scales, sid -> output
```

项目将 language 和 speaker 分开训练，但在部署时组合成：

```text
sid = speaker_id * num_languages + language_id
```

所有实际 `sid` 都会写入 `model.onnx.json`。

## 验证 ONNX 音频

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer synthesize-onnx \
  --model-dir artifacts/model_1 \
  --text "Hello world" \
  --language en \
  --speaker voice_01 \
  --output artifacts/test/en_voice01.wav
```

## Qwen 模型管理

Qwen 权重统一放在项目的 `models/qwen/`，不会静默下载到用户全局缓存。

检查状态，不下载：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer models status
```

需要时下载：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer models ensure voice-design-1.7b
PYTHONPATH=src .venv/bin/python -m tts_trainer models ensure base-1.7b
```

获取确定的本地路径：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer models path base-1.7b
```

## 已知限制

1. eSpeak-ng 七语路由和版本契约已经完成，但它仍是工程基线，不代表中日韩
   已达到产品级多音字、音高重音和音变质量。
2. eSpeak 处理含汉字的日语时可能退回英文拼读；程序默认检测并拒绝该结果。
3. stock sherpa-onnx Piper/VITS 前端一次配置一个 eSpeak voice；一个 ONNX 在请求间
   切换七种语言需要 App/native 前端路由或 sherpa 适配。
4. 当前没有验证集、自动 MOS、ASR 回识别和 speaker similarity 评测。
5. 当前没有自动 best checkpoint、蒸馏、剪枝和 INT8。
6. 当前 VITS 实现是移动端工程基线，正式音质必须用真实数据和母语者试听验证。

不要因为训练 loss 下降就直接发布模型。至少需要七种语言母语者试听，并检查
数字、日期、人名、长句、疑问句和 App 实际业务文本。

## 开发与测试

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m compileall -q src tests deployment
git diff --check
```

架构细节见 [docs/architecture.md](docs/architecture.md)，移动端前端边界见
[docs/mobile_runtime.md](docs/mobile_runtime.md)。

## 贡献方向

欢迎优先贡献：

- 中文多音字和文本规范化
- 日语 OpenJTalk 前端及移动端一致性
- 韩语 G2P
- Qwen 样本 ASR/声纹评分、多候选自动筛选
- ASR/CER/WER 与 speaker similarity 自动评测
- best model 选择
- 蒸馏和选择性 INT8
- Android/iOS 示例

提交修改前请运行全部测试，并避免提交模型权重、训练音频、`runs/`、`artifacts/`
或未经授权的声音数据。

## 声音与数据合规

只使用有权处理和训练的文本、录音与参考声音。克隆真人声音前应取得明确授权，
并根据产品所在地要求提供数据删除、授权撤回和生成内容标识机制。第三方代码、
模型权重和数据集分别遵循其自身许可证；复制到仓库中不代表许可证自动统一。

## 许可证与第三方软件

TTSTRAINER 自有源码采用 Apache-2.0，见 [LICENSE](LICENSE)。Qwen3-TTS 源码不
包含在本仓库中，通过官方 PyPI 包作为可选依赖安装；第三方软件和模型说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
