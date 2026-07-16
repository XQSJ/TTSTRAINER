# tts-trainer

## 最快开始：改一份配置，运行几条命令

先复制示例配置：

```bash
cp training_configs/train1.json training_configs/my_model.json
```

### 训练文本从哪里来

有两种最常用的方式。

方式一，先用项目内置模板自动生成，用来验证整个流程。`train1.json` 默认已经开启：

```json
"text_generation": {
  "enabled": true,
  "provider": "builtin",
  "sentences_per_language": 20
}
```

执行完整流水线时会自动生成；安装下面的依赖后，也可以先单独生成并查看：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer generate-texts --config training_configs/my_model.json
```

生成文本放在公共语料库 `datasets/text_corpora/<corpus-id>/texts.csv`，不会放进
`datasets/my_model/`。`corpus-id` 由语言、provider、seed 和过滤配置计算；另一个
模型使用相同配置时会直接命中缓存，不会再次生成。内置模板不需要下载文本大模型，
但内容只是覆盖数字、日期、问句等流程种子，不足以训练产品级声音。

方式二，正式训练使用自己的业务文案、已获得许可的语料或人工整理文本，保存为
UTF-8 CSV。每种启用语言至少准备足够多的句子：

```csv
text,language
你好，欢迎使用语音助手。,zh
今天下午三点有一场会议。,zh
Hello, welcome to the voice assistant.,en
Your meeting starts at three this afternoon.,en
```

例如保存为 `datasets/my_texts.csv`，然后在配置中改为：

```json
"text_generation": {
  "enabled": false
},
"generation": {
  "text_manifest": "datasets/my_texts.csv"
}
```

CSV 里的语言必须包含 `experiment.languages` 选择的每一种语言。项目也支持从已有
CSV 或 OpenAI-compatible 文本服务自动扩写，配置示例见
`training_configs/auto-text.example.json`；无论文本来源是什么，正式数据都应经过
版权确认、去重和母语者审核。

打开 `training_configs/my_model.json`，第一次只改下面这些内容：

```json
{
  "experiment": {
    "name": "my_model",
    "languages": ["zh", "en"]
  },
  "text_generation": {
    "enabled": true,
    "sentences_per_language": 20
  },
  "generation": {
    "voice": {
      "mode": "design",
      "speaker": "voice_01",
      "prompt": "A warm, natural adult voice with clear pronunciation."
    }
  },
  "training": {
    "batch_size": 8,
    "epochs": 1000
  }
}
```

不要用上面的片段覆盖整个文件，只修改原配置中的同名字段。`name` 是模型和输出
目录名，`languages` 是训练语言，`text_generation` 自动准备流程验证文本，`voice` 决定音色；
显存不足时把 `batch_size` 改成 `4` 或 `2`。

安装并开始训练：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[qwen,export,japanese,asian]'
PYTHONPATH=src .venv/bin/python -m tts_trainer language-check --config training_configs/my_model.json
PYTHONPATH=src .venv/bin/python -m tts_trainer run-pipeline --config training_configs/my_model.json
```

这里不要给整条安装命令增加 `--no-build-isolation`。日语依赖 `pyopenjtalk`
目前从源码构建，并通过隔离构建环境中的 `setuptools_scm` 生成版本号；禁用隔离但
没有预装该构建依赖时，pip 会把 0.4.1 错误识别为 0.0.0。若已经遇到
`expected '0.4.1', but metadata has '0.0.0'`，执行：

```bash
.venv/bin/pip install 'setuptools_scm>=8' 'cython>=0.29.16' cmake
.venv/bin/pip install 'pyopenjtalk==0.4.1'
.venv/bin/pip install -e '.[qwen,export,japanese,asian]'
```

Linux 没有可用的预编译 wheel 时还需要系统 C/C++ 编译器。Conda 用户应直接将
上面的 `.venv/bin/pip` 替换为 `python -m pip`，不要在 Conda 环境内再创建一层
venv。

训练结果在 `runs/my_model/checkpoints/best/`，移动端资源在
`artifacts/my_model/`。Windows 请将 `.venv/bin/python` 换成
`.venv\Scripts\python.exe`。更换音色、自动生成文本、续训、增加音色和专家参数见
后面的完整说明。

---

这是一个面向移动端离线部署的多语言、多音色 VITS 训练工程。即使你没有语音
模型训练经验，也可以从一份带中英双语说明的 JSON 配置开始，完成样本生成、
VITS 训练、checkpoint 保存和 ONNX 导出。

整个过程可以理解为：

```text
准备所选语言文本
  → Qwen 设计/克隆音色并生成 WAV（可跳过）
  → 冻结音素
  → 训练 Multilingual VITS
  → 保存可恢复 checkpoint
  → 导出动态 ONNX
  → ONNX Runtime 验证
  → 集成 Android / iOS
```

内置语言注册表覆盖 Qwen3-TTS 官方的 10 种语言：中文、英语、日语、韩语、
德语、法语、俄语、葡萄牙语、西班牙语和意大利语。也可以注册使用自备音频的
其他已注册前端支持的语言。模型内部将
`language embedding` 和 `speaker embedding` 分开，所以既可以训练一个固定
音色，也可以在同一个模型中继续增加音色。

> 项目目前处于工程验证阶段。训练、恢复、扩音色、验证集、best checkpoint、
> 基础音频质检和 ONNX 导出已经跑通；中日韩均使用逐语言专用 G2P。
> ASR/声纹质检为可选功能，蒸馏、INT8 与真机基准仍在完善。正式投入大规模训练前，请先
> 阅读“已知限制”。

## 第一次使用：按这 5 步做

### 第 1 步：选择一份配置

所有给普通用户修改的配置都在 `training_configs/`，一份 JSON 就代表一个训练
任务。先根据目标选择：

| 你的目标 | 从哪个文件开始 |
|---|---|
| 用 Prompt 设计一个新音色并从零训练 | `training_configs/train1.json` |
| 再训练一个互不影响的模型 | `training_configs/train2.json` |
| 训练包含德语、俄语和意大利语的模型 | `training_configs/european.example.json` |
| 自动生成文本后训练 | `training_configs/auto-text.example.json` |
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

开始下载或训练前，先检查配置选择的语言：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer language-check \
  --config training_configs/my_reader.json
```

只有 Teacher 映射和实际 G2P 音素烟雾测试全部通过，自动流水线才会继续。

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

### 第 3 步：选择文本来源

最简单的流程验证可以直接启用内置文本生成器：

```json
"text_generation": {
  "enabled": true,
  "provider": "builtin",
  "sentences_per_language": 100
}
```

完整示例见
[auto-text.example.json](training_configs/auto-text.example.json)。内置模板不需要
下载文本模型，但只适合烟雾测试和覆盖种子，不应作为产品语料的唯一来源。

也可以编辑 `datasets/texts.example.csv`，或者新建 CSV 并把路径填入
`generation.text_manifest`，同时保持 `text_generation.enabled=false`：

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
- 英语及欧洲语言的默认音素化需要 `espeak-ng`
- 选择日语时需要安装 `japanese` extra（pyopenjtalk/Open JTalk）
- 选择中文或韩语时需要安装 `asian` extra（Piper Plus G2P；韩语包含 MeCab）

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
.venv/bin/pip install -e '.[qwen,export,japanese,asian,dev]'
```

项目不复制 Qwen3-TTS 源码。`qwen` extra 会安装固定版本的官方 `qwen-tts`
Python 包。只使用自己准备的 WAV、不需要 Qwen 生成样本时，可以改为：

```bash
.venv/bin/pip install -e '.[export,japanese,asian,dev]'
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
  → 按配置生成/导入训练文本和质检报告（可跳过）
  → 检查逐语言 Teacher/G2P/voice 并显示结果
  → 检查/按需下载项目内 Qwen 权重
  → 生成 PCM16 训练 WAV 和 metadata.csv
  → 冻结音素
  → 信号质检并固定训练/验证集
  → VITS 训练、验证并保存 best/last
  → ONNX 导出与运行时验证
```

Qwen 权重不会在安装或读取配置时下载。只有真正执行样本生成，并且
`generation.auto_download_models=true` 时，缺少的权重才会下载到项目自己的
`models/qwen/`，后续始终复用这里的文件。

日语的 Open JTalk 词典也遵循项目内资源策略：首次实际检查或音素化日语时，程序
检测 `models/frontends/openjtalk/`；缺少时才下载约 24 MB 的官方词典并校验
SHA-256。可提前查看或准备：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer frontends status openjtalk
PYTHONPATH=src .venv/bin/python -m tts_trainer frontends ensure openjtalk
```

韩语的 CMU 发音词典同样只保存在项目内，缺少时下载约 0.9 MB 并校验
SHA-256。中文 Piper Plus G2P 不需要下载额外词典：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer frontends status korean
PYTHONPATH=src .venv/bin/python -m tts_trainer frontends ensure korean
```

### 第 5 步：找到结果

运行成功后，重点看两个目录：

```text
runs/<name>/checkpoints/best/training-state.pt  # 当前验证集最佳中间态
runs/<name>/checkpoints/last/training-state.pt  # 最近一次可恢复中间态
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
| 按配置动态创建 language embedding | 已完成 |
| 十语配置注册表与自定义外部数据语言 | 已完成 |
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
| builtin/file/OpenAI-compatible 自动文本生成 | 已完成 |
| 按语言路由 G2P、烟雾测试与前端版本契约 | 已完成 |
| 日语 Open JTalk 汉字 G2P | 已完成（训练端）；移动端需集成同版原生前端 |
| 中文 Piper Plus pinyin→IPA、声调 G2P | 已完成（训练端）；移动端需跑 conformance |
| 韩语 Piper Plus g2pk2/MeCab 音韵 G2P | 已完成（训练端）；移动端需跑 conformance |
| 分层验证集、best checkpoint、信号音质门禁 | 已完成 |
| ASR/CER/WER 与 ECAPA 声纹质检 | 已完成（可选，默认不下载模型） |
| 蒸馏、结构化剪枝、选择性 INT8、真机基准 | 可选发布优化，尚未自动化 |

### 哪些是必须的

| 阶段 | 是否必须 | 什么时候做 |
|---|---|---|
| 逐语言 G2P 与 conformance | 必须 | 训练前和每次移动端发布前 |
| 信号质检、验证集、best checkpoint | 必须 | 每次正式训练，流水线默认执行 |
| 每种语言母语者试听 | 必须 | 发布前；自动指标不能替代 |
| ASR 回识别、声纹相似度 | 推荐 | 数据量变大、批量合成或多音色时启用 |
| 真机 RTF、内存、首句延迟和发热 | 移动发布必须 | 得到 FP32 ONNX 后在目标机型测 |
| 蒸馏、剪枝、INT8 | 条件性 | 只有 FP32 在目标机不达标时再做 |

因此首版训练不应等待蒸馏或剪枝。先得到可试听、可验证的 FP32 ONNX；若真机已经
满足速度和包体目标，完全可以不做剪枝。中间 PyTorch checkpoint 必须保留，以便
以后扩音色、增加数据或做压缩微调。

## 项目目录

```text
tts_trainer/
├── training_configs/           # 用户的所有训练任务配置
│   ├── train1.json             # 训练 model_1
│   ├── train2.json             # 训练 model_2
│   ├── european.example.json   # 德/俄/意等欧洲语言组合
│   ├── auto-text.example.json  # 自动生成文本后训练
│   ├── clone.example.json      # 上传参考音色
│   ├── resume.example.json     # 恢复完整训练状态
│   └── add-speaker.example.json # 扩展新音色
├── configs/
│   ├── internal/             # Qwen 生成细节和自动流水线默认值
│   ├── system/               # 语言注册表和 VITS 模型结构契约
│   └── models.json           # Qwen 模型注册
├── datasets/
│   ├── texts.example.csv      # 给 Qwen 生成的文本清单
│   └── metadata.example.csv
├── scripts/
│   ├── generate_texts.py       # 只执行自动文本阶段
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

按配置建立多语言 G2P 基线。日语走 Open JTalk，中韩走 Piper Plus，其余内置
语言走 eSpeak-ng：

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

音素以空白分隔的 token unit 保存：eSpeak IPA 当前按 Unicode codepoint，
Open JTalk 则保留 `ch`、`sh`、`N` 等完整日语音素。`<space>` 表示真实空格 token。
命令还会在同一
目录生成 `frontend.lock.json`，逐语言记录 provider、eSpeak voice 或 Open JTalk
词典、引擎版本、规范化规则和 token 格式。训练 checkpoint 和 ONNX 导出会继续
携带这份契约。

这个路由使用新的 `routed-phoneme-units-v1` token 契约。由旧版纯 eSpeak 前端
训练的 checkpoint 不能直接续训；应重新音素化数据并创建新模型，避免相同 token ID
在新旧版本中代表不同发音。

查看当前机器最终生效的多语言前端，不生成音频或下载模型：

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

### 如何确认一种语言的 Teacher 和 G2P 可用

列出注册表中的全部语言并执行本机 G2P 烟雾测试，不会下载 Qwen 模型：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer languages \
  --config training_configs/train1.json
```

只检查当前模型选择的语言：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer languages \
  --config training_configs/train1.json \
  --selected-only
```

也可以点名检查：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer language-check de ru it \
  --config training_configs/european.example.json
```

输出中的 `ready` 表示配置有合法 Teacher 映射，并且该语言配置的前端已经用真实
示例文本生成了非空音素；eSpeak 路由还会拒绝跨语言 fallback。它不等于产品级质量；
正式发布仍需母语者试听。

日语已默认改用 Open JTalk，不再把含汉字文本交给 eSpeak-ng。若日语显示
`failed` 且提示缺少 pyopenjtalk，请执行：

```bash
.venv/bin/pip install -e '.[japanese]'
```

pyopenjtalk 某些平台需要 CMake 和 C/C++ 编译器。未安装时程序不会退回 eSpeak，
因为静默 fallback 会制造错误训练 token。

中文和韩语默认使用 Piper Plus G2P。若显示缺少 `piper_plus_g2p`、`g2pk2`
或 `mecab`，请执行：

```bash
.venv/bin/pip install -e '.[asian]'
```

韩语第一次音素化会检测项目内 CMU 词典；允许自动下载时保存到
`models/frontends/korean/`，不写入用户全局 NLTK 缓存。

G2P 使用现有前端，不需要训练另一套 G2P 模型。默认路由为：

```text
zh=PiperPlus-Mandarin  en=en-us  ja=OpenJTalk  ko=PiperPlus-Korean  de=de
fr=fr-fr  ru=ru  es=es  pt=pt-br  it=it
```

eSpeak-ng 和 Open JTalk 都有原生实现；Piper Plus G2P 提供 Android/Kotlin、
iOS/Swift 等实现。移动端必须按 `frontend.json` 路由，并用
`frontend.conformance.json` 逐条验证音素与 token ID；Python 包本身不会塞进 App。

可在专家配置覆盖口音，例如欧葡：

```json
"frontend": {
  "provider": "language-router",
  "voices": {"pt": "pt-pt"}
}
```

修改 provider、voice、规范化规则或 token 语义后，不应继续复用不兼容的旧
checkpoint。程序会比较 `frontend.lock.json` 和 checkpoint 契约并拒绝静默混用。

训练流水线会在生成音频前自动运行同一套检查。某语言缺少 voice、Qwen Teacher
映射或者出现 G2P fallback 时，会输出 `language failed` 并停止，不会等训练到
中途才报错。

## 配置怎么改：从常用参数到专家参数

普通用户只需在 `training_configs/` 中新建或复制 JSON。
例如 [train1.json](training_configs/train1.json) 和 [train2.json](training_configs/train2.json)。
每个文件对应一个独立模型，并通过 `extends`
自动继承内部默认值：

```text
configs/system/language_registry.json              # 内置语言声明
                    ↓
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
| `log_every_steps` | 输出 GAN/Mel loss 的步数间隔 | 调试 1～10，正式 10～100 |
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
    "log_every_steps": 10,
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

模型私有路径由 `name` 推导：

```text
datasets/<name>/metadata.phonemes.csv
runs/<name>/
artifacts/<name>/
```

只有要共享数据或改变存储根目录时，才需要在专家配置中覆盖
`dataset_root`、`metadata`、`run_root` 或 `artifact_root`。

公共资源不跟模型名走：

```text
datasets/text_corpora/  # 自动生成/清洗后的共享文本，相同配置直接复用
models/qwen/            # Qwen Teacher 权重
models/frontends/       # 日语、韩语等前端词典
models/quality/         # 可选 ASR/声纹质检权重
```

只有音色参考、生成 WAV、metadata、checkpoint 和 ONNX 属于具体模型/数据任务。

### 语言选项

内置并可直接用于 Qwen 样本生成的语言代码：

```text
zh  中文
en  英语
ja  日语
ko  韩语
de  德语
fr  法语
ru  俄语
es  西班牙语
pt  葡萄牙语
it  意大利语
```

原七语模型：

```json
"languages": ["zh", "en", "ja", "ko", "fr", "es", "pt"]
```

只训练英、法、西、葡：

```json
"languages": ["en", "fr", "es", "pt"]
```

训练新增的德、俄、意三语：

```json
"languages": ["de", "ru", "it"]
```

完整示例见
[european.example.json](training_configs/european.example.json)。只要所选语言存在于
`datasets/texts.example.csv` 或自己的 `generation.text_manifest`，同一个一键入口
会自动完成筛选、Qwen 生成、G2P、动态 language embedding、训练和 ONNX 导出。

#### 注册 Qwen 十语之外的语言

如果使用自备 WAV 和 metadata，可以在外层训练配置添加语言声明。例如波兰语：

```json
{
  "language_registry": {
    "pl": {
      "name": "Polish",
      "teacher": null,
      "frontend": {"provider": "espeak-ng", "voice": "pl"},
      "smoke_text": "Dzień dobry, witamy w systemie głosowym."
    }
  },
  "experiment": {
    "name": "polish_reader",
    "languages": ["pl"]
  },
  "generation": {
    "enabled": false,
    "raw_metadata": "datasets/polish_reader/metadata.csv"
  }
}
```

`teacher=null` 表示项目不调用 Qwen，用户提供该语言的真实录音或其他合法 Teacher
产生的 WAV。配置仍会检查 eSpeak voice 和烟雾文本。如果目标语言无法由
eSpeak 正确音素化，需要先实现训练端和手机端一致的专用前端，不能仅绕过检查。

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
- language registry、frontend provider、eSpeak voice 和严格 fallback 检查
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

## 自动生成训练文本

只执行文本阶段，不会下载 Qwen-TTS 权重或生成音频：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer generate-texts \
  --config training_configs/auto-text.example.json
```

也可运行脚本：

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_texts.py \
  --config training_configs/auto-text.example.json
```

默认输出位于共享语料目录，不包含 `experiment.name`：

```text
datasets/text_corpora/<corpus-id>/
├── texts.csv
└── texts.report.json
```

`corpus-id` 默认由 provider、语言集合和完整生成配置指纹得到。流水线再次请求相同
语料时直接复用现有 CSV；不会因为 `experiment.name` 改变而重新生成。报告记录
配置指纹、逐语言目标数、通过数、拒绝原因和最多 20 条拒绝示例。

需要人为指定稳定名称时使用 `corpus_name`；同名但配置不一致时程序会拒绝覆盖：

```json
"text_generation": {
  "corpus_name": "seven_language_business_v1",
  "reuse": true,
  "overwrite": false
}
```

只有明确要重建同名语料时才临时设置 `overwrite=true`。

### `builtin`：内置确定性模板

```json
"text_generation": {
  "enabled": true,
  "provider": "builtin",
  "sentences_per_language": 100,
  "seed": 1337
}
```

相同配置和 seed 会得到相同文本，适合 CI、烟雾训练和补充数字/日期/金额覆盖。
它不是自然语料替代品；正式模型不要只使用模板数据。

### `file`：导入并清洗已有 CSV

```json
"text_generation": {
  "enabled": true,
  "provider": "file",
  "input": "datasets/my_source_texts.csv",
  "sentences_per_language": 5000
}
```

输入至少需要 `text,language` 两列，可以额外提供 `category,source`。程序只保留
`experiment.languages` 中的语言，并进行统一过滤、去重和限额。

### `openai_compatible`：调用文本大模型

```json
"text_generation": {
  "enabled": true,
  "provider": "openai_compatible",
  "endpoint": "https://your-text-service.example/v1",
  "model": "your-text-model",
  "api_key_env": "TEXT_LLM_API_KEY",
  "sentences_per_language": 5000,
  "batch_size": 50
}
```

密钥只从环境变量读取，不要写进 JSON：

```bash
export TEXT_LLM_API_KEY="..."
```

`api_key_env` 填的是环境变量名称，不是密钥值。为兼容常见服务配置，也接受
`openai_compatible` 子对象和其中的 `base_url` 别名，但推荐使用上面的扁平写法；
启动流水线时会在生成文本或下载模型前检查这些必填项，并拒绝疑似直接写入的密钥。
HTTP 401/403 通常表示 Key 无效，或者 Key 所属套餐与 endpoint 不匹配；TLS/EOF
错误则应检查服务器的 `HTTPS_PROXY` 和 `NO_PROXY`。程序会保留服务端错误摘要，
但不会把请求中的 Key 写入日志。

无需认证的本地 OpenAI-compatible 服务可以设置 `"api_key_env": null`。接口应
提供 `/chat/completions`，并返回标准 `choices[0].message.content`；内容必须是
由 `text` 和 `category` 组成的 JSON 数组。

常用质量参数：

```json
"filters": {
  "min_characters": 5,
  "max_characters": 180,
  "deduplicate": true,
  "reject_mixed_language": true,
  "require_g2p_pass": false
}
```

`reject_mixed_language` 只做基础 Unicode 文字脚本检查，不是完整语言识别。
`require_g2p_pass=true` 会逐句实际调用各语言路由后的 G2P，准确但明显更慢。
日语汉字由 Open JTalk 处理。产品数据仍需独立语言识别、版权检查和母语者抽检。

## 用 Qwen 生成训练样本

生成文本 CSV 只需两列：

```csv
text,language
你好，欢迎使用多语言语音系统。,zh
Hello, welcome to the multilingual speech system.,en
```

支持 `zh en ja ko de fr ru pt es it`。使用一个参考音色生成全部语言，生成器会写入：

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
  "generate_texts": true,
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

终端会持续显示阶段和训练状态，例如：

```text
INFO | pipeline | language ready code=de teacher=qwen:German g2p=espeak-ng:de
INFO | pipeline | stage=generate_texts status=started
INFO | text_generation | text generation completed ...
INFO | pipeline | stage=generate_samples status=started
INFO | sample_generation | generation jobs total=1000 pending=960 cached=40
INFO | trainer | epoch=1 step=10 generator=... discriminator=... mel=...
INFO | trainer | checkpoint step=5000 status=saving
INFO | pipeline | stage=export status=completed model=artifacts/.../model.onnx
```

日志级别由内部默认配置的 `logging.level` 控制，也可临时使用环境变量：

```bash
TTS_TRAINER_LOG_LEVEL=DEBUG PYTHONPATH=src .venv/bin/python -m tts_trainer \
  run-pipeline --config training_configs/train1.json
```

训练损失输出频率由外层配置的 `training.log_every_steps` 控制。

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
├── splits/                   # 固定的 train/validation CSV 与指纹
├── quality/                  # 信号、可选 ASR/声纹质检报告
├── checkpoints/
│   ├── best/                 # validation.metric 最优
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
├── splits/
│   ├── train.csv
│   ├── validation.csv
│   └── split-report.json
├── quality/
│   ├── audio-quality-report.json
│   └── semantic-quality-report.json  # 仅启用可选语义质检时存在
└── checkpoints/
    ├── best/
    │   ├── training-state.pt
    │   └── metadata.json
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

`last/` 每个 epoch 更新，适合断点恢复；`best/` 只在配置的验证指标改善时更新，
默认用于导出；`step-xxxxxxxxx/` 是不会覆盖的周期快照。扩音色或续训通常从
`last/` 开始，发布通常从 `best/` 开始。

## 导出 ONNX

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer export-vits \
  --config training_configs/train1.json \
  --validate-runtime
```

`--config` 默认使用 `runs/<name>/checkpoints/best`（没有 best 时回退 last）、
配置中的采样率以及 `artifacts/<name>/`。也可以用 `--checkpoint`、`--output`和 `--sample-rate`
手动导出某个历史快照。

输出：

```text
artifacts/model_1/
├── model.onnx
├── model.onnx.json
├── frontend.json
├── frontend.conformance.json
├── tokens.json
└── tokens.txt
```

- `model.onnx`：移动端推理图
- `model.onnx.json`：采样率、默认 scales 和 voice profile 映射
- `frontend.json`：训练时逐语言使用的 G2P provider、引擎版本、voice/词典
- `frontend.conformance.json`：每种语言的代表性「原文 → 音素 → token ID」校验样例
- `tokens.json`：App/Python 使用的完整词表
- `tokens.txt`：Piper/sherpa 风格词表

不同 G2P provider 不会产生多个声学模型。`frontend.json` 只负责把原始文字路由成
统一 token ID；所有语言和音色最终仍进入同一个 `model.onnx`。部署时必须整体携带
上述六个文件，不能只复制 ONNX。

导出后先验证当前机器的前端实现是否与训练时完全一致：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer verify-frontend \
  --model-dir artifacts/model_1
```

成功时输出 `"ready": true`。工具会同时检查 eSpeak/Open JTalk/Piper Plus 版本、
音素序列和 token ID；任何一项不一致都会以非 0 状态退出。使用 Open JTalk
用户词典的模型还需要传入 `--user-dictionary /path/to/user.dic`。
移动端应使用同一份 conformance 数据对 native 前端做相同的自测。

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

## 数据质检与可选模型

训练前默认执行不需要模型的信号质检：时长、RMS、削波、DC 偏移、首尾静音和
文本/音素语速。可以单独运行：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer quality-check \
  --config training_configs/train1.json
```

更严格的 ASR 回识别（中文/日语 CER，其余 WER）和 ECAPA 声纹相似度默认关闭，
因为它们增加依赖、耗时和约 575 MB 的本地模型。需要时安装并显式下载：

```bash
.venv/bin/pip install -e '.[quality]'
PYTHONPATH=src .venv/bin/python -m tts_trainer quality-models status
PYTHONPATH=src .venv/bin/python -m tts_trainer quality-models ensure asr-small
PYTHONPATH=src .venv/bin/python -m tts_trainer quality-models ensure speaker-ecapa
```

模型只保存到 `models/quality/`。然后在用户配置覆盖
`quality.semantic.enabled=true`；若没有给某个 speaker 配参考 WAV，声纹检查会
明确失败而不是猜测。阈值等专家参数在 `configs/internal/pipeline_defaults.json`。

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

1. 多前端路由和版本契约已经完成；日语使用 Open JTalk，中文/韩语使用
   Piper Plus G2P。专用规则减少了明显错误，但仍不等于母语者产品验收。
2. 日语训练端已支持汉字，但移动 App 仍需接入能复现同一输出的原生 Open JTalk
   和词典；只复制 ONNX 不足以处理原始日语文本。当前输入使用 Open JTalk 基础音素，
   尚未把完整日语音高重音标签作为独立条件送入模型。
3. stock sherpa-onnx Piper/VITS 前端一次配置一个 eSpeak voice；一个 ONNX 在请求间
   切换多种语言需要 App/native 前端路由或 sherpa 适配。
4. 已有确定性验证集、best checkpoint、信号质检、可选 ASR 和 speaker similarity；
   尚没有能替代人工试听的自动 MOS。声纹阈值必须按自己的数据校准。
5. 蒸馏、结构化剪枝、选择性 INT8 和真机 benchmark 尚未自动化。它们是模型达到
   质量基线后的发布优化，不是第一次成功训练的前置条件。
6. 当前 VITS 实现是移动端工程基线，正式音质必须用真实数据和母语者试听验证。

不要因为训练 loss 下降就直接发布模型。至少需要每种启用语言的母语者试听，并检查
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

- 中文多音字和七语文本规范化
- 日语 Open JTalk 原生移动端适配与一致性测试
- Qwen 样本 ASR/声纹评分、多候选自动筛选
- 蒸馏和选择性 INT8
- Android/iOS Piper Plus/Open JTalk 前端路由与真机 benchmark 示例

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
