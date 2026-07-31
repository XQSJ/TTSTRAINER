# 训练与部署架构

## 条件设计

模型始终保留独立条件：

```text
phoneme_ids ───────────────┐
language_id -> embedding ──┼-> MultilingualVITS -> waveform
speaker_id  -> embedding ──┘
```

`speaker_id` 不能代替 `language_id`。即便第一版只有一个声音，也保留 Speaker
Embedding 参数和 `speaker_map.json`，保证 checkpoint 格式以后不变。需要注意：
单说话人数据不能教会模型完整的说话人空间；增加音色时仍需多说话人数据微调。

`training_configs/*.json` 中的 `experiment.languages` 决定 language embedding
数量和 ID 顺序。例如 `["en", "fr"]` 会得到 `en=0, fr=1`，而不会
为未选中的语言保留空 embedding。

语言能力来自 `configs/system/language_registry.json`。每项声明 Teacher 映射、
G2P provider/voice 和烟雾测试文本。Qwen 官方十语已经内置；使用自备音频时，
外层配置可以追加 `teacher=null` 的语言。训练开始前先检查注册、Teacher 和实际
G2P 输出，失败时不会创建错误的训练 token。

## 训练图

可选的文本生成阶段发生在音频生成之前：

```text
builtin templates / source CSV / OpenAI-compatible text model
        -> normalize / script check / deduplicate / optional G2P
        -> datasets/text_corpora/<config-fingerprint>/texts.csv + report
        -> Qwen Teacher 生成 WAV
```

文本生成只负责构造输入语句，不参与 VITS 梯度训练。固定 `seed` 的 builtin
provider 是确定性的；LLM provider 的输出必须经过同一过滤器并保留来源字段。
公共文本 corpus 由语言和完整生成配置指纹寻址，与模型名称无关；重复训练直接
复用。Qwen 权重、前端词典和质检权重也属于项目公共资源，只有音色 WAV、训练状态
和导出模型按具体任务隔离。

训练前先固定文本前端：

```text
原始文本 -> NFKC/空白规范化 -> language frontend router
        -> eSpeak IPA / Open JTalk Japanese phones / Piper Plus Mandarin-Korean IPA
        -> 统一 token vocabulary -> metadata.phonemes.csv
        -> frontend.lock.json
```

`frontend.lock.json` 逐语言记录 provider、引擎版本、voice 或词典、规范化规则和
token 规则。训练会校验它与配置、已有 checkpoint 是否兼容，避免同一个 token ID
在继续训练后表示不同读音。G2P 本身不参与 VITS 训练；日语默认调用 Open JTalk，
中文和韩语调用 Piper Plus G2P，其余内置语言调用 eSpeak-ng。

训练 checkpoint 还保存从冻结 metadata 中抽取的前端 conformance case。
导出时写入 `frontend.conformance.json`，使 Python 部署机和移动端 native
前端都能核对「原文 → 音素 → token ID」，而不只比较版本字符串。

自动流水线的阶段、逐语言预检、样本生成批次、epoch/step/loss 和 checkpoint
保存都通过标准 Python logging 输出。`training.log_every_steps` 只控制高频 loss
日志，不会关闭阶段与错误日志。

训练前的信号质检会阻止削波、静音、时长和语速异常样本进入训练。数据随后按
`language + speaker` 分层、固定随机种子划分 train/validation，并保存 CSV 与
指纹。每轮验证计算 Mel/Duration/KL 等确定性指标，`best/` 与 `last/` 分开保存。
ASR 回识别和 speaker embedding 相似度属于可选重型门禁，不会默认下载模型。

训练态包含 Text Encoder、Duration Predictor、Posterior Encoder、MAS、Flow、
Waveform Decoder、Multi-Period/Scale Discriminator。损失包括 Mel、KL、Duration、
Adversarial 和 Feature Matching。

Posterior Encoder + Decoder 的重建是基础主链。`aligned-text-prior.wav` 是隔离
Text Encoder/Flow/MAS 的验证信号，不作为额外 Mel 损失反向更新共享 Decoder；
文本先验使用标准 VITS 的 KL 和 Duration 目标训练。

验证目录同时保存 posterior 均值重建与随机重建。历史文件名
`posterior-reconstruction.wav` 始终指稳定均值版，随机训练路径单独保存为
`posterior-sampled-reconstruction.wav`；二者需要结合判断。

自动指标不能直接“听懂”语音。`best` 默认只按 posterior Mel 保存为主链诊断，
`prior_mel` 作为文字先验诊断；自动流水线默认导出 `last`。产品验收仍需结合
`text-only-*`、ASR 回识别和人工试听，不能把单个 Mel 最小值等同于最好听的模型。

## 推理图

部署仅执行：

```text
Text Encoder -> Duration Predictor -> sample prior
-> inverse Flow -> Waveform Decoder
```

Posterior Encoder、MAS 和所有判别器不会进入最终 ONNX。

## 实际产物与必须保留的中间态

```text
runs/<model-name>/
├── splits/              # 固定 train/validation 与指纹
├── quality/             # 自动质检报告
└── checkpoints/
    ├── best/            # 发布候选
    ├── last/            # 断点恢复/扩音色
    └── step-*/          # 周期历史快照

artifacts/<model-name>/
└── model.onnx           # 部署产物，不可继续训练
```

每个训练 checkpoint 包含 generator、discriminator、两个 optimizer、训练步数、
模型配置、token、language map、speaker map 和文本前端契约。ONNX 只是部署产物，
不能替代它。

## 发布压缩顺序（可选，不阻塞第一版）

1. 先训练质量基线并永久保留完整 checkpoint。
2. 导出 FP32 ONNX，先在目标手机建立质量、内存和 RTF 基线。
3. 如果不达标，优先直接定义更小的 Student，以 Mel/Duration 蒸馏。
4. 仍需缩小时再做结构化剪枝并继续微调。
5. 最后选择性 INT8；音频 Decoder 的敏感层允许保留 FP16/FP32。

普通 ONNX Runtime 不会因为非结构化权重变成零而自动加速，所以不把非结构化
剪枝作为默认方案。

## Piper/sherpa 部署输入

导出模型保持四个标准输入：

```text
input, input_lengths, scales, sid -> output
```

为了不牺牲内部的语言/说话人解耦，`sid` 是部署 profile：

```text
sid = speaker_id * num_languages + language_id
```

导出目录的 `model.onnx.json` 会列出所有 profile，App 不需要自行计算。每次
增加内置 speaker 后需要从保留的 PyTorch checkpoint 继续训练并重新导出 ONNX。
同目录的 `frontend.json` 告诉 App 每种语言应调用哪个 provider 及 voice/词典；
移动端必须得到与训练 metadata 相同的 token unit 后，再调用同一个 ONNX。
`frontend.conformance.json` 用于在发布前证明这一映射确实一致。

`preset=mobile` 使用独立的 v3 token contract。训练核心只学习：

```text
BOS, (phoneme)*, EOS
```

stock sherpa 仍通过 Piper wire 传入 `BOS,(phoneme,PAD)*,EOS`；修复版还可能在
BOS 后包含一个 PAD。ONNX 导出 wrapper 的 `strip-piper-pads-v1` 会删除所有 wire
PAD，再交给 VITS 核心。这样 MAS 不必给大量重复传输 blank 强制分配音频帧，同时
兼容 sherpa 1.13.4 与修正后的句首 PAD 语义。`preset=quality` 不启用该适配器，
仍保留逐语言专用 G2P。
