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

训练前先固定文本前端：

```text
原始文本 -> NFKC/空白规范化 -> language 对应的 eSpeak voice
        -> IPA UTF-8 codepoint -> metadata.phonemes.csv
        -> frontend.lock.json
```

`frontend.lock.json` 记录 provider、引擎版本、逐语言 voice、规范化规则和 token
规则。训练会校验它与配置、已有 checkpoint 是否兼容，避免同一个 token ID 在
继续训练后表示不同读音。G2P 本身不参与 VITS 训练；默认调用现有 eSpeak-ng。

自动流水线的阶段、逐语言预检、样本生成批次、epoch/step/loss 和 checkpoint
保存都通过标准 Python logging 输出。`training.log_every_steps` 只控制高频 loss
日志，不会关闭阶段与错误日志。

训练态包含 Text Encoder、Duration Predictor、Posterior Encoder、MAS、Flow、
Waveform Decoder、Multi-Period/Scale Discriminator。损失包括 Mel、KL、Duration、
Adversarial 和 Feature Matching。

## 推理图

部署仅执行：

```text
Text Encoder -> Duration Predictor -> sample prior
-> inverse Flow -> Waveform Decoder
```

Posterior Encoder、MAS 和所有判别器不会进入最终 ONNX。

## 必须保留的训练阶段

```text
runs/<model-name>/
├── baseline-full/       # 未压缩 FP32，可继续训练
├── distilled/           # teacher -> student
├── pruned-finetuned/    # 结构调整后恢复训练
├── qat/                 # 可选量化感知训练
└── release/
    ├── model-fp32.onnx
    └── model-int8.onnx
```

每个训练 checkpoint 包含 generator、discriminator、两个 optimizer、训练步数、
模型配置、token、language map、speaker map 和文本前端契约。ONNX 只是部署产物，
不能替代它。

## 压缩顺序

1. 先训练质量基线并永久保留完整 checkpoint。
2. 直接定义更小的 Student，以 Mel/Duration/中间特征蒸馏。
3. 如果仍需缩小，再做结构化剪枝并继续微调。
4. 导出 FP32 ONNX，建立质量和真机速度基线。
5. 选择性 INT8；音频 Decoder 的敏感层允许保留 FP16/FP32。

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
同目录的 `frontend.json` 告诉 App 每种语言应调用哪个 eSpeak voice；移动端必须
得到与训练 metadata 相同的 UTF-8 codepoint token 后，再调用 ONNX。
