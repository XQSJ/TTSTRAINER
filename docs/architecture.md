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

## 训练图

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
模型配置、token、language map 和 speaker map。ONNX 只是部署产物，不能替代它。

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
