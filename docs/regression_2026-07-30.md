# 2026-07-30 训练回归说明 / Training Regression Note

## 结论 / Conclusion

2026-07-30 17:08 的 `2a850e1` 是可观察到“模型从效果一般变成无法说话”的版本
断点。问题不是 Qwen 数据突然失效，而是训练目标和 mobile token 序列同时发生了
高风险变化。

Commit `2a850e1` at 2026-07-30 17:08 is the observable boundary where models
changed from imperfect-but-speaking to unusable. The Qwen dataset did not
suddenly fail; two high-risk training changes overlapped.

## 原因一：错误的 prior Mel 反传 / Cause 1: harmful prior-Mel backprop

当天 14:10 的 `03ed2fa` 加入：

```text
generator_loss += 10 × aligned_prior_mel
```

当时没有延迟或 warm-up。训练从第一个 step 就把尚未学习好的 text prior 送入逆向
Flow 和共享 Decoder，并用目标音频直接反向传播。Decoder 同时服务 posterior
重建和文本推理，因此这条梯度会破坏本来能逐步学会人声的 posterior 主链。

Commit `03ed2fa` added the loss above without a delay or warm-up. From the first
step, an untrained text prior was decoded through the inverse flow and shared
decoder. Because that decoder is also used by posterior reconstruction, the
auxiliary gradient damaged the main reconstruction path.

同初始化、同数据的 400-step 消融结果：

| 训练方式 / Training mode | posterior Mel |
|---|---:|
| 标准 VITS，无辅助 prior Mel / Standard VITS | 0.763 |
| `aligned_prior_mel_weight=10` | 1.126 |

当前版本已完全删除这条训练损失，并拒绝仍包含相关参数的旧配置。

The current version completely removes this training loss and rejects old
configs that still contain its options.

## 原因二：mobile 训练序列膨胀 / Cause 2: inflated mobile token sequence

`2a850e1` 把 mobile 训练输入改为：

```text
BOS, PAD, phoneme_1, PAD, phoneme_2, PAD, ..., EOS
```

这会接近翻倍文本 token 数量，同时增加 MAS 和 Duration 学习难度。PAD 是
sherpa/Piper 的传输协议细节，不应成为当前轻量 VITS 必须学习的语言内容。

Commit `2a850e1` nearly doubled the text sequence with transport PAD tokens,
making MAS and duration learning harder. Those PADs belong to the
sherpa/Piper wire protocol and should not be learned as linguistic content by
the compact VITS core.

当前 mobile v3 训练使用：

```text
BOS, phoneme_1, phoneme_2, ..., EOS
```

sherpa-onnx 1.13.4 仍可发送旧 Piper wire 序列，但导出 ONNX 会先删除 PAD，再把
紧凑序列交给模型。

Mobile v3 now trains on the compact sequence. The exported ONNX accepts the
legacy sherpa-onnx 1.13.4 wire sequence, strips transport PADs, and only then
invokes the VITS core.

## 当前保护措施 / Current safeguards

- `aligned_prior_mel_*` 配置会立即报错，不再静默生效。
- `decode_aligned_prior()` 处于 `no_grad`，只用于验证。
- quality 与 mobile 都使用标准 VITS 训练损失。
- mobile 使用 compact training tokens，部署层单独适配 Piper wire tokens。
- ONNX 导出使用真实 BOS→音素→EOS 输入，并比较 PyTorch/ONNX 数值。
- checkpoint 保存训练目标和音频参数；导出检查采样率及语言/音色维度。
- `best` 只作为 posterior 重建诊断，流水线默认导出 `last`。

- Old `aligned_prior_mel_*` options fail immediately.
- `decode_aligned_prior()` is `no_grad` and validation-only.
- Both quality and mobile use the standard VITS objective.
- Mobile learns compact tokens; Piper wire adaptation stays in deployment.
- ONNX export uses real BOS→phoneme→EOS input and checks PyTorch/ONNX parity.
- Checkpoints record the objective and audio settings; export checks dimensions.
- `best` diagnoses posterior reconstruction; the pipeline exports `last`.

## 旧模型处理 / Handling affected models

在 `03ed2fa` 到本次修复之间从零训练的模型，不应继续作为质量基线。最可靠的处理是
复用公共文本和 WAV 数据，用新的 `experiment.name` 从零训练。旧 checkpoint 可以
用于研究对比，但不应直接发布。

Models trained from scratch between `03ed2fa` and this fix should not remain
quality baselines. Reuse the shared text/WAV cache, choose a new
`experiment.name`, and retrain from scratch. Keep affected checkpoints only for
diagnostic comparison, not release.
