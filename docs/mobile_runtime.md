# 移动端运行策略

## 当前单模型接口

`export-vits` 生成：

```text
model.onnx
model.onnx.json
frontend.json
frontend.conformance.json
tokens.json
tokens.txt
```

ONNX 输入形状与 Piper 一致：

```text
input:         int64 [B, T]
input_lengths: int64 [B]
scales:        float [3]
sid:           int64 [B]
output:        float [B, 1, N]
```

`sid = speaker_id * num_languages + language_id`。`num_languages` 由该模型的
`experiment.languages` 决定。Python 参考实现位于
`tts_trainer.vits.runtime.OnnxTTS`。

## 为什么还不能直接称为 stock sherpa 多语模型

sherpa-onnx 的 Piper/VITS 前端从 ONNX metadata 读取一个 eSpeak `voice`，然后
把原始文本转为 token。我们的模型在一次加载中需要根据每条请求切换多个
language profile；stock VITS 配置没有对应的逐请求 language 参数。因此：

- ONNX 推理图本身可在移动端 ONNX Runtime 运行；
- App 必须先按 language 生成与训练相同的 phoneme unit；或者
- 给 sherpa-onnx VITS 前端增加 `language/profile` 路由；或者
- 导出多个固定语言入口，但这样会重复模型权重。

项目采用“一个 ONNX 核心 + App/native 前端路由”作为默认方向。完成 native
适配前，不在模型中写入错误的单语言 Piper metadata。

## 前端一致性

`frontend.json` 是训练和移动端之间的硬契约，包含：

```text
provider = language-router
normalization contract
token contract
language -> provider + engine version + eSpeak voice/Open JTalk/Piper Plus resource
```

App 应加载该文件，按请求语言选择前端，再将结果映射到 `tokens.json`。开发期
若对应前端版本不一致，参考运行时默认拒绝合成；这可以避免升级前端后出现
难以定位的读音漂移。真机发布时应固定并记录 eSpeak-ng、Open JTalk 和词典版本。

`frontend.conformance.json` 是可执行的一致性样例，保存了每种语言的原文、
训练时音素和最终 token ID。在训练机或部署机上可先执行：

```bash
PYTHONPATH=src .venv/bin/python -m tts_trainer verify-frontend \
  --model-dir artifacts/model_1
```

移动端集成测试也应遍历这些 case，并逐个断言音素和 token ID 一致。
这能及时发现 native 分词、词典、规范化或版本差异，而不是等到合成出错音再排查。

Piper 的词表前四项固定为：

```text
_ 0
^ 1
$ 2
  3
```

eSpeak IPA 目前按 UTF-8 codepoint 映射；Open JTalk 与 Piper Plus 保留 `ch`、
`tɕʰ`、`tone4` 等完整音素单元。训练 metadata 使用 `<space>` 序列化真实空格 token。

Latin 语言可直接使用 eSpeak。日语默认使用 Open JTalk/MeCab；中文使用 Piper Plus
pinyin→IPA 与声调规则；韩语使用 g2pk2/MeCab 音韵规则再转 IPA。缺少专用依赖时
预处理器直接失败，不会退回 eSpeak。母语者评测仍然是发布前的必要步骤。

新增语言后必须把导出的 `frontend.json`、`frontend.conformance.json`、
`tokens.json` 和 voice profile 与 ONNX 一起更新。配置中 Teacher/G2P 预检通过，
只证明训练端可生成一致 token；移动端
仍必须包含相同 provider 版本以及对应 voice/词典数据。

eSpeak-ng 使用 GPL-3.0-or-later。TTSTRAINER 不复制或打包它；移动 App 如果分发
eSpeak-ng native 库，需要单独完成许可证和发布方式评估。

训练端的 pyopenjtalk 是 Open JTalk 的 Python wrapper，不直接部署到手机。Android/
iOS 需要集成兼容的 Open JTalk/MeCab 原生前端及相同词典，或者由业务层传入已经
生成好的 phoneme IDs。无论采用哪种方式，声学生成仍只调用一个 `model.onnx`。

中文/韩语可优先采用 Piper Plus 提供的 Android/Kotlin 或 iOS/Swift G2P 实现，
但不能只相信“同一项目”就认为输出相同；必须运行随模型导出的
`frontend.conformance.json`，逐条验证 token ID 后才能发布。
