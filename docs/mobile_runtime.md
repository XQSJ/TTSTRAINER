# 移动端运行策略

## 当前单模型接口

`export-vits` 生成：

```text
model.onnx
model.onnx.json
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
- App 必须先按 language 生成与训练相同的 codepoint token；或者
- 给 sherpa-onnx VITS 前端增加 `language/profile` 路由；或者
- 导出多个固定语言入口，但这样会重复模型权重。

项目采用“一个 ONNX 核心 + App/native 前端路由”作为默认方向。完成 native
适配前，不在模型中写入错误的单语言 Piper metadata。

## 前端一致性

Piper 的词表前四项固定为：

```text
_ 0
^ 1
$ 2
  3
```

其余 IPA 符号按 UTF-8 codepoint 映射。训练 metadata 使用 `<space>` 序列化
真实空格 token，读取后恢复为 codepoint ` `。

Latin 语言可直接使用 eSpeak。中日韩需要逐语言评测；尤其是 eSpeak 日语处理
汉字时可能切换到英文拼读，预处理器默认检测并拒绝这种 fallback。
