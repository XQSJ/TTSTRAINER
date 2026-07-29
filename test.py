from vits.runtime import OnnxTTS, write_wav

tts = OnnxTTS("artifacts/model_1")

samples = tts.synthesize_text(
    "你好，欢迎使用多语言语音系统。",
    language="zh",
    speaker="voice_01",
)

write_wav(
    "output.wav",
    samples,
    tts.sample_rate,
)