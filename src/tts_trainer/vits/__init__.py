"""VITS 子包：波形 TTS 的模型、训练与导出。

模块地图 /
- model.py         MultilingualVITS 模型结构。
- modules.py       网络组件（编码器/解码器/流等积木）。
- trainer.py       训练循环（训练入口 train_vits）。
- exporter.py      ONNX 导出（导出入口 export_vits_onnx，Piper 形状输入）。
- composable.py    可组合包导出。
- data.py          训练数据采样。
- losses.py        GAN 损失函数。
- discriminators.py 判别器。
- validation.py    训练期验证。
- runtime.py       导出后 ONNX 冒烟运行。
- config.py        VitsConfig 配置。

训练入口 train_vits（trainer.py）→ 导出入口 export_vits_onnx（exporter.py）。

English: VITS subpackage — model, GAN training, and Piper-shaped ONNX
export. Training entry: train_vits (trainer.py); export entry:
export_vits_onnx (exporter.py).
"""

from .config import VitsConfig, load_vits_config
from .model import MultilingualVITS, VitsTrainingOutput
from .discriminators import VitsDiscriminator

__all__ = ["VitsConfig", "load_vits_config", "MultilingualVITS", "VitsTrainingOutput", "VitsDiscriminator"]
