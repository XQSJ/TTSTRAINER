"""VITS 训练使用的判别器与先验损失函数。 / Loss functions used by VITS training."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def discriminator_loss(real_outputs, fake_outputs):
    """LSGAN 判别器损失：真音频趋向 1，假音频趋向 0。 / LSGAN discriminator loss."""
    loss = real_outputs[0][0].new_zeros(())
    for (real_score, _), (fake_score, _) in zip(real_outputs, fake_outputs):
        loss = loss + ((1.0 - real_score) ** 2).mean() + (fake_score ** 2).mean()
    return loss


def generator_adversarial_loss(fake_outputs):
    """生成器对抗损失：让假音频被判别为真。 / Generator adversarial loss."""
    return sum(((1.0 - score) ** 2).mean() for score, _ in fake_outputs)


def feature_matching_loss(real_outputs, fake_outputs):
    """逐层 L1 匹配判别器中间特征，稳定 GAN 训练。 / Layer-wise L1 on discriminator features."""
    loss = fake_outputs[0][0].new_zeros(())
    for (_, real_features), (_, fake_features) in zip(real_outputs, fake_outputs):
        for real, fake in zip(real_features, fake_features):
            # detach 真实特征：该损失只约束生成器方向 / Detach so only the generator is constrained
            loss = loss + F.l1_loss(fake, real.detach())
    return loss


def kl_loss(latent_prior, posterior_log_scale, prior_mean, prior_log_scale, mask):
    """后验与文本先验之间的逐帧 KL 散度。 / Frame-wise KL between posterior and text prior."""
    value = (
        prior_log_scale - posterior_log_scale - 0.5
        + 0.5 * (latent_prior - prior_mean).square() * torch.exp(-2.0 * prior_log_scale)
    )
    # 用 mask 求平均，clamp_min 防止空掩码除零 / Masked mean; clamp guards against empty masks
    return (value * mask).sum() / mask.sum().clamp_min(1.0)
