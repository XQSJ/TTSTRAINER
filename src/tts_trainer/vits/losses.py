from __future__ import annotations

import torch
from torch.nn import functional as F


def discriminator_loss(real_outputs, fake_outputs):
    loss = real_outputs[0][0].new_zeros(())
    for (real_score, _), (fake_score, _) in zip(real_outputs, fake_outputs):
        loss = loss + ((1.0 - real_score) ** 2).mean() + (fake_score ** 2).mean()
    return loss


def generator_adversarial_loss(fake_outputs):
    return sum(((1.0 - score) ** 2).mean() for score, _ in fake_outputs)


def feature_matching_loss(real_outputs, fake_outputs):
    loss = fake_outputs[0][0].new_zeros(())
    for (_, real_features), (_, fake_features) in zip(real_outputs, fake_outputs):
        for real, fake in zip(real_features, fake_features):
            loss = loss + F.l1_loss(fake, real.detach())
    return loss


def kl_loss(latent_prior, posterior_log_scale, prior_mean, prior_log_scale, mask):
    value = (
        prior_log_scale - posterior_log_scale - 0.5
        + 0.5 * (latent_prior - prior_mean).square() * torch.exp(-2.0 * prior_log_scale)
    )
    return (value * mask).sum() / mask.sum().clamp_min(1.0)
