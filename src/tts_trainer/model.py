from __future__ import annotations

from .optional import require_training_dependencies


def build_model(vocab_size: int, n_mels: int, config, *, num_languages: int = 7):
    torch, _ = require_training_dependencies()
    nn = torch.nn

    class AcousticModel(nn.Module):
        """Compact language-conditioned text-to-Mel baseline.

        This is deliberately not labelled VITS: it has no MAS, flow, posterior
        encoder, adversarial decoder, or waveform output.
        """

        def __init__(self):
            super().__init__()
            h = config.hidden_size
            self.token_embedding = nn.Embedding(vocab_size, h, padding_idx=0)
            self.language_embedding = nn.Embedding(num_languages, config.language_embedding_size)
            self.language_projection = nn.Linear(config.language_embedding_size, h)
            layer = nn.TransformerEncoderLayer(
                d_model=h, nhead=config.encoder_heads, dim_feedforward=h * 4,
                dropout=config.dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, config.encoder_layers)
            self.mel_projection = nn.Linear(h, n_mels)

        def forward(self, tokens, language_ids, padding_mask=None):
            x = self.token_embedding(tokens)
            language = self.language_projection(self.language_embedding(language_ids)).unsqueeze(1)
            x = self.encoder(x + language, src_key_padding_mask=padding_mask)
            return self.mel_projection(x).transpose(1, 2)

    return AcousticModel()
