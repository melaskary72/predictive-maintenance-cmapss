"""PyTorch model definitions for predictive maintenance.

All models share a dual-head architecture: a backbone produces a context vector,
which is fed to a regression head (RUL prediction) and a classification head
(failure-within-30-cycles prediction).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class DualHead(nn.Module):
    """Shared dual-head module: regression + classification on the same feature."""

    def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(hidden, 1)
        self.clf_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.reg_head(h).squeeze(-1), self.clf_head(h).squeeze(-1)


class LSTMRegressor(nn.Module):
    """Stacked LSTM with dual-head output.

    Optionally accepts an auxiliary tabular feature vector that is concatenated
    to the LSTM's final hidden state before the dual head, used for ablation.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        aux_features: int = 0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.aux_features = aux_features
        head_in = hidden + aux_features
        self.head = DualHead(head_in, hidden=hidden, dropout=dropout)

    def forward(
        self, x: torch.Tensor, aux: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, (h, _) = self.lstm(x)
        last = h[-1]
        if self.aux_features > 0 and aux is not None:
            last = torch.cat([last, aux], dim=-1)
        return self.head(last)


class DilatedConv1DRegressor(nn.Module):
    """1D CNN with stacked dilated convolutions (TCN-style).

    Each block: dilated Conv1d -> GELU -> Dropout. Dilation doubles each block,
    giving an exponentially growing receptive field while keeping parameters small.
    """

    def __init__(
        self,
        n_features: int,
        n_channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        aux_features: int = 0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for i in range(n_blocks):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(
                nn.Conv1d(in_ch, n_channels, kernel_size=kernel_size,
                          dilation=dilation, padding=padding)
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_ch = n_channels
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.aux_features = aux_features
        head_in = n_channels + aux_features
        self.head = DualHead(head_in, hidden=n_channels, dropout=dropout)

    def forward(
        self, x: torch.Tensor, aux: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        if self.aux_features > 0 and aux is not None:
            h = torch.cat([h, aux], dim=-1)
        return self.head(h)


class TransformerRegressor(nn.Module):
    """Transformer encoder with learned positional embedding and dual-head output.

    Uses standard nn.TransformerEncoder with batch_first=True. The window's
    feature dimension is projected to d_model, learned positional embeddings
    are added, the encoder produces contextualized token embeddings, and
    we mean-pool across the sequence dimension before the dual head.

    Optionally accepts auxiliary tabular features for the ablation experiment.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 30,
        aux_features: int = 0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.aux_features = aux_features
        head_in = d_model + aux_features
        self.head = DualHead(head_in, hidden=d_model, dropout=dropout)

    def forward(
        self, x: torch.Tensor, aux: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        seq_len = h.size(1)
        h = h + self.pos_embedding[:, :seq_len, :]
        h = self.encoder(h)
        h = h.mean(dim=1)
        if self.aux_features > 0 and aux is not None:
            h = torch.cat([h, aux], dim=-1)
        return self.head(h)
