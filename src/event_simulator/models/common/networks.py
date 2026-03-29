from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUNextEventModel(nn.Module):
    def __init__(self, vocab_size: int, state_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim + 1, hidden_size=hidden_dim, batch_first=True)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.label_head = nn.Linear(hidden_dim, vocab_size)
        self.delta_head = nn.Linear(hidden_dim, 1)

    def forward(self, labels: torch.Tensor, deltas: torch.Tensor, lengths: torch.Tensor, state_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(labels)
        delta_feature = torch.log1p(deltas).unsqueeze(-1)
        packed = pack_padded_sequence(torch.cat([embedded, delta_feature], dim=-1), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden = hidden[-1]
        trunk = self.shared(torch.cat([hidden, state_features], dim=-1))
        return self.label_head(trunk), self.delta_head(trunk).squeeze(-1)


class GRUMultitaskTPPModel(nn.Module):
    def __init__(self, vocab_size: int, state_dim: int, hidden_dim: int, embedding_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim + 1, hidden_size=hidden_dim, batch_first=True)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.label_head = nn.Linear(hidden_dim, vocab_size)
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.condition_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, condition_dim),
        )

    def forward(
        self,
        labels: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,
        state_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(labels)
        delta_feature = torch.log1p(deltas).unsqueeze(-1)
        packed = pack_padded_sequence(torch.cat([embedded, delta_feature], dim=-1), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden = hidden[-1]
        trunk = self.shared(torch.cat([hidden, state_features], dim=-1))
        return self.label_head(trunk), self.delta_head(trunk).squeeze(-1), self.condition_head(trunk)


class ContinuousLSTMNextEventModel(nn.Module):
    def __init__(self, vocab_size: int, state_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_proj = nn.Linear(embedding_dim + 2, hidden_dim)
        self.cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.decay = nn.Sequential(nn.Linear(1, hidden_dim), nn.Sigmoid())
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.label_head = nn.Linear(hidden_dim, vocab_size)
        self.delta_head = nn.Linear(hidden_dim, 1)

    def forward(self, labels: torch.Tensor, deltas: torch.Tensor, lengths: torch.Tensor, state_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = labels.shape
        hidden_dim = self.label_head.in_features
        h = torch.zeros(batch_size, hidden_dim, device=labels.device)
        c = torch.zeros(batch_size, hidden_dim, device=labels.device)
        valid = torch.arange(seq_len, device=labels.device).unsqueeze(0) < lengths.unsqueeze(1)

        for step in range(seq_len):
            delta_step = deltas[:, step : step + 1]
            delta_log = torch.log1p(delta_step)
            embed = self.embedding(labels[:, step])
            h = h * self.decay(delta_log)
            projected = self.input_proj(torch.cat([embed, delta_log, valid[:, step : step + 1].float()], dim=-1))
            new_h, new_c = self.cell(projected, (h, c))
            mask = valid[:, step].unsqueeze(-1)
            h = torch.where(mask, new_h, h)
            c = torch.where(mask, new_c, c)

        trunk = self.shared(torch.cat([h, state_features], dim=-1))
        return self.label_head(trunk), self.delta_head(trunk).squeeze(-1)


class CausalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attn_out, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        return x + self.ff(self.norm2(x))


class AttentionTPPModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        state_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        max_len: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len + 1, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.input_proj = nn.Linear(embedding_dim + hidden_dim // 2, hidden_dim)
        self.predict_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.decoder_blocks = nn.ModuleList(
            [CausalAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.state_context = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2 + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.label_head = nn.Linear(hidden_dim, vocab_size)
        self.delta_head = nn.Linear(hidden_dim, 1)

    def compute_trunk(self, labels: torch.Tensor, deltas: torch.Tensor, lengths: torch.Tensor, state_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = labels.shape
        position_ids = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, seq_len)
        valid_mask = position_ids < lengths.unsqueeze(1)
        embed = self.embedding(labels)
        delta_log = torch.log1p(deltas)
        cumulative_time = torch.cumsum(deltas, dim=1)
        seq_scale = cumulative_time.gather(1, (lengths - 1).clamp(min=0).unsqueeze(1)).clamp(min=1.0)
        recency = (cumulative_time / seq_scale).clamp(0.0, 1.0)
        remaining = (1.0 - recency).clamp(0.0, 1.0)
        time_features = torch.stack([delta_log, recency, remaining, valid_mask.float()], dim=-1)
        h = self.input_proj(torch.cat([embed, self.time_mlp(time_features)], dim=-1))
        h = h + self.position_embedding(position_ids.clamp(max=self.max_len - 1))

        predict_token = self.predict_token.expand(batch_size, -1, -1) + self.state_context(state_features).unsqueeze(1)
        predict_token = predict_token + self.position_embedding(
            torch.full((batch_size, 1), seq_len, device=labels.device, dtype=torch.long).clamp(max=self.max_len)
        )

        encoded = torch.cat([h, predict_token], dim=1)
        decoder_valid = torch.cat([valid_mask, torch.ones(batch_size, 1, device=labels.device, dtype=torch.bool)], dim=1)
        causal_mask = torch.triu(torch.ones(seq_len + 1, seq_len + 1, device=labels.device, dtype=torch.bool), diagonal=1)
        for block in self.decoder_blocks:
            encoded = block(encoded, attn_mask=causal_mask, key_padding_mask=~decoder_valid)
        encoded = self.final_norm(encoded)

        history_encoded = encoded[:, :seq_len, :]
        predict_encoded = encoded[:, -1, :]
        attn_scores = torch.matmul(predict_encoded.unsqueeze(1), history_encoded.transpose(1, 2)).squeeze(1) / math.sqrt(self.hidden_dim)
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        pooled = torch.bmm(torch.softmax(attn_scores, dim=-1).unsqueeze(1), history_encoded).squeeze(1)
        return self.shared(torch.cat([predict_encoded, pooled, state_features], dim=-1))

    def forward(self, labels: torch.Tensor, deltas: torch.Tensor, lengths: torch.Tensor, state_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.compute_trunk(labels, deltas, lengths, state_features)
        return self.label_head(trunk), self.delta_head(trunk).squeeze(-1)
