# gru_encoder.py
# GRU-based temporal state encoder
# Drop-in replacement for TransformerEncoder
# Used for ablation comparison (V6: GRU+SAC)

import torch
import torch.nn as nn
import numpy as np
from config import (
    SEQUENCE_LENGTH,
    TRANSFORMER_DIM,
    DROPOUT_RATE
)


class GRUEncoder(nn.Module):
    """
    GRU Encoder for temporal state encoding.

    Drop-in replacement for TransformerEncoder.
    Same input/output interface:
      Input:  (batch, seq_len, state_dim)
      Output: (batch, TRANSFORMER_DIM)

    Used for ablation study (V6: GRU+SAC)
    to compare against Transformer encoder.

    Architecture:
      Linear projection: state_dim -> TRANSFORMER_DIM
      2-layer bidirectional GRU
      Output projection: 2*TRANSFORMER_DIM -> TRANSFORMER_DIM

    GRU vs LSTM: fewer parameters (no cell state),
    often trains faster, comparable performance.
    """
    def __init__(self, state_dim):
        super().__init__()

        self.state_dim  = state_dim
        self.hidden_dim = TRANSFORMER_DIM
        self.num_layers = 2

        # Project input to GRU dimension
        self.input_projection = nn.Linear(
            state_dim,
            self.hidden_dim)

        # Bidirectional GRU
        # 2 layers, same depth as Transformer
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=DROPOUT_RATE,
            bidirectional=True)

        # Project bidirectional output
        # back to TRANSFORMER_DIM
        self.output_projection = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 2,
                self.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(
            self.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, state_sequence):
        """
        Process sequence of states through GRU.

        state_sequence shape:
          (batch, seq_len, state_dim)
          OR (seq_len, state_dim)

        Returns enriched state vector
        of shape (TRANSFORMER_DIM,)
        """
        # Add batch dim if needed
        if state_sequence.dim() == 2:
            state_sequence = state_sequence.unsqueeze(0)

        # NaN protection on input
        if torch.isnan(state_sequence).any():
            state_sequence = torch.nan_to_num(
                state_sequence, nan=0.0)

        # Project input to hidden dim
        x = self.input_projection(state_sequence)
        x = self.dropout(x)

        # NaN check after projection
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        # GRU forward pass
        # gru_out: (batch, seq_len, 2*hidden_dim)
        # hidden: final hidden states
        try:
            gru_out, hidden = self.gru(x)

            # NaN check after GRU
            if torch.isnan(gru_out).any():
                gru_out = torch.nan_to_num(
                    gru_out, nan=0.0)

            # Use last timestep output
            last_output = gru_out[:, -1, :]

        except Exception:
            # Fallback: use projected input
            last_output = torch.cat(
                [x[:, -1, :], x[:, -1, :]], dim=-1)

        # Project to TRANSFORMER_DIM
        output = self.output_projection(last_output)
        output = self.layer_norm(output)

        # Final NaN protection
        output = torch.nan_to_num(
            output,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0)

        return output.squeeze(0)

    def get_output_dim(self):
        return self.hidden_dim