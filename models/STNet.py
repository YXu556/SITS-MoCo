"""
This script is re-implementation of Rußwurm et al. 2020
http://arxiv.org/abs/1905.11893
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class STNet(nn.Module):
    def __init__(self, input_dim=10, num_classes=9, d_model=128, n_head=16, n_layers=1, d_inner=128,
                 activation="relu", dropout=0.2, max_len=366, max_seq_len=70, T=1000, max_temporal_shift=30):
        super(STNet, self).__init__()
        self.modelname = self._get_name()
        self.max_seq_len = max_seq_len

        self.mlp_dim = [input_dim, 32, 64, d_model]
        layers = []
        for i in range(len(self.mlp_dim) - 1):
            layers.append(linlayer(self.mlp_dim[i], self.mlp_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        self.inlayernorm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(d_model, max_len=max_len + 2 * max_temporal_shift, T=T)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformerencoder = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        layers = []
        decoder = [d_model, 64, 32, num_classes]
        for i in range(len(decoder) - 1):
            layers.append(nn.Linear(decoder[i], decoder[i + 1]))
            if i < (len(decoder) - 2):
                layers.extend([
                    nn.BatchNorm1d(decoder[i + 1]),
                    nn.ReLU()
                ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, is_bert=False):
        x, mask, doy, weight = x

        x = x.permute((0, 2, 1))
        x = self.mlp1(x)
        x = x.permute((0, 2, 1))

        x = self.inlayernorm(x)
        x = self.dropout(x + self.position_enc(doy))

        x = x.transpose(0, 1)  # N x T x D -> T x N x D
        x = self.transformerencoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)  # T x N x D -> N x T x D

        # weight
        if not is_bert:
            weight = self.dropout(weight)
            weight /= weight.sum(1).unsqueeze(1)
            x = torch.bmm(weight.unsqueeze(1), x).squeeze()

        logits = self.decoder(x)

        return logits


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, T: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(T) / d_model))
        pe = torch.zeros(max_len + 1, d_model)
        pe[1:, 0::2] = torch.sin(position * div_term)
        pe[1:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, doy):
        """
        Args:
            doy: Tensor, shape [batch_size, seq_len]
        """
        return self.pe[doy]


class linlayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        out = input.permute((0, 2, 1))  # to channel last
        out = self.lin(out)

        out = out.permute((0, 2, 1))  # to channel first
        out = self.bn(out)
        out = F.relu(out)

        return out