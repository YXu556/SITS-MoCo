import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os


class LSTM(nn.Module):
    def __init__(self, input_dim=10, num_classes=9, hidden_dims=128, num_layers=4, bidirectional=True, dropout=0.2, ):
        super(LSTM, self).__init__()
        self.modelname = self._get_name()

        self.num_classes = num_classes

        self.d_model = num_layers * hidden_dims
        self.inlayernorm = nn.LayerNorm(input_dim)
        self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) * num_layers)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        hidden_dims = hidden_dims * 2

        self.decoder = nn.Linear(hidden_dims * num_layers, num_classes, bias=True)

    def forward(self, x, is_bert=False):
        x, *_ = x

        x = self.inlayernorm(x)

        self.lstm.flatten_parameters()
        outputs, (h, c) = self.lstm(x)  # outputs: (batchsize, t, n_hidden*2)

        if is_bert:
            h = outputs
        else:
            nlayers, batchsize, n_hidden = c.shape  # and h
            h = self.clayernorm(c.transpose(0, 1).contiguous().view(batchsize, nlayers * n_hidden))

        logits = self.decoder(h)

        return logits
