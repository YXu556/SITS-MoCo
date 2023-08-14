"""
This script is re-implementation of Pelletier et al. 2019
https://www.mdpi.com/2072-4292/11/5/523
"""

import os

import torch
import torch.nn as nn
import torch.utils.data


class TempCNN(nn.Module):
    def __init__(self, input_dim=10, num_classes=9, hidden_dims=128, kernel_size=7, dropout=0.2, max_seq_len=70, ):
        super(TempCNN, self).__init__()
        self.modelname = self._get_name()

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)

        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims * max_seq_len, 4 * hidden_dims, drop_probability=dropout)
        self.decoder = nn.Linear(4 * hidden_dims, num_classes)

    def forward(self, x, is_bert=False):
        x, *_ = x
        # require NxTxD
        x = x.transpose(1,2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        if is_bert:
            x = x.permute((0, 2, 1))
        else:
            x = self.flatten(x)
            x = self.dense(x)
        x = self.decoder(x)
        return x


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)
