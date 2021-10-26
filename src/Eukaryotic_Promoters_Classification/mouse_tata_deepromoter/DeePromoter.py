import random
import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils


class ParallelCNN(nn.Module):
    def __init__(self, para_ker, pool_kernel=6, drop=0.5):
        """
        Multiple CNN layer apply on input and concatenate the output
        :param para_ker: List of kernel size that will be used
        :param pool_kernel: Pooling parameter after CNN
        :param drop: Dropout parameter
        """
        super(ParallelCNN, self).__init__()
        self.lseq = nn.ModuleList()
        for k in para_ker:
            seq = nn.Sequential(
                nn.Conv1d(4, 4, kernel_size=k, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(pool_kernel),
                nn.Dropout(drop)
            )
            self.lseq.append(seq)

    def forward(self, inputs):
        """
        :param inputs: DNA onehot sequences [batch_size x 4 x length]
        :return: Stack CNN output feature from different kernel size [batch_size x 12 x length]
        """
        _x = list()
        for seq in self.lseq:
            x = seq(inputs)
            _x.append(x)
        # concate outputs of every conv layer to a tensor
        _x = torch.cat(_x, 1)
        return _x


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs):
        """
        :param inputs: visual feature [batch_size x T x input_size]
        :return: contextual feature [batch_size x T x output_size]
        """

        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(inputs)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class DeePromoter(nn.Module):
    def __init__(self, para_ker, input_shape=(64, 300, 4), pool_kernel=6, drop=0.5):
        """
        Deepromoter
        :param para_ker: List of kernel size that will be used
        :param input_shape: Specifies the input shape for model(fixed)
        :param pool_kernel: Pooling parameter after CNN
        :param drop: Dropout parameter
        """
        super(DeePromoter, self).__init__()
        binode = len(para_ker) * 4

        self.pconv = ParallelCNN(para_ker, pool_kernel, drop)
        self.bilstm = BidirectionalLSTM(binode, binode, binode)
        self.flatten = nn.Flatten()
        x = torch.zeros(input_shape)
        shape = self.get_feature_shape(x)

        self.fc = nn.Sequential(
            nn.Linear(shape, shape),
            nn.ReLU(),
            nn.Linear(shape, 2),
        )

    def get_feature_shape(self, x):
        """Pass a dummy input through to find the shape
        after flatten layer for Linear layer construction"""
        x = x.permute(0, 2, 1)
        x = self.pconv(x)
        x = x.permute(0, 2, 1)
        x = self.bilstm(x)
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pconv(x)
        x = x.permute(0, 2, 1)
        x = self.bilstm(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
