import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    r"""
      Construct a MLP, include a single fully-connected layer (64 units),
    followed by layer normalization and then ReLU.
    """

    def __init__(self, inputSize, outputSize, hiddenSize = 64, noReLU = False):
        r"""
        self.norm is layer normalization.
        :param inputSize: the size of input layer.
        :param outputSize: the size of output layer.
        :param noReLU: indicates weather to pass through ReLU, because the predict network don't need.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, outputSize)
        # self.norm = LayerNorm(outputSize)
        self.norm1 = torch.nn.LayerNorm(hiddenSize)
        self.norm2 = torch.nn.LayerNorm(outputSize)
        self.noReLU = noReLU

    def forward(self, x):
        r"""

        :param x: x.shape = [batch, inputSize]
        :return:
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        if self.noReLU:
            return x
        x = F.relu(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        r"""
        Layer normalization implemented by myself. 'feature' is the length of input.
        :param features: length of input.
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        r"""

        :param x: x.shape = [batch, feature]
        :return:
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2