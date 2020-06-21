import torch
from torch import nn
import torch.nn.functional as F

import MLP
from configure import device

r"""
  Construct subgraph for a polyline.
"""


class SubGraph(nn.Module):
    r"""
      Subgraph of VectorNet. This network accept a number of initiated vectors belong to
    the same polyline, flow three layers of network, then output this polyline's feature vector.
    """

    def __init__(self, len, layersNumber):
        r"""
          Given all vectors of this polyline, we should build a 3-layers subgraph network,
        get the output which is the polyline's feature vector.
        :param len: the length of vector.
        :param layersNumber: the number of subgraph network.
        """
        super(SubGraph, self).__init__()

        self.layers = nn.ModuleList([SubGraphLayer(len),
                                     SubGraphLayer(len * (2 ** 1)),
                                     SubGraphLayer(len * (2 ** 2))])
        # self.layers = nn.ModuleList([SubGraphLayer(len),
        #                              SubGraphLayer(len),
        #                              SubGraphLayer(len)])

    def forward(self, x):
        r"""

        :param x: a number of vectors. x.shape=[batch size, vNumber, len].
        :return: The vector of this polyline. Shape is [batch size, output len].
        """

        # x = torch.tensor(
        #     [[[1, 0, 0, 0, 0, 0, 0, 0, 0],
        #       [1, 2, 3, 1, -1, -2, -3, -1, 1],
        #       [3, 2, 1, 2, 3, 1, -1, -2, -3]],
        #
        #      [[0, 0, 0, 0, 2, 1, 2, 3, 1],
        #       [1, 3, 2, 1, 3, 1, -1, -2, -3],
        #       [3, 3, 3, 2, 0, 0, 0, 2, 1]]]).float().to(device)

        for layer in self.layers:
            # print('sub graph !!!')
            x = layer(x)
        # x's shape is [batch size, vNumber, output len]

        # print(x)
        #
        # import numpy as np
        # y = torch.zeros(x.shape[0], x.shape[2])
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         for k in range(x.shape[2]):
        #             y[i, k] = np.max([y[i, k], x[i, j, k]])

        x = x.permute(0, 2, 1)  # [batch size, output len, vNumber]
        x = F.max_pool1d(x, kernel_size=x.shape[2])  # [batch size, output len, 1]
        x = x.permute(0, 2, 1)  # [batch size, 1, output len]
        x.squeeze_(1)
        # print(x)
        #
        # for i in range(y.shape[0]):
        #     for j in range(y.shape[1]):
        #         print(y[i, j], x[i, j], y[i, j] == x[i, j])

        # exit(0)
        return x


class SubGraphLayer(nn.Module):
    r"""
      One layer of subgraph, include the MLP of g_enc.
      The calculation detail in this paper's 3.2 section.
      Input some vectors with 'len' length, the output's length is '2*len'(because of
    concat operator).
    """

    def __init__(self, len):
        r"""

        :param len: the length of input vector.
        """
        super(SubGraphLayer, self).__init__()
        self.g_enc = MLP.MLP(len, len)
        # self.decD = MLP.MLP(2*len, len)

    def forward(self, x):
        r"""

        :param x: A number of vectors. x.shape = [batch size, vNumber, len]
        :return: All processed vectors with shape [batch size, vNumber, len*2]
        """

        # print(x)
        x = self.g_enc(x)
        batchSize, vNumber, len = x.shape

        x = x.permute(1, 0, 2)  # [vNumber, batch size, len]

        mp = x.permute(1, 2, 0)
        mp = F.max_pool1d(mp, kernel_size=mp.shape[2])  # [batch size, len, 1]
        # print(x.shape)
        # print(mp.shape)
        mp = torch.cat([mp] * vNumber, dim=2)  # [batch size, len, vNumber]
        # print(mp.shape)
        y = torch.cat((mp.permute(0, 2, 1), x.permute(1, 0, 2)), dim=2)

        # y = torch.zeros(batchSize, 0, len * 2).to(device)
        # for i in range(x.shape[0]):
        #     L, tmp, R = torch.split(x, [i, 1, x.shape[0] - i - 1], dim=0)
        #     L = L.to(device)
        #     R = R.to(device)
        #
        #     # print('sub graph layer 1', i, L.shape, tmp.shape, R.shape)
        #     # tmp's shape is [1, batch size, len]
        #
        #     t = torch.cat((L, R), dim=0)  # [vNumber-1, batch size, len]
        #
        #     if t.shape[0] == 0:
        #         t = torch.zeros(batchSize, len, 1).to(device)
        #     else:
        #         t = t.permute(1, 2, 0)  # [batch size, len, vNumber-1]
        #         t = F.max_pool1d(t, kernel_size=t.shape[2])  # [batch size, len, 1]   agg
        #     tmp.squeeze_(0)
        #     t.squeeze_(2)
        #     tmp = torch.cat((tmp, t), dim = 1) # [batch size, len * 2]    rel
        #     tmp = tmp.unsqueeze(1)
        #     y = torch.cat((y, tmp), dim = 1)

        # y's shape is [batch size, vNumber, len * 2]
        # print('y\'s shape',y.shape)

        # y = self.decD(y)
        return y
