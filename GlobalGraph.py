import copy

import torch
from torch import nn
import torch.nn.functional as F

from configure import device

r"""
Generate N same modules.
"""


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention(nn.Module):
    r"""
    Self-Attention module, corresponding the global graph.
    Given lots of polyline vectors, each length is 'C', we want to get the predicted feature vector.
    """

    def __init__(self, C):
        r"""
        self.linear is 3 linear transformers for Q, K, V.
        :param C: the length of input feature vector.
        """
        super(Attention, self).__init__()
        self.linear = clones(nn.Linear(C, C), 3)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, P, id):
        r"""

        :param P: a list of polyline vectors, form a tensor.
                P.shape = [batch size, n, C]
        :param id: index of predicted vector.
                id.shape = [batch size]
        :return: output.
        """

        batchSize, n, C = P.shape

        Q = torch.zeros(0, C).to(device)

        Qt = self.linear[0](P)  # [batch size, n, C]
        for i in range(id.shape[0]):
            x = id[i].item()
            q = Qt[i, x].unsqueeze(0)  # [1, C]
            Q = torch.cat((Q, q), dim=0)
        Q.unsqueeze_(1)  # Q's shape is # [batch size, 1, C]

        # Q = self.linear[0](P) #!!!
        K = self.linear[1](P)  # [batch size, n, C]
        V = self.linear[2](P)

        ans = torch.matmul(Q, K.permute(0, 2, 1))  # [batch size, 1, n]
        ans = F.softmax(ans, dim=2)
        ans = torch.matmul(ans, V)  # [batch size, 1, C]

        ans.squeeze_(1)

        return ans
