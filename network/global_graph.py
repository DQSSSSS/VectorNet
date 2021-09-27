import copy
import torch
from torch import nn
import torch.nn.functional as F

import config

class GlobalGraph(nn.Module):
    r"""
    Self-Attention module, corresponding the global graph.
    Given lots of polyline vectors, each length is 'C', we want to get the predicted feature vector.
    """

    def __init__(self, len, layers_number):
        r"""
        self.linear is 3 linear transformers for Q, K, V.

        Args:
            len: the length of input feature vector.
        """
        super(GlobalGraph, self).__init__()
        self.linears = [nn.Linear(len, len).to(config.device) for _ in range(3)]
        self.layers_number = layers_number

    def last_layer(self, P, index):
        batch_size, n, len = P.shape
        Q = self.linears[0](P) # [batch_size, n, len]
        K = self.linears[1](P)  
        V = self.linears[2](P)
        index = torch.stack([index] * n * len, dim=1).view(batch_size, n, len)
        Q = torch.gather(Q, 1, index)[:, 0:1, :] # [batch_size, 1, len]
        ans = torch.matmul(Q, K.permute(0, 2, 1))  # [batch_size, 1, len] x [batch_size, len, n] = [batch_size, 1, n]
        ans = F.softmax(ans, dim=2)
        ans = torch.matmul(ans, V)  # [batch_size, 1, len]
        ans.squeeze_(1)
        assert ans.shape == (batch_size, len)
        return ans

    def not_last_layer(self, P):
        assert False
        return P

    def forward(self, P, id):
        r"""
        Args:
            P: a list of polyline vectors, form a tensor.
                P.shape = [batch size, n, len]
            id: index of predicted agent.
                id.shape = [batch_size, 1]
        
        Returns: 
            The feature of predicted agent. [batch_size, len]
        """
        for i in range(self.layers_number - 1):
            P = self.not_last_layer(P)
        ans = self.last_layer(P, id)
        return ans
