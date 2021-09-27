import torch
from torch import nn
import torch.nn.functional as F

from network.mlp import MLP


class SubGraph(nn.Module):
    r"""
    Subgraph of VectorNet. This network accept a number of initiated vectors belong to
    the same polyline, flow three layers of network, then output this polyline's feature vector.
    """

    def __init__(self, v_len, layers_number):
        r"""
        Given all vectors of this polyline, we should build a 3-layers subgraph network,
        get the output which is the polyline's feature vector.

        Args:
            v_len: the length of vector.
            layers_number: the number of subgraph network.
        """
        super(SubGraph, self).__init__()
        self.layers = nn.Sequential()
        for i in range(layers_number):
            self.layers.add_module("sub{}".format(i), SubGraphLayer(v_len * (2 ** i)))
        self.v_len = v_len
        self.layers_number = layers_number

    def forward(self, x):
        r"""
        Args:
            x: a number of vectors. x.shape=[batch_size, v_number, v_len].
        Returns:
            The feature of this polyline. Shape is [batch_size, p_len], p_len = v_len * (2 ** layers_number)
        """
        assert len(x.shape) == 3
        batch_size = x.shape[0]
        x = self.layers(x) # [batch_size, v_number, p_len]
        x = x.permute(0, 2, 1)  # [batch size, p_len, v_number]
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        x = x.permute(0, 2, 1)  # [batch size, 1, p_len]
        x.squeeze_(1)
        assert x.shape == (batch_size, self.v_len * (2 ** self.layers_number))
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
        Args:
            len: the length of input vector.
        """
        super(SubGraphLayer, self).__init__()
        self.g_enc = MLP(len, len)

    def forward(self, x):
        r"""
        Args:
            x: A number of vectors. x.shape = [batch_size, n, len]
        
        Returns: 
            All processed vectors with shape [batch_size, n, len*2]
        """
        assert len(x.shape) == 3
        x = self.g_enc(x)
        batch_size, n, length = x.shape

        x2 = x.permute(0, 2, 1) # [batch_size, len, n]
        x2 = F.max_pool1d(x2, kernel_size=x2.shape[2])  # [batch_size, len, 1]
        x2 = torch.cat([x2] * n, dim=2)  # [batch_size, len, n]

        y = torch.cat((x2.permute(0, 2, 1), x), dim=2)
        assert y.shape == (batch_size, n, length*2)
        return y
