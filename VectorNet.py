import copy

import torch
from torch import nn

import MLP
from GlobalGraph import Attention, clones
from SubGraph import SubGraph
from configure import device
import torch.nn.functional as F


class VectorNet(nn.Module):

    r"""
    Vector network.
    """

    # def __init__(self, len, pNumber):
    def __init__(self, len):
        r"""
        Construct a VectorNet.
        :param len: length of each vector v ([ds,de,a,j]).
        """
        super(VectorNet, self).__init__()
        layersNumber = 3
        # self.subGraphs = clones(SubGraph(layersNumber=3, len=len), 3)
        self.subGraphs = SubGraph(layersNumber=layersNumber, len=len)
        # self.pLen = len
        self.pLen = len * (2 ** layersNumber)
        self.globalGraph = Attention(C=self.pLen)

    def forward(self, data):
        r"""

        :param data: the input data of network. Each coordinate of key position is centered by
              predicted agent, and the first input feature vector is like [id,0,0,...,0], 'id'
              means the index of predicted agent, and other vectors are sorted by corresponding
              polyline index.

              For each batch, it looks like:
                [[id,0,...,0],
                 [a11,a12,...,a1k,a1_id],
                 [a21,a22,...,a2k,a2_id],
                 ...
                 [an1,an2,...,ank,an_id]]
              satisfied a(i)_id <= a(i+1)_id

              shape: data.shape = [batch size, vNumber, len]
        :return: output
        """
        data = data.permute(1, 0, 2)  # [vNumber, batch size, len]
        id = data[0, :, 0].long()
        pID = data[:, 0, -1].long()
        data[:, :, -1] = 0

        batchSize, len = data.shape[1], data.shape[2]
        P = torch.zeros(batchSize, 0, self.pLen).to(device)

        j = 1
        for i in range(1, data.shape[0]):
            if i + 1 == data.shape[0] or \
                    pID[i] != pID[i + 1]:
                tmp = torch.zeros(batchSize, 0, len).to(device)
                while j <= i:
                    t = data[j]  # [batch size, len]
                    t.unsqueeze_(1)  # [batch size, 1, len]
                    tmp = torch.cat((tmp, t), dim=1)
                    j += 1

                # tmp's shape is [batch size, pvNumber, Len]
                # subGraphId = int(data[i, 0, len - 1].item())
                # print(tmp.shape)
                p = self.subGraphs(tmp)  # [batch size, pLen]
                p.unsqueeze_(1)  # [batch size, 1, pLen]
                P = torch.cat((P, p), dim=1)
                # print('2 VectorNet',i,j, 'subGraphId =',subGraphId)

        # P's shape is [batch size, pNumber, pLen]
        # P = F.normalize(P, dim=2)
        feature = self.globalGraph(P, id)  # [batch size, pLen]
        # print(feature.device)
        # print(feature.shape)
        # raise NotImplementedError
        return feature

class VectorNetWithPredicting(nn.Module):

    r"""
      A class for packaging the VectorNet and future trajectory prediction module.
      The future trajectory prediction module uses MLP without ReLu(because we
    hope the coordinate of trajectory can be negative).
    """

    def __init__(self, len, timeStampNumber):
        r"""
        Construct a VectorNet with predicting.
        :param len: same as VectorNet.
        :param timeStampNumber: the length of time stamp for predicting the future trajectory.
        """
        super(VectorNetWithPredicting, self).__init__()
        self.vectorNet = VectorNet(len=len)
        self.trajDecoder = MLP.MLP(inputSize=self.vectorNet.pLen,
                                   outputSize=timeStampNumber * 2,
                                   noReLU=False)


    def forward(self, x):
        r"""

        :param x: the same as VectorNet.
        :return: Future trajectory vector with length timeStampNumber*2, the form is (x1,y1,x2,y2,...).
        """
        x = self.vectorNet(x)
        x = self.trajDecoder(x)
        return x