import copy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import os

import json
import math


sigma = 0.2
for i in range(0, 10):
    print("%.100f" % (1 - math.erf(1/sigma)))
    sigma -= 0.01
# a = torch.tensor([[[2, 2, 3, 4],
#                    [1, 3, 4, 5],
#                    [0, 1, 2, 6]],
#
#                   [[2, 3, 4, 1],
#                    [1, 1, 1, 1],
#                    [0, 1, 1, 0]]]).float()
#
# # a = torch.tensor([[[0, 1, 2, 3]],
# #
# #                   [[0, 1, 1, 0]]]).float()
#
# b = torch.tensor([[[1, 2, 3],
#                    [1, 2, 3],
#                    [1, 2, 3],
#                    [4, 4, 4]],
#
#                   [[2, 1, 1],
#                    [2, 1, 1],
#                    [2, 1, 1],
#                    [1, 1, 2]]]).float()
#
# # a = F.max_pool1d(a.permute(0, 2, 1), kernel_size=a.shape[1]).permute(0, 2, 1)
# a = torch.cat([a] * 5, dim=2)  # [batch size, len, vNumber]
# print(a)

# ans = torch.matmul(a, b).float()
# print(ans)
# ans = F.softmax(ans, dim=2)
# print(ans)
# print(torch.sum(ans, dim=2))

# tensor([[[0.0024, 0.0473, 0.9503]],
#
#         [[0.7870, 0.1065, 0.1065]]])

# a = torch.tensor([1-1e-8])
#
# print(a.long())

# a = [1,2,3,4]
#
# print(a[1:2])

# TRAIN_FILE = ['100015.csv', '101451.csv', '102423.csv', '103369.csv', '10503.csv', '105292.csv', '105327.csv', '105529.csv', '108080.csv', '108557.csv', '109076.csv', '109874.csv', '110673.csv', '111415.csv', '111870.csv', '113003.csv', '113137.csv', '113555.csv', '115988.csv', '116257.csv', '117092.csv', '117295.csv', '117323.csv', '11800.csv', '118563.csv', '123099.csv', '126725.csv', '127736.csv', '129602.csv', '130908.csv', '134225.csv', '136495.csv', '136839.csv', '139049.csv', '141651.csv', '146491.csv', '148506.csv', '148845.csv', '150891.csv', '155122.csv', '155444.csv', '155622.csv', '156762.csv', '157834.csv', '159493.csv', '159522.csv', '161459.csv', '163620.csv', '172532.csv', '173988.csv', '174662.csv', '176035.csv', '176664.csv', '178553.csv', '178929.csv', '179455.csv', '17993.csv', '18101.csv', '181246.csv', '187368.csv', '189291.csv', '190905.csv', '195493.csv', '19796.csv', '198714.csv', '201698.csv', '202667.csv', '2045.csv', '204886.csv', '205516.csv', '210073.csv', '21666.csv', '24051.csv', '2790.csv', '28682.csv', '31341.csv', '34674.csv', '35153.csv', '35546.csv', '37821.csv', '37921.csv', '39695.csv', '40043.csv', '40196.csv', '47123.csv', '497.csv', '50981.csv', '51052.csv', '51259.csv', '55503.csv', '56324.csv', '57386.csv', '6033.csv', '61312.csv', '62470.csv', '63764.csv', '64668.csv', '65255.csv', '66304.csv', '66892.csv', '68845.csv', '69218.csv', '73846.csv', '74312.csv', '77260.csv', '79390.csv', '79659.csv', '79815.csv', '79914.csv', '80340.csv', '82154.csv', '83076.csv', '84282.csv', '85064.csv', '85095.csv', '85876.csv', '86632.csv', '86719.csv', '87741.csv', '88506.csv', '9124.csv', '93097.csv', '93200.csv', '93686.csv', '95097.csv', '96589.csv', '96640.csv', '9787.csv']
# TEST_FILE = ['107054.csv', '115319.csv', '119103.csv', '121396.csv', '122.csv', '12206.csv', '123449.csv', '129.csv', '131.csv', '135.csv', '138.csv', '140500.csv', '142.csv', '151320.csv', '152867.csv', '154708.csv', '156140.csv', '158448.csv', '159.csv', '167.csv', '169.csv', '174125.csv', '184.csv', '187.csv', '196.csv', '198139.csv', '199.csv', '208.csv', '209274.csv', '210261.csv', '210323.csv', '211.csv', '211084.csv', '21449.csv', '218.csv', '219.csv', '32545.csv', '51339.csv', '64962.csv', '67031.csv', '70752.csv', '77236.csv', '79230.csv', '80000.csv', '82035.csv', '90171.csv', '96033.csv']
#
# print("TRAIN_FILE = ")
# for root, dirs, files in os.walk('data/data-full/data-train'):
#     f = files
# f.
# print(f)

# print("TEST_FILE = ")
# for root, dirs, files in os.walk('data/data-full/data-test'):
#     print(files)

# print("TEST_FILE = ")
# for root, dirs, files in os.walk('data/data-full/val-data'):
#     print(files)
#
# for i in range(len(TRAIN_FILE)):
#     for j in range(len(TEST_FILE)):
#         if TRAIN_FILE[i] == TEST_FILE[j]:
#             print("!!!")

# print(len(test))
# print(len(train))
#
# print(np.max([1,2]))


# import pandas as pd
#
# data = pd.read_pickle('data/feature/forecasting_features_train.pkl')
# ans = pd.read_pickle('data/feature/train.pkl')
#
# print(data['FEATURES'].values)
# print(ans)


# data = torch.tensor(
#     [[[1, 0, 0, 0],
#       [1, 2, 3, 1],
#       [3, 2, 1, 2]],
#
#      [[0, 0, 0, 0],
#       [1, 3, 2, 1],
#       [3, 3, 3, 2]]]).float()
# data = torch.tensor(data,requires_grad=True)
#
# print(data.shape)
# loss = torch.max(data,dim=2)
# loss = torch.sum(loss)
# print(loss)
# torch.nn.NLLLoss
# loss2 = F.max_pool1d(data, kernel_size=data.shape[2])
# loss2 = data * 2
# loss2 = torch.sum(loss2)
# loss2.backward()
# print(loss2.grad)


#
# print(data.shape)
# print((data*data).sum(axis=2).shape)
# print((data*data).sum(axis=2))
#
# # a = data / (data*data).sum(axis=2)
# # print(a)
#
# print(F.normalize(data,dim=2))
# print((F.normalize(data,dim=2) * F.normalize(data,dim=2)).sum(axis=2))


# a = "is"
# b = "i"
# b += 's'
# print(a)
# print(b)
# print(a == b)
# print(a is b)

# a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
# b = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
#
# print(a.shape)
# print(np.stack(a).shape)

# data = torch.tensor(
#     [[[1, 0, 0, 0],
#       [1, 2, 3, 1],
#       [3, 2, 1, 2]],
#
#      [[0, 0, 0, 0],
#       [1, 3, 2, 1],
#       [3, 3, 3, 2]]]).float()
#
# x = data
#
# x = x.permute(0,2,1)
# x = F.max_pool1d(x, kernel_size=x.shape[2])
# x = x.permute(0,2,1)
# x.squeeze_()
# # print(x)
#
#
# a = torch.tensor(
#     [[1,2,3,4],
#      [1,1,1,1]]).float()
#
# print(a.mean(-1,keepdim=True))


# a = torch.zeros(3,0,3)
# print(a)
# b = torch.randn(3,2,3)
# print(b)
#
# c = torch.cat((a,b),dim=1)
# print(c)

# class Net(nn.Module):
#
#     def __init__(self, num_classes=10):
#         super(Net, self).__init__()
#         self.linear = nn.Linear(3,3)
#
#     def forward(self, x):
#         print(x.x)
#         x = self.linear(x.x)
#         return x
#
#
# class MyData:
#     def __init__(self,list):
#         self.x = torch.Tensor(list)
#
#
#
# input = MyData([1,2,3])
# net = Net()
# out = net(input)
# print(input)
# print(out)

# batch = 2
# pNum = 3
# N = 4
# linears = nn.ModuleList([copy.deepcopy(nn.Linear(N, N)) for _ in range(N)])
#
# query = key = value = torch.rand(batch, pNum, N)
#
# query, key, value = \
#     [l(x) for l, x in zip(linears, (query, key, value))]
#
# for l, x in zip(linears, (query, key, value)):
#     print(l)
#     print(x)
#     print(l(x[0]))
#     print(l(x[0][0]))
#     print("---")
#
# print(query)
# print(key)
# print(value)

# ansList = [torch.rand(3).unsqueeze(0) for i in range(3)]
# ans = ansList[0]
# for i in range(1, len(ansList)):
#     ans = torch.cat((ans, ansList[i]), 0)
# print(ansList)
# print(ans)
