import argparse
import os
import pandas as pd

import numpy as np
from configure import device

import torch

from VectorNet import VectorNet
from VectorNet import VectorNetWithPredicting
from dataloader import load_train, load_test
import torch.nn.functional as F

from generateH5 import generate_forecasting_h5


def load_data(DATA_PATH, nameList):
    X = []
    Y = []
    polyline_ID = 8
    type_ID = 4
    maxSize = []
    offset = []
    for name in nameList:
        ans = pd.read_csv(DATA_PATH + name, header=None)
        ans = np.array(ans)
        offset.append(ans[-1, :])
        x, tx, y = [], [], []
        j = 0

        maxX, maxY = 0, 0
        for i in range(ans.shape[0] - 1):
            if ans[i, type_ID] == 0:
                maxX = np.max([maxX, np.abs(ans[i, 0]), np.abs(ans[i, 2])])
                maxY = np.max([maxY, np.abs(ans[i, 1]), np.abs(ans[i, 3])])

        dx, dy = 1, 1
        for i in range(ans.shape[0] - 1):
            if i + 1 == ans.shape[0] - 1 or \
                    ans[i, polyline_ID] != ans[i + 1, polyline_ID]:
                id = int(ans[i, polyline_ID])
                while len(maxSize) <= id:
                    maxSize.append(0)
                if ans[i, type_ID] == 0:  # predicted agent
                    t = np.zeros_like(ans[0]).astype('float')
                    t[0] = ans[i, polyline_ID]
                    x.append(t)
                    assert i - j + 1 == 19
                    if ans[j, 0] > 0:
                        dx = -1
                    if ans[j, 1] > 0:
                        dy = -1
                    maxSize[id] = np.max([maxSize[id], 19])
                    for l in range(0, 19):
                        tx.append(ans[j])
                        j += 1
                else:
                    maxSize[id] = np.max([maxSize[id], i - j + 1])
                    while j <= i:
                        tx.append(ans[j])
                        j += 1
        for xx in tx:
            xx[0] *= dx
            xx[2] *= dx
            xx[1] *= dy
            xx[3] *= dy
            xx[0] /= maxX
            xx[2] /= maxX
            xx[1] /= maxY
            xx[3] /= maxY
            x.append(xx)
        offset[-1][3] = dx
        offset[-1][4] = dy
        offset[-1][5] = maxX
        offset[-1][6] = maxY
        x = np.array(x).astype('float')
        y = np.array(y).astype('float')
        X.append(x)
        Y.append(y)
    XX = []
    YY = Y
    maxSize = np.array(maxSize)
    for it in range(len(X)):
        x = []
        x.append(X[it][0])
        j = 1
        for i in range(0, maxSize.shape[0]):
            if maxSize[i] == 0:
                break
            tmp = maxSize[i]
            lst = np.zeros(9)
            lst[polyline_ID] = i
            while j < X[it].shape[0] and \
                    X[it][j, polyline_ID] == i:
                x.append(X[it][j])
                lst = X[it][j]
                j += 1
                tmp -= 1
            while tmp > 0:
                x.append(lst)
                tmp -= 1
        XX.append(x)
    for i in range(len(offset)):
        XX[i].append(offset[i])
    XX = np.array(XX).astype('float')
    YY = np.array(YY).astype('float')
    XX = torch.from_numpy(XX)
    YY = torch.from_numpy(YY)
    XX = XX.float()
    YY = YY.float()
    train = torch.utils.data.TensorDataset(XX, YY)
    return train


vectorNet = torch.load('VectorNet-test.model')
vectorNet = vectorNet.to(device)


def getTraj(data, offset):
    tmp = vectorNet(data)  # [batch size, len*2]
    outputs = torch.zeros(tmp.shape[0], tmp.shape[1] // 2, 2).to(device)
    maxX, maxY = offset[:, 5], offset[:, 6]
    for i in range(0, tmp.shape[1], 2):
        outputs[:, i // 2, 0] = tmp[:, i] * maxX * offset[:, 3] + offset[:, 0]
        outputs[:, i // 2, 1] = tmp[:, i + 1] * maxY * offset[:, 4] + offset[:, 1]
    return outputs


if __name__ == '__main__':
    # root_dir = '/mnt/e/paper/VectorNet/test_obs/data/'
    root_dir = 'E:/paper/VectorNet/test_obs/data-final/'

    BATCH_SIZE = 32
    TEST_FILE = []
    for root, dirs, files in os.walk(root_dir):
        TEST_FILE = files

    # tmp = []
    # for i in range(10):
    #     tmp.append(TEST_FILE[i])
    # TEST_FILE = tmp

    output_all = {}
    sum = 0
    for it in range(0, len(TEST_FILE), BATCH_SIZE):
        maxJ = np.min([it + BATCH_SIZE, len(TEST_FILE)])
        test = load_data(root_dir, TEST_FILE[it:maxJ])
        test_set = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

        for data, target in test_set:
            data = data.to(device)

            offset = data[:, -1, :]  # [offset_X, offset_Y, id, 0, ..., 0]
            data = data[:, 0:data.shape[1] - 1, :]

            outputs = getTraj(data, offset)  # [batch size, len, 2]
            for i in range(data.shape[0]):
                seq = int(offset[i, 2])
                output_all[seq] = outputs[i, :, :].unsqueeze(0). \
                    to(torch.device("cpu")).detach().numpy()
                # print(seq)
                # print(output_all[seq])
                # print('------------')

        sum += 1
        if sum % 50 == 0:
            print(sum)

    output_path = 'E:/paper/VectorNet/test_obs/'
    generate_forecasting_h5(output_all, output_path)
