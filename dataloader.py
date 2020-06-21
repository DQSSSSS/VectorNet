import os

import pandas as pd
import numpy as np
import torch

# TEST_DATA_PATH = 'data/data/'
# TRAIN_DATA_PATH = 'data/data/'
# TRAIN_FILE = ['2645.csv','3700.csv']
# TEST_FILE = ['3828.csv','3861.csv','4791.csv']


TEST_DATA_PATH = 'data/test-data/'
TRAIN_DATA_PATH = 'data/train-data/'

for root, dirs, files in os.walk(TEST_DATA_PATH):
    TEST_FILE = files

for root, dirs, files in os.walk(TRAIN_DATA_PATH):
    TRAIN_FILE = files

r"""
data structure:
  [start_X, start_Y, end_X, end_Y, type, att1, att2, att3, polyline_id]
  
  type: 0 for predicted agent, 1 for other agents, 2 for lane.
  att:
    For agents:
      'att1' and 'att2' are time stamp of begin and end points.
      'att3' is speed. 
    For lanes:
      'att1' represents whether it has traffic control.
      'att2' represents it's direction. 0 is NONE, 1 and 2 are LEFT and RIGHT respectively.
      'att3' represents whether it is a intersection.
    
"""


def load_data(DATA_PATH, nameList):
    r"""
    Loading data from files.
    :param nameList: the files for generating.
    :return: X and Y represents input and label.
    """

    X = []
    Y = []
    polyline_ID = 8
    type_ID = 4
    maxSize = np.zeros(300)
    offset = []
    for name in nameList:
        ans = pd.read_csv(DATA_PATH + name, header=None)
        ans = np.array(ans)
        x, tx, y = [], [], []
        j = 0

        maxX, maxY = 0, 0
        for i in range(ans.shape[0]):
            if ans[i, type_ID] == 0:
                maxX = np.max([maxX, np.abs(ans[i, 0]), np.abs(ans[i, 2])])
                maxY = np.max([maxY, np.abs(ans[i, 1]), np.abs(ans[i, 3])])

        for i in range(ans.shape[0]):
            if ans[i, type_ID] != 2:
                ans[i, 5] = ans[i, 6] = ans[i, 7] = 0

        dx, dy = 1, 1
        for i in range(ans.shape[0]):
            if i + 1 == ans.shape[0] or \
                    ans[i, polyline_ID] != ans[i + 1, polyline_ID]:
                id = int(ans[i, polyline_ID])
                # if ans[i, type_ID] == 2:
                #     j = i + 1
                #     continue
                if ans[i, type_ID] == 0:  # predicted agent
                    t = np.zeros_like(ans[0]).astype('float')
                    t[0] = ans[i, polyline_ID]
                    x.append(t)
                    # print(i)

                    # if i-j+1 != 49:
                    #     print(DATA_PATH + 'data_' + name)

                    assert i - j + 1 == 49
                    maxSize[id] = np.max([maxSize[id], 19])
                    if ans[j, 0] > 0:
                        dx = -1
                    if ans[j, 1] > 0:
                        dy = -1

                    for l in range(0, 19):
                        tx.append(ans[j])
                        j += 1
                    for l in range(19, 49):
                        y.append(ans[j, 2])
                        y.append(ans[j, 3])
                        j += 1
                else:
                    maxSize[id] = np.max([maxSize[id], i - j + 1])
                    while j <= i:
                        tx.append(ans[j])
                        j += 1
        print(dx, dy, name)

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
        for i in range(0, len(y), 2):
            y[i] *= dx
            y[i + 1] *= dy
            y[i] /= maxX
            y[i + 1] /= maxY

        offset.append([0, 0, 0, 0, 0, maxX, maxY, 0, 0])
        x = np.array(x).astype('float')
        y = np.array(y).astype('float')

        # print(x.shape)

        X.append(x)
        Y.append(y)

    ans = 0
    for i in range(0, maxSize.shape[0]):
        ans += maxSize[i]

    # print(ans)
    XX = []
    YY = Y
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

    # print(XX)

    # print(XX.shape)
    # print(YY.shape)
    # for i in range(XX.shape[1]):
    #     print(XX[0,i,polyline_ID],XX[1,i,polyline_ID])
    # exit(0)

    XX = torch.from_numpy(XX)
    YY = torch.from_numpy(YY)

    XX = XX.float()
    YY = YY.float()

    train = torch.utils.data.TensorDataset(XX, YY)
    return train


def load_train():
    r"""
    Loading train set.
    :return: train set.
    """
    return load_data(TRAIN_DATA_PATH, TRAIN_FILE)


def load_test():
    r"""
    Loading test set.
    :return: test set.
    """
    return load_data(TEST_DATA_PATH, TEST_FILE)

# if __name__ == '__main__':
#     load_train()
# np_arr = np.array([[1], [2], [3], [4]])
# tor_arr = torch.from_numpy(np_arr)
# print(type(np_arr))
