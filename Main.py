import argparse
import numpy as np

import torch

from VectorNet import VectorNet
from VectorNet import VectorNetWithPredicting
from configure import device
from dataloader import load_train, load_test
import torch.nn.functional as F


def askADE(a, b):
    r"""
    Calculate Average Displacement Error(ADE).
    :param a:
    :param b:
    :return: ADE
    """
    # print('loss')
    # print(a)
    # print(b)
    ans = torch.zeros(1).to(device)
    minDis = torch.zeros(1).to(device)
    minDis[0] = float('inf')
    for it in range(a.shape[0]):
        tmp = 0
        for i in range(0, a.shape[1], 2):
            x1, y1 = a[it, i], a[it, i + 1]
            x2, y2 = b[it, i], b[it, i + 1]
            tmp += torch.sqrt(torch.square(x1 - x2).to(device)
                              + torch.square(y1 - y2).to(device))
        endDis = torch.sqrt(torch.square(a[it, -1] - b[it, -1]).to(device)
                            + torch.square(a[it, -2] - b[it, -2]).to(device))
        tmp /= a.shape[1] / 2
        if minDis > endDis:
            minDis = endDis
            ans = tmp
        print(endDis.item(), tmp)
        # ans += tmp
    return [minDis, ans]


# def askLoss(a, b):
#     # print('loss')
#     # print(a.shape)
#     # print(b)
#     ans = torch.zeros(1).to(device)
#     for it in range(a.shape[0]):
#         for i in range(0, a.shape[1], 2):
#             x1, y1 = a[it, i], a[it, i + 1]
#             x2, y2 = b[it, i], b[it, i + 1]
#             ans += torch.sqrt(torch.square(x1 - x2).to(device) + torch.square(y1 - y2).to(device)).to(device)
#     ans /= a.shape[0] * a.shape[1] / 2
#
#     return ans


def random_train(epoch, learningRate, batchSize):
    r"""
    A simple test, demonstrate my VectorNet is working.
    If you want to see convergence in training, please set the epoch lager and delete the code
    'learningRate *= 0.7'.
    :param epoch:
    :param learningRate:
    :param batchSize:
    :return:
    """
    timeStampNumber = 2
    len = 5
    train_X = torch.tensor(
        [[[1, 0, 0, 0, 0],
          [1, 2, 3, 0, 0],
          [0, 0, 1, 1, 1],
          [1, 1, 2, 2, 1]],

         [[0, 0, 0, 0, 0],
          [0, 0, 1, 1, 0],
          [1, 1, 2, 2, 0],
          [3, 3, 3, 1, 1]]]).float()
    train_Y = torch.tensor([[-1, -1, -2, -2],
                            [-1, -1, -2, -2]]).float()
    train = torch.utils.data.TensorDataset(train_X, train_Y)
    train_set = torch.utils.data.DataLoader(train, batch_size=batchSize)

    vectorNet = VectorNetWithPredicting(len=len, timeStampNumber=timeStampNumber)
    vectorNet = vectorNet.to(device)
    # print(vectorNet)
    # raise NotImplementedError
    lossfunc = torch.nn.MSELoss()

    for iterator in range(epoch):
        if iterator % 5 == 0:
            # learningRate *= 0.7
            optimizer = torch.optim.Adam(vectorNet.parameters(), lr=learningRate)
        for data, target in train_set:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            # print("in")
            outputs = vectorNet(data)
            # print("out")
            loss = lossfunc(outputs, target).to(device)
            print('outputs:',outputs)
            print('target:',target)
            # print(loss.device)
            loss.backward()
            optimizer.step()
            # print("???")
        print(iterator)



def train(epoch, learningRate, batchSize):
    r"""
    Training and saving my model.
    :param epoch:
    :param learningRate:
    :param batchSize:
    :return: None
    """

    train = load_train()
    trainset = torch.utils.data.DataLoader(train, batch_size=batchSize)
    test = load_test()
    testset = torch.utils.data.DataLoader(test, batch_size=batchSize)

    vectorNet = VectorNetWithPredicting(len=9, timeStampNumber=30)

    # for para in vectorNet.named_parameters():
    #     print(para)
    # exit(0)
    vectorNet = vectorNet.to(device)

    lossfunc = torch.nn.MSELoss()
    lr = learningRate
    optimizer = torch.optim.Adam(vectorNet.parameters(), lr=lr)
    for iterator in range(epoch):
        for data, target in trainset:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            offset = data[:, -1, :]  # [0, 0, 0, 0, 0, maxX, maxY, ..., 0]
            data = data[:, 0:data.shape[1] - 1, :]

            outputs = vectorNet(data) # [batch size, len*2]

            loss = lossfunc(outputs, target)
            loss.backward()
            # print(iterator)
            # print('loss=',loss.item())
            # t = askADE(outputs, target)
            # print('minDis=', t[0].item(), 'ADE=', t[1].item())
            optimizer.step()
        print(iterator)
        if (iterator + 1) % 5 == 0:
            lr *= 0.3
            optimizer = torch.optim.Adam(vectorNet.parameters(), lr=lr)

            minADE = torch.zeros(1).to(device)
            minDis = torch.zeros(1).to(device)
            minDis[0] = float('inf')
            for data, target in testset:
                data = data.to(device)
                target = target.to(device)

                offset = data[:, -1, :]  # [0, 0, 0, 0, 0, maxX, maxY, ..., 0]
                data = data[:, 0:data.shape[1] - 1, :]

                outputs = vectorNet(data)

                for i in range(0, outputs.shape[1], 2):
                    outputs[:, i] *= offset[:, 5]
                    target[:, i] *= offset[:, 5]
                    outputs[:, i + 1] *= offset[:, 6]
                    target[:, i + 1] *= offset[:, 6]

                print(outputs)
                print(target)
                # loss = lossfunc(outputs, target)
                # print('loss=', loss.item())
                t = askADE(outputs, target)
                # print('minDis=', t[0].item(), 'ADE=', t[1].item())

                # minADE += t[1]
                if minDis > t[0]:
                    minDis = t[0]
                    minADE = t[1]
            print('minDis=', minDis.item(), 'minADE=', minADE.item())

    torch.save(vectorNet, 'VectorNet-test.model')


def test(batchSize):
    r"""
    Test my model from file.
    :return: None
    """
    test = load_test()
    testset = torch.utils.data.DataLoader(test, batch_size=batchSize)

    vectorNet = torch.load('VectorNet-test.model')
    vectorNet = vectorNet.to(device)


    lossfunc = torch.nn.MSELoss()
    minADE = torch.zeros(1).to(device)
    minDis = torch.zeros(1).to(device)
    minDis[0] = -1
    for data, target in testset:
        data = data.to(device)
        target = target.to(device)

        offset = data[:, -1, :]  # [0, 0, 0, 0, 0, maxX, maxY, ..., 0]
        data = data[:, 0:data.shape[1] - 1, :]

        outputs = vectorNet(data)

        for i in range(0, outputs.shape[1], 2):
            outputs[:, i] *= offset[:, 5]
            outputs[:, i + 1] *= offset[:, 6]

        loss = lossfunc(outputs, target)
        # print("-----------")
        # print(outputs)
        # print(target)
        t = askADE(outputs, target)
        print('loss=', loss.item(), 'minDis=', t[0].item(), 'minADE=', t[1].item())

    #     if minDis.item() == -1 or minDis > t[0]:
    #         minDis = t[0]
    #         minADE = t[1]
    # print('minDis=', minDis.item(), 'minADE=', minADE.item())
    # ade += t
    # ade /= len(testXList)
    # print(ade)

def testCV(batchSize):
    test = load_test()
    testset = torch.utils.data.DataLoader(test, batch_size=batchSize)

    lossfunc = torch.nn.MSELoss()
    minADE = torch.zeros(1).to(device)
    minDis = torch.zeros(1).to(device)
    minDis[0] = float('inf')
    for data, target in testset:
        data = data.to(device) # [batch size, vNumber, len]
        target = target.to(device)

        # print(data.shape)
        # print(target.shape)

        outputs = torch.zeros_like(target).to(device)
        for i in range(data.shape[0]):
            vx, vy, t = 0, 0, 0
            lx, ly = 0, 0
            for j in range(data.shape[1]):
                if data[i, j, 4] == 0:
                    vx += data[i, j, 2] - data[i, j, 0]
                    vy += data[i, j, 3] - data[i, j, 1]
                    lx, ly = data[i, j, 2], data[i, j, 3]
                    t += 1
            vx = vx / t
            vy = vy / t
            for j in range(30):
                lx += vx
                ly += vy
                outputs[i, j*2] = lx
                outputs[i, j*2+1] = ly

        loss = lossfunc(outputs, target)
        # print("-----------")
        # print(outputs)
        # print(target)
        t = askADE(outputs, target)
        print('loss=', loss.item(), 'minDis=', t[0].item(), 'ADE=', t[1].item())

        if minDis > t[0]:
            minDis = t[0]
            minADE = t[1]
    print('minDis=', minDis.item(), 'minADE=', minADE.item())
    # ade += t
    # ade /= len(testXList)
    # print(ade)


if __name__ == '__main__':
    r"""
    Main function, include the parameters' building and a simple train.
    All default parameters are set to the values of paper.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", dest="batch_size", default=32, type=int)
    parser.add_argument("--epoch", dest="epoch", default=25, type=int)
    parser.add_argument("--learning_rate", dest="lr", default=0.001, type=float)
    args = parser.parse_args()

    # train(epoch=args.epoch, learningRate=args.lr, batchSize=args.batch_size)
    test(batchSize=args.batch_size)
    # testCV(batchSize=args.batch_size)

    # random_train(epoch=args.epoch, learningRate=args.lr, batchSize=args.batch_size)


"""
0.001:
loss= tensor(174.4507, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([18.6789], device='cuda:0', grad_fn=<DivBackward0>)
loss= tensor(200.6239, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([20.0312], device='cuda:0', grad_fn=<DivBackward0>)
loss= tensor(0.9591, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([1.3850], device='cuda:0', grad_fn=<DivBackward0>)

0.01:
loss= tensor(171.2146, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([18.5048], device='cuda:0', grad_fn=<DivBackward0>)
loss= tensor(204.9662, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([20.2468], device='cuda:0', grad_fn=<DivBackward0>)
loss= tensor(0.7863, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([1.2541], device='cuda:0', grad_fn=<DivBackward0>)
tensor([13.3352], device='cuda:0', grad_fn=<DivBackward0>)

"""
