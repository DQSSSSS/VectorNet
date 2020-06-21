import torch

from configure import device
from dataloader import load_test, load_train
import matplotlib.pyplot as plt

vectorNet = torch.load('VectorNet-test.model').to(device)

def viz(x, y):

    data = x.permute(1, 0, 2)  # [vNumber, batch size, len]
    id = data[0, :, 0].long()
    print(id.item())

    batchSize, len = data.shape[1], data.shape[2]
    j = 1
    for i in range(1, data.shape[0]):
        if i == data.shape[0] - 1 or \
                data[i, 0, len - 1] != data[i + 1, 0, len - 1]:
            listX = []
            listY = []
            listX.append(data[j, 0, 0])
            listY.append(data[j, 0, 1])
            while j <= i:
                listX.append(data[j, 0, 2])
                listY.append(data[j, 0, 3])
                j += 1
            if data[i, 0, -1] == id.item():
                plt.plot(listX, listY, 'r', linewidth=3) # agent
                print(listX)
                print(listY)
                print('---------')
            elif data[i, 0, 4] == 1:
                plt.plot(listX, listY, 'black', linewidth=3) # others
            else:
                plt.plot(listX, listY, 'b') # lane

    listX = [0]
    listY = [0]
    for i in range(0, y.shape[1], 2):
        listX.append(y[0, i])
        listY.append(y[0, i + 1])
    print(listX)
    print(listY)
    print('---------')
    plt.plot(listX, listY, 'g', linewidth=3) # ground truth

    x = x.to(device)
    myPredict = vectorNet(x)
    listX = [0]
    listY = [0]
    for i in range(0, y.shape[1], 2):
        listX.append(myPredict[0, i])
        listY.append(myPredict[0, i + 1])
    print(listX)
    print(listY)
    print('---------')
    plt.plot(listX, listY, 'yellow', linewidth=3) # predict

    plt.show()

if __name__ == '__main__':
    data = load_train()
    dataset = torch.utils.data.DataLoader(data, batch_size=1)

    for data, y in dataset:
        offset = data[:, -1, :]  # [0, 0, 0, 0, 0, maxX, maxY, ..., 0]
        data = data[:, 0:data.shape[1] - 1, :]
        viz(data, y)
        # exit(0)
