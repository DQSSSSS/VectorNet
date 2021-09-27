import torch
import config

def askADE(a, b):
    r"""
    Calculate Average Displacement Error(ADE).
    :param a:
    :param b:
    :return: ADE
    """
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
