import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np


class VectorNetData(Dataset):
    
    def __init__(self, data) -> None:
        r"""
        :param data: a list, each element is `[X, Y]` represents the input data and label.
            `X` is a dict, consists `item_num`, `target_id`, `polyline_list`.
                `item_num`: the number of items in this data
                `target_id`: the prediction target index, 0 <= `target_id` < `item_num` 
                `polyline_list`: a list, consists `item_num` elements, each element is a set of vectors
                    Note: all data has the same length of vector.
            `Y` is a dict, consists `future`.
                `future`: the future trajectory list
        """
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]
        
    def __len__(self):
        return len(self.data)


def get_random_data(N, v_len, item_num_max=10, polyline_size_max=10, future_len=30, data_dim=2):  
    ans = []
    for i in range(N):
        item_num = torch.randint(2, item_num_max, (1,))
        target_id = torch.randint(0, item_num[0], (1,))

        polyline_list = []
        for j in range(item_num):
            v_num = torch.randint(1, polyline_size_max, (1,)).int()
            polyline = torch.rand(v_num[0], v_len)
            polyline_list.append(polyline)

        X = {"item_num" : item_num, 
            "target_id" : target_id,
            "polyline_list": polyline_list}
        
        Y = {"future" : torch.rand(future_len, data_dim)}
        ans.append([X, Y])
    return ans

if __name__ == '__main__':
    data = get_random_data(20, 5)
    
    train_data = VectorNetData(data)
    train_loader = DataLoader(dataset=train_data,
                            batch_size=1,
                            shuffle=True)

    for epoch in range(1):
        for i, data in enumerate(train_loader):
            inputs, labels = data

            print("epoch", epoch, "的第" , i, "个inputs", 
                "item_num:", inputs["item_num"].data.size(), 
                "target_id:", inputs["target_id"].data.size(), 
                "polyline_list:", len(inputs["polyline_list"]), 
                "labels", 
                "future:", labels["future"].data.size())

   # for i in range()