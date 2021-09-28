import torch

from dataloader.dataloader import VectorNetDataset
from torch.utils.data import DataLoader

class RandomDataloader:

    def __init__(self, training_size, test_size, eval_size, v_len) -> None:
        self.training_dataloader = DataLoader(dataset=VectorNetDataset(self.get_random_data(training_size, v_len)),
                                                batch_size=1, shuffle=True)
        self.test_dataloader = DataLoader(dataset=VectorNetDataset(self.get_random_data(test_size, v_len)),
                                            batch_size=1, shuffle=False)
        self.eval_dataloader = DataLoader(dataset=VectorNetDataset(self.get_random_data(eval_size, v_len)),
                                            batch_size=1, shuffle=False)
 
    def get_random_data(self, N, v_len, item_num_max=10, polyline_size_max=10, future_len=30, data_dim=2):  
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

    loader = RandomDataloader(20, 5, 2, 5)

    for epoch in range(1):
        for i, data in enumerate(loader.training_dataloader):
            inputs, labels = data

            print("epoch", epoch, "的第" , i, "个inputs", 
                "item_num:", inputs["item_num"].data.size(), 
                "target_id:", inputs["target_id"].data.size(), 
                "polyline_list:", len(inputs["polyline_list"]), 
                "labels", 
                "future:", labels["future"].data.size())

   # for i in range()