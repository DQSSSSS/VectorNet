import torch
from torch.utils.data import Dataset

class VectorNetDataset(Dataset):
    
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



