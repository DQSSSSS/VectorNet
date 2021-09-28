import torch

from dataloader.dataloader import VectorNetDataset
from torch.utils.data import DataLoader

class ArgoverseDataloader:

    # Note: future trajectory is saved as the pre-step offset, not the absolute coordinate. See paper section 3.4

    def __init__(self) -> None:
        # TODO
        # self.training_dataloader = 
        # self.test_dataloader = 
        # self.eval_dataloader = 
        pass