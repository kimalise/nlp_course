import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

xs = np.array([1, 2, 3])
ys = np.array([2, 4, 6])

class MyDataset(Dataset):
    def __init__(self, xs, ys):
        super(MyDataset, self).__init__()
        self.x_tensor = torch.tensor(xs)
        self.y_tensor = torch.tensor(ys)

    # callback function
    def __len__(self):
        return self.x_tensor.shape[0]

    def __getitem__(self, index):
        return self.x_tensor[index], self.y_tensor[index]





