import torch
import numpy as np
from torch import dtype
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import MSELoss

from nlp2024.linear_numpy import learning_rate

max_epoch = 100
learning_rate = 1e-2
batch_size = 1

# 准备数据
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

dataset = MyDataset(xs, ys)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# test MyDataset
# dataset = MyDataset(xs, ys)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
#
# for batch in dataloader:
#     print(batch)

# 设计模型类
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x

# test model
# model = LinearModel()
# x = torch.tensor([1], dtype=torch.float)
# y = model(x)
# print(y)

model = LinearModel()

# 创建损失函数和优化器
criterion = MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(max_epoch):
    for step, batch in enumerate(dataloader):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1 == 0:
            loss_v = loss.cpu().item()
            print("Epoch: {}, Step: {}, Loss: {:.4f}".format(epoch, step, loss_v))

# 测试
pred_x = torch.tensor([5.0])
pred_y = model(pred_x)
print(pred_y)







