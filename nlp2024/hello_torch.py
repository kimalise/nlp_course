import torch
import numpy as np

# x = torch.Tensor(2, 3)
# print(x)
#
# x = torch.zeros(2, 3)
# print(x)
#
# x = torch.ones(2, 3)
# print(x)

# x = torch.rand(2, 3)
# print(x)
#
# y = torch.randn(2, 3)
# print(y)
#
# y.fill_(5)
# print(y)

# x = torch.tensor([1, 2, 3])
# print(x)
#
# x = np.array([1, 2, 3], dtype=np.float32)
# x = torch.from_numpy(x)

# x = torch.ones(3)
# y = torch.arange(0, 3.0)
# # [1, 1, 1] dot [0, 1, 2]
# print(x.type)
# print(y.type)

# z = torch.dot(x, y)
# print(z)

# [2, 3] * [3, 1] = [2, 1]
'''
    [
        [0, 1, 2],
        [3, 4, 5]
    ]
    [[1], [1], [1]]
'''
# x = torch.arange(6).view(2, 1, 1, 3)
# y = torch.ones(3, 1).long()
# z = torch.mm(x, y)
# print(z)
# print(x)
# y = torch.transpose(x, 0, 1)
# print(y)
# y = torch.squeeze(x)
# print(y)

# x = torch.arange(6).view(2, 3)
# y = torch.unsqueeze(x, dim=1)
# print(y.shape)

# x = torch.arange(25).view(5, 5)
# print(x)
# # y = x[3:, 3:]
# print(y)

# y = x[[1, 3], :]
# print(y)

# indices = torch.tensor([1, 3])
# y = torch.index_select(x, 0, indices)
# print(y)
# row_index = torch.tensor([0, 2])
# col_index = torch.tensor([0, 1])
# y = x[row_index, col_index]
# print(y)

# x = torch.arange(6).view(2, 3)
# y = torch.arange(6).view(2, 3)
# y = torch.arange(12).view(4, 3)
# z = torch.cat([x, y], dim=0)
# print(z)
# z = torch.stack([x, y], dim=0)
# print(z)

# z = x * y
# dz / dx = y = 3
# dz / dy = x = 2
# x = torch.tensor([2.0], requires_grad=True)
# y = torch.tensor([3.0], requires_grad=True)
# z = x * y
#
# z.backward()
# print(x.grad)
# print(y.grad)

x = torch.tensor([1, 2, 3])
device = "cuda:0" if torch.cuda.is_available() else "cpu"
x = x.to(device)
print(x)
y = torch.tensor([1, 1, 1]).to(device)
z = x + y
print(z)
z = z.cpu().numpy().tolist()
print(type(z))
print(z)






































































































































