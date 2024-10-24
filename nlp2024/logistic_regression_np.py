import numpy as np
import matplotlib.pyplot as plt
from torch import dtype

# 生成数据
def generate_data():
    x1 = np.array([1, 4], dtype=np.float64)
    x2 = np.array([1, 4], dtype=np.float64)
    y = np.array([0, 1])

    x1 = np.repeat(x1, 100)
    x2 = np.repeat(x2, 100)
    y = np.repeat(y, 100)
    # print('finish')

    x1 += np.random.randn(x1.shape[0]) * 1.0
    x2 += np.random.randn(x2.shape[0]) * 1.0

    # plt.scatter(x1, x2, c=y)
    # plt.show()

    return x1, x2, y

def split_data(x1, x2, y, train_rate=0.8):
    index = np.arange(x1.shape[0])
    np.random.shuffle(index)

    x1 = x1[index]
    x2 = x2[index]
    y = y[index]

    num_train = int(x1.shape[0] * train_rate)
    train_x1 = x1[:num_train]
    train_x2 = x2[:num_train]
    train_y = y[:num_train]

    val_x1 = x1[num_train:]
    val_x2 = x2[num_train:]
    val_y = y[num_train:]

    return train_x1, train_x2, train_y, val_x1, val_x2, val_y


