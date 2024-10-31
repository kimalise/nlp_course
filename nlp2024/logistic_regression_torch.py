import copy
import numpy as np
import matplotlib.pyplot as plt
from torch import dtype
# install sklearn: pip install scikit-learn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# f1 = 2 * P * R / (P + R)
import torch
from torch.utils.data import Dataset, DataLoader

from nlp2024.linear_torch import criterion, optimizer

num_epochs = 100
learning_rate = 0.01
batch_size = 4

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

# dataset
class MyDataset(Dataset):
    def __init__(self, x1, x2, y):
        x1 = torch.tensor(x1)
        x2 = torch.tensor(x2)
        self.x = torch.stack([x1, x2], dim=1)
        self.y = torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# model
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x



# loss function and optimizer
# criterion = torch.nn.BCELoss()
# optimizer = torch.optim.Adam()

def test(model, dataloader):
    pred_y_list = []
    true_y_list = []
    # model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            pred_y = model(batch_x)
            pred_y = pred_y > 0.5
            pred_y_list.extend(pred_y.cpu().numpy().tolist())
            true_y_list.extend(batch_y.numpy().tolist())

    f1 = f1_score(true_y_list, pred_y_list)
    return f1

def train(model, train_dataloader, val_dataloader):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model, lr=learning_rate)

    best_f1 = -1

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch_x, batch_y = batch

            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: {}, Step: {}, Loss: {:.4f}".format(epoch, step, loss.item()))

        f1 = test(model, val_dataloader)
        print("Validation f1: {:.4f}".format(f1))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model, "best_model.pt")
