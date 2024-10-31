import copy
import numpy as np
import matplotlib.pyplot as plt
from torch import dtype
# install sklearn: pip install scikit-learn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# f1 = 2 * P * R / (P + R)

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

class DataLoader():
    def __init__(self, x1, x2, y, batch_size):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.batch_size = batch_size

    def get_batch(self, batch_index):
        start = batch_index * self.batch_size
        end = min(len(self.x1), (batch_index + 1) * self.batch_size)
        return self.x1[start:end], self.x2[start:end], self.y[start:end]

# 数据集
def test_dataloader():
    x1, x2, y = generate_data()
    data = split_data(x1, x2, y, train_rate=0.8)
    train_x1, train_x2, train_y, val_x1, val_x2, val_y = data
    batch_size = 4
    dataloader = DataLoader(train_x1, train_x2, train_y, batch_size)
    num_batch = len(train_x1) // batch_size
    for i in range(num_batch):
        batch = dataloader.get_batch(i)
        print(batch)

# 逻辑回归模型
class LogisticRegressionModel():
    def __init__(self):
        self.w1 = 0
        self.w2 = 0
        self.b = 0
        self.grad_w1 = 0
        self.grad_w2 = 0
        self.grad_b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x1, x2):
        z = self.w1 * x1 + self.w2 * x2 + self.b
        return self.sigmoid(z)

    def loss(self, x1, x2, y):
        l = y * np.log(self.forward(x1, x2)) + (1 - y) * np.log(1 - self.forward(x1, x2))
        return -np.mean(l)

    def backward(self, x1, x2, y):
        self.grad_w1 = np.sum((self.forward(x1, x2) - y) * x1)
        self.grad_w2 = np.sum((self.forward(x1, x2) - y) * x2)
        self.grad_b = np.sum((self.forward(x1, x2) - y))

    def step(self, lr):
        self.w1 = self.w1 - lr * self.grad_w1
        self.w2 = self.w2 - lr * self.grad_w2
        self.b = self.b - lr * self.grad_b

def train(model, train_dataloader, val_dataloader):
    best_f1 = -1
    best_model = LogisticRegressionModel()

    num_batch = len(train_dataloader.x1) // batch_size
    for epoch in range(num_epochs):
        for step in range(num_batch):
            batch = train_dataloader.get_batch(step)
            batch_x1, batch_x2, batch_y = batch
            pred_y = model.forward(batch_x1, batch_x2)
            loss = model.loss(batch_x1, batch_x2, batch_y)

            model.backward(batch_x1, batch_x2, batch_y)
            model.step(learning_rate)

            print("Epoch: {}, Step: {}, Loss: {:.4f}".format(epoch, step, loss))

        # validation
        f1 = test(model, val_dataloader)
        if f1 > best_f1:
            best_model = copy.deepcopy(model)
            best_f1 = f1
        print("f1: {:.4f}, best_f1: {:.4f}".format(f1, best_f1))

    return best_model

def test(model, dataloader):
    pred_y = []
    true_y = []
    num_batch = len(dataloader.x1) // dataloader.batch_size
    for step in range(num_batch):
        batch = dataloader.get_batch(step)
        batch_x1, batch_x2, batch_y = batch
        batch_pred_y = model.forward(batch_x1, batch_x2)
        batch_pred_y = batch_pred_y > 0.5
        pred_y.extend(batch_pred_y.tolist())
        true_y.extend(batch_y.tolist())

    f1 = f1_score(true_y, pred_y, average="macro")
    return f1

# w1 * x1 + w2 * x2 + b = 0
# x2 = -(w1 / w2) * x1 - (b / w2)
def boundary_line(model, x):
    return -(model.w1 / model.w2) * x - (model.b / model.w2)

if __name__ == '__main__':
    # test_dataloader()
    x1, x2, y = generate_data()
    train_x1, train_x2, train_y, val_x1, val_x2, val_y = split_data(x1, x2, y)
    train_dataloader = DataLoader(train_x1, train_x2, train_y, batch_size)
    val_dataloader = DataLoader(val_x1, val_x2, val_y, batch_size)

    model = LogisticRegressionModel()

    best_model = train(model, train_dataloader, val_dataloader)

    plt.scatter(x1, x2, c=y)

    line_x1 = []
    line_x2 = []
    for i in np.arange(0, 6, 0.1):
        j = boundary_line(best_model, i)
        line_x1.append(i)
        line_x2.append(j)
    plt.plot(line_x1, line_x2)
    plt.show()

    print("finish")
