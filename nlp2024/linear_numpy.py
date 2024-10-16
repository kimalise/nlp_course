import numpy as np
# import matplotlib as plt

# hyper parameters
max_epoch = 100
learning_rate = 0.01

# data
xs = np.array([1, 2, 3])
ys = np.array([2, 4, 6])

# y = w * x
# L = (pred_y - y) ** 2 = (w * x - y) ** 2
# dL / dw = 2 * x * (w * x - y)
# w = w - learning_rate * dL / dw
# lim f(x + delta) - f(x) / delta

# model
class Model():
    def __init__(self):
        self.w = 0

    def forward(self, x):
        return self.w * x

    def loss_fn(self, x, y):
        return (self.w * x - y) ** 2

    def gradient(self, x, y):
        return 2 * x * (self.w * x - y)
    # def gradient(self, x, y):
    #     delta = 1e-5
    #     # return (self.forward(x + delta) - self.forward(x)) / delta
    #     return (((self.w + delta) * x - y) ** 2 - (self.w * x - y) ** 2) / delta

# model object
model = Model()

# train model
for epoch in range(max_epoch):
    for step, batch in enumerate(zip(xs, ys)):
        batch_x, batch_y = batch

        pred_y = model.forward(batch_x)
        loss = model.loss_fn(batch_x, batch_y)

        grad = model.gradient(batch_x, batch_y)
        model.w = model.w - learning_rate * grad

        print("Epoch: {}, Step: {}, Loss: {:.4f}".format(epoch, step, loss))

# model test
test_x = 5
pred_y = model.forward(test_x)
print(pred_y)



















