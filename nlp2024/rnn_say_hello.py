import torch
import numpy as np
from collections import Counter

learning_rate = 0.1
epochs = 100

input_text = "hihell"
output_text = "ihello"

# [(c, f), (), ()]
char_freq = dict(Counter(input_text + output_text))
# idx2char
# char2idx
idx2char = [c for c, f in char_freq.items()]
char2idx = {c:i for i, c in enumerate(idx2char)}

def text2onehot(text):
    output = []
    for c in text:
        idx = char2idx[c]
        vec = [0] * len(char2idx)
        vec[idx] = 1
        output.append(vec)
    return output

# test
# input_index =[char2idx[c] for c in input_text]
# print(input_index)
#
# one_hot = text2onehot(input_text)
# print(one_hot)

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=False,
            num_layers=1,
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''
        :param x: [1, s, input_size]
        :return:
        '''
        output, hidden = self.rnn(x) # [1, s, hidden_size] -> [s, hidden_size]
        output = output.view(output.shape[1], output.shape[2])
        output = self.fc(output) # [s, hidden_size] -> [s, output_size]
        return output

rnn = RNNModel(len(idx2char), 2, len(idx2char))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

for epoch in range(epochs):
    input_onehot = text2onehot(input_text)
    output_onehot = text2onehot(output_text)

    input_tensor = torch.tensor(input_onehot, dtype=torch.float32) # [6, 5]
    output_tensor = torch.tensor(output_onehot, dtype=torch.long) # [6, 5]

    # input_tensor = torch.unsqueeze(input_tensor, dim=0)
    # input_tensor = input_tensor.view(1, input_tensor.shape[0], input_tensor.shape[1])
    input_tensor = torch.reshape(input_tensor, (1, input_tensor.shape[0], input_tensor.shape[1])) # [1, 6, 5]

    output_tensor = torch.argmax(output_tensor, dim=1) # [6]

    pred_output = rnn(input_tensor) # [6, 5]

    loss = criterion(pred_output, output_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("epoch: {}, loss: {:.4f}".format(epoch, loss.item()))

    pred_output = torch.argmax(pred_output, dim=1)
    pred_output = pred_output.numpy().tolist()
    pred_output = [idx2char[i] for i in pred_output]
    print(pred_output)




