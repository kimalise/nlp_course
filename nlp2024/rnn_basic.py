import torch

# unbatched
# rnn = torch.nn.RNN(
#     input_size=10,
#     hidden_size=8,
#     num_layers=1,
#     bidirectional=False,
#     batch_first=True,
# )
#
# h = torch.zeros(1, 8)
# x = torch.randn(1, 10)
# output, h = rnn(x, h)

rnn = torch.nn.RNN(
    input_size=10,
    hidden_size=8,
    num_layers=3,
    bidirectional=True,
    batch_first=False,
)

h = torch.zeros(6, 4, 8)
# x = torch.randn(4, 100, 10)
x = torch.randn(100, 4, 10)

output, h = rnn(x, h)

print("finish")