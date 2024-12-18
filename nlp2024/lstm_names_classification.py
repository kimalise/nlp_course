import torch
import numpy as np
from collections import Counter
import unicodedata
import string
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import f1_score

from nlp2024.linear_torch import optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10
learning_rate = 1e-3
batch_size = 128

allowed_characters = string.ascii_letters + " .,;'"
n_letters = len(allowed_characters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )

def preprocess():
    names = []
    labels = []
    root = "data/names"
    for file_name in os.listdir(root):
        with open(os.path.join(root, file_name), "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n")
                names.append(unicodeToAscii(line))
                labels.append(os.path.splitext(file_name)[0])

    name_seq = "".join(names)
    char_freq = dict(Counter(name_seq).most_common())
    idx2char = ['<pad>'] + [c for c in char_freq.keys()]
    char2idx = {c:i for i, c in enumerate(idx2char)}

    # label_freq = dict(Counter(labels))
    idx2label = list(set(labels))
    label2idx = {l:i for i, l in enumerate(idx2label)}

    return names, labels, idx2char, char2idx, idx2label, label2idx

def split_dataset(names, labels, train_rate):
    num_train = int(len(names) * train_rate)
    shuffled_indices = np.arange(len(names))
    np.random.shuffle(shuffled_indices)

    names = np.array(names)[shuffled_indices]
    labels = np.array(labels)[shuffled_indices]

    train_names = names[:num_train]
    val_names = names[num_train:]

    train_labels = labels[:num_train]
    val_labels = labels[num_train:]

    return train_names, val_names, train_labels, val_labels

class NameDataset(Dataset):
    def __init__(self, names, labels, idx2char, char2idx, idx2label, label2idx):
        super(NameDataset, self).__init__()
        self.names = names
        self.labels = labels
        self.idx2char = idx2char
        self.char2idx = char2idx
        self.idx2label = idx2label
        self.label2idx = label2idx
        self.name_lens = [len(name) for name in self.names]
        sort_index = np.argsort(self.name_lens)
        sort_index = sort_index[::-1]
        self.names = np.array(self.names)[sort_index].tolist()
        self.labels = np.array(self.labels)[sort_index].tolist()
        self.name_lens = np.array(self.name_lens)[sort_index].tolist()

        self.names = [[self.char2idx[c] for c in name] for name in self.names]
        self.labels = [self.label2idx[l] for l in labels]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        return self.names[index], self.labels[index], self.name_lens[index]

def collate_fn(batch):
    '''
        batch: [(name, label, name_len), (), ()]
        name: [B, s] [s, B]
        label: [B]
        name_len: [B]
    '''
    names = [torch.tensor(item[0]) for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    name_lens = torch.tensor([item[2] for item in batch])

    names = pad_sequence(names, batch_first=False, padding_value=0)

    return names, labels, name_lens

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, char_vec_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(
            vocab_size,
            char_vec_dim
        )

        self.lstm = torch.nn.LSTM(
            input_size=char_vec_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            num_layers=1,
            batch_first=False,
        )

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, names, name_lens):
        '''
        :param names: [s, B]
        :param name_lens: [B]
        :return:
        '''
        names_emb = self.embedding(names) # [s, B, d]
        names_packed = pack_padded_sequence(
            names_emb,
            name_lens,
            batch_first=False
        )
        output, (h, c) = self.lstm(names_packed)
        # [2, B, d] h[0] + h[1]
        fc_output = self.fc(h[0] + h[1]) # [B, 18]
        # fc_output = self.fc(torch.cat([h[0], h[1]], dim=1))
        return fc_output

def test_dataset():
    names, labels, idx2char, char2idx, idx2label, label2idx = preprocess()
    dataset = NameDataset(names, labels, idx2char, char2idx, idx2label, label2idx)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    for step, batch in enumerate(dataloader):
        print(batch)

def train(model, train_dataloader, val_dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_f1 = -1
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            names, labels, name_lens = batch
            names = names.to(device)
            labels = labels.to(device)
            name_lens = name_lens.to(device)

            pred_logits = model(names, name_lens)
            loss = criterion(pred_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: {}, Step: {}, Loss: {:.4f}".format(epoch, step, loss.item()))

        f1 = test(model, val_dataloader)
        print("Validation f1: {:.4f}".format(f1))
        if f1 > best_f1:
            torch.save(model.state_dict(), "best_model.pth")
            best_f1 = f1

def test(model, dataloader):
    model.eval()

    golden_labels = []
    predicted_labels = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            names, labels, name_lens = batch
            names = names.to(device)
            labels = labels.to(device)
            name_lens = name_lens.to(device)

            pred_logits = model(names, name_lens)
            pred_labels = torch.argmax(pred_logits, dim=1)
            golden_labels.extend(labels.cpu().numpy().tolist())
            predicted_labels.extend(pred_labels.cpu().numpy().tolist())

    f1 = f1_score(golden_labels, predicted_labels)
    return f1

if __name__ == '__main__':
    # names, labels, idx2char, char2idx, idx2label, label2idx = preprocess()
    # print("finish")
    test_dataset()
