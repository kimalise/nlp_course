import torch
import numpy as np
from collections import Counter
import unicodedata
import string
import os
from torch.utils.data import Dataset, DataLoader

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

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

if __name__ == '__main__':
    names, labels, idx2char, char2idx, idx2label, label2idx = preprocess()
    print("finish")
