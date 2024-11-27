import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import cv2

# install opencv: pip install opencv-python

root = "data/trainingSet"
train_rate = 0.8

def write_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")

def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = [line.strip("\n") for line in lines]

    return data

def preprocess():
    image_paths = []
    labels = []
    for path in os.listdir(root):
        for image_file in os.listdir(os.path.join(root, path)):
            image_paths.append(os.path.join(path, image_file))
            labels.append(path)

    indices = np.array(list(range(len(image_paths))))
    np.random.shuffle(indices)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    image_paths = image_paths[indices]
    labels = labels[indices]

    num_train = int(len(image_paths) * train_rate)

    train_image_paths = image_paths[:num_train]
    train_labels = labels[:num_train]

    val_image_paths = image_paths[num_train:]
    val_labels = image_paths[num_train:]

    write_data(train_image_paths, "data/train_image_paths.txt")
    write_data(train_labels, "data/train_labels.txt")
    write_data(val_image_paths, "data/val_image_paths.txt")
    write_data(val_labels, "data/val_labels.txt")

    return

class MnistDataset(Dataset):
    def __init__(self, image_path_file, label_file, transform):
        super(MnistDataset, self).__init__()
        self.image_paths = read_data(image_path_file)
        self.labels = read_data(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_file = self.image_paths[index]
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

        image = self.transform(image)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, label

if __name__ == '__main__':
    # preprocess()
    pass