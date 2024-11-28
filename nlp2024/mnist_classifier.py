import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import cv2
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from sklearn.metrics import f1_score

# install opencv: pip install opencv-python

root = "data/trainingSet"
train_rate = 0.8
learning_rate = 1e-3
epochs = 2
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        image = cv2.imread(os.path.join(root, image_file), cv2.IMREAD_UNCHANGED)

        image = self.transform(image)
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)

        return image, label

def test_dataset():
    image_path_file = "data/train_image_paths.txt"
    label_file = "data/train_labels.txt"
    trans = transforms.Compose([
        ToTensor(),
        transforms.Normalize((0.1307), (0.3081)),
    ])
    train_dataset = MnistDataset(image_path_file, label_file, trans)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    for step, batch in enumerate(train_loader):
        image, label = batch
        break

class FCModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x [B, 1, w, h] -> [B, w * h]
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class CNNModel(torch.nn.Module):
    def __init__(self):
        self.conv1 = torch.nn.Sequential([
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2,
            )
        ])

        self.conv2 = torch.nn.Sequential([
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )
        ])
        self.classifier = torch.nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.conv1(x) # [B, 32, 14, 14]
        x = self.conv2(x) # [B, 64, 7, 7]
        x = torch.reshape(x, (x.shape[0], -1)) # [B, 64 * 7 * 7]
        x = self.classifier(x) # [B, 10]

        return x

def train(dataloader, model):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch_images, batch_labels = batch
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            pred_logits = model(batch_images)
            loss = criterion(pred_logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: {}, Step: {}, Loss: {:.4f}".format(epoch, step, loss.item()))

def test(dataloader, model):
    y_true = []
    y_pred = []
    model.to(device)
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch_images, batch_labels = batch
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            batch_pred = model(batch_images) # [B, 10]
            batch_pred = torch.argmax(batch_pred, dim=1) # [B]

            y_pred.extend(batch_pred.cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

    f1 = f1_score(y_true, y_pred, average="macro")
    return f1

if __name__ == '__main__':
    # preprocess()
    # test_dataset()
    image_path_file = "data/train_image_paths.txt"
    label_file = "data/train_labels.txt"
    trans = transforms.Compose([
        ToTensor(),
        transforms.Normalize((0.1307), (0.3081)),
    ])
    train_dataset = MnistDataset(image_path_file, label_file, trans)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_path_file = "data/val_image_paths.txt"
    val_label_file = "data/val_labels.txt"
    val_dataset = MnistDataset(val_path_file, val_label_file, trans)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model = FCModel(784, 10, 100)
    model = CNNModel()

    train(train_loader, model)
    test(val_loader, model)