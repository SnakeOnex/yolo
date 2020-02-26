import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset import DarknetDataset
from utils import show_boxes
from transforms import PadToSquare, Rescale, SampleToYoloTensor

from model import YOLO
from loss import YOLOLoss

# load data
path = "data"
classes = 4
train_path = "data/train.txt"
valid_path = "data/valid.txt"
names_path = "data/classes.names"

input_size = 448
bs = 16
epochs = 1
checkpoint_interval = 1

## transforms
composed = transforms.Compose([PadToSquare(), Rescale(input_size), SampleToYoloTensor(7, classes)])

## dataset
train_data = DarknetDataset(train_path, transform=composed)
valid_data = DarknetDataset(valid_path, transform=composed)
print(len(train_data))
print(len(valid_data))

def save_checkpoint(path):
    path = f'checkpoints/{path}'
    torch.save({
        'epoch': epoch,
        'model_state_dict': yolo.state_dict(),
        'optim_state_dict': optim.state_dict(),
        }, path)

    print('saved to: ', path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    yolo.load_state_dict(checkpoint['model_state_dict'])
    # optim.load_state_dict(checkpoint['optim_state_dict'])

    print("loaded checkpoint: ", path)

train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=False, num_workers=0)
valid_dataloader = DataLoader(valid_data, batch_size=bs, shuffle=False, num_workers=0)

# show_boxes(image, boxes)
yolo = YOLO(4, 2)
loss_fn = YOLOLoss(classes, 2)

optim = torch.optim.SGD(yolo.parameters(), lr=0.001, momentum=0.9)

# checkpoint_path = "checkpoints/test2.checkpoint"
# yolo.load_state_dict(torch.load(checkpoint_path))
# print("loading checkpoint")

for epoch in range(epochs):
    print("epoch: ", epoch)

    # validation set eval
    with torch.no_grad():
        valid_loss = 0.
        for i_batch, sample_batched in enumerate(valid_dataloader):

            output = yolo(sample_batched['image'].float())

            valid_loss += loss_fn.forward(output, sample_batched['boxes'])

        valid_loss /= len(valid_data) // 8
        print("valid_loss: ", valid_loss)

    # train set epoch
    train_loss = 0.
    for i_batch, sample_batched in enumerate(train_dataloader):

        output = yolo(sample_batched['image'].float())

        loss = loss_fn.forward(output, sample_batched['boxes'])
        train_loss += loss

        print(f"{epoch}: {i_batch} loss: {loss}")


        optim.zero_grad()
        loss.backward()

        optim.step()

    train_loss /= len(train_data) // 8
    print("train_loss: ", train_loss)

    if epoch % checkpoint_interval == 0:
        print("Saving checkpoint")
        torch.save(yolo.state_dict(), f"checkpoints/yolo_ckpt_{epoch}.pth")
