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

## transforms
composed = transforms.Compose([PadToSquare(), Rescale(input_size), SampleToYoloTensor(7, classes)])

## dataset
train_data = DarknetDataset(train_path, transform=composed)
valid_data = DarknetDataset(valid_path, transform=composed)
print(len(train_data))
print(len(valid_data))



train_dataloader = DataLoader(train_data, batch_size=8, shuffle=False, num_workers=0)
valid_dataloader = DataLoader(valid_data, batch_size=8, shuffle=False, num_workers=0)

# show_boxes(image, boxes)
yolo = YOLO(4, 2)
loss_fn = YOLOLoss(classes, 2)

optim = torch.optim.SGD(yolo.parameters(), lr=5 * 1e-5)

for i in range(20):
    print("epoch: ", i)

    train_loss = 0
    valid_loss = 0.
    for i_batch, sample_batched in enumerate(valid_dataloader):
        # # print(i_batch, sample_batched['image'].size(), sample_batched['boxes'].size())

        output = yolo(sample_batched['image'].float())
        # # print(output.shape)

        valid_loss += loss_fn.forward(output, sample_batched['boxes'])
        # # print(f"{i}: {i_batch} loss: {loss}")

    valid_loss /= len(valid_data)
    print("valid_loss: ", valid_loss)

    for i_batch, sample_batched in enumerate(train_dataloader):
        # print(i_batch, sample_batched['image'].size(), sample_batched['boxes'].size())

        output = yolo(sample_batched['image'].float())
        # print(output.shape)

        loss = loss_fn.forward(output, sample_batched['boxes'])
        train_loss += loss

        if i_batch == 0 and i == 0:
            print("train_loss: ", train_loss)
        # print(f"{i}: {i_batch} loss: {loss}")


        loss.backward()
        optim.step()

    train_loss /= len(train_data)
    print("train_loss: ", train_loss)

    if i == 1:
        print("changing lr")
        optim = torch.optim.SGD(yolo.parameters(), lr=1e-6)
    elif i == 2:
        print("changing lr2")
        optim = torch.optim.SGD(yolo.parameters(), lr=1e-7 * 5)
    elif i == 6:
        print("changing lr3")
        optim = torch.optim.SGD(yolo.parameters(), lr=1e-7)




## i want a 448x448 image here

# forward pass of the network

