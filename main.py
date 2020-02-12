import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset import DarknetDataset
from utils import show_boxes
from transforms import PadToSquare, Rescale

from model import YOLO

# load data
path = "data"
classes = 4
train_path = "data/train.txt"
valid_path = "data/valid.txt"
names_path = "data/classes.names"

input_size = 448

## transforms
composed = transforms.Compose([PadToSquare(), Rescale(448)])

## dataset
image_dataset = DarknetDataset(train_path, transform=composed)
print(len(image_dataset))


dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=4)

sample = image_dataset[444]
image, boxes = sample['image'], sample['boxes']

print("XD")
print(image.shape)
print(boxes.shape)

# show_boxes(image, boxes)
yolo = YOLO(3)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['boxes'].size())
    output = yolo(sample_batched['image'].float())
    print(output.shape)
    break


## i want a 448x448 image here

# forward pass of the network

