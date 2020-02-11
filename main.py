import torch
from torchvision import transforms

from dataset import DarknetDataset
from utils import show_boxes
from transforms import PadToSquare

# load data
path = "data"
classes = 4
train_path = "data/train.txt"
valid_path = "data/valid.txt"
names_path = "data/classes.names"

## transforms
composed = transforms.Compose([PadToSquare()])

## dataset
image_dataset = DarknetDataset(train_path, transform=composed)
print(len(image_dataset))

sample = image_dataset[54]
image, boxes = sample['image'], sample['boxes']

print("XD")
print(image.shape)
print(boxes)

show_boxes(image, boxes)

## i want a 448x448 image here

# forward pass of the network

