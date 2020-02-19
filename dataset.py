import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

from PIL import Image
from skimage import io, transform

class DarknetDataset(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        # img paths
        with open(path, "r") as f:
            self.img_files = f.readlines()

        self.img_files = [path.replace("\n", "") for path in self.img_files]


        self.label_files = [path.replace("images", "labels").replace(".jpg", ".txt") for path in self.img_files]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        img = io.imread(img_path)
        # print(img.shape)
        img = np.transpose(img, (2, 0, 1))
        # print(img.shape)

        img = torch.from_numpy(img)

        boxes = torch.from_numpy(np.loadtxt(label_path)).reshape((-1, 5))
        sample = {'image': img, 'boxes': boxes}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    from transforms import PadToSquare, Rescale, SampleToYoloTensor
    from torchvision import transforms
    # load data
    train_path = "data/train.txt"
    input_size = 448

    composed = transforms.Compose([PadToSquare(), Rescale(input_size), SampleToYoloTensor(7, 4)])
    image_dataset = DarknetDataset(train_path, transform=composed)

    print(len(image_dataset))
    sample = image_dataset[0]
    image, boxes = sample['image'], sample['boxes']
    print(image.shape)
    print(boxes.shape)
