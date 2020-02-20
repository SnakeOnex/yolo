import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLO(nn.Module):
    def __init__(self, classes, bboxes):
        super(YOLO, self).__init__()
        self.classes = classes
        self.bboxes = bboxes

        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv7 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.fc1 = nn.Linear(7 * 7 * 1024, 256)
        self.fc2 = nn.Linear(256, 4096)
        self.output = nn.Linear(4096, 7 * 7 * (5 * self.bboxes + self.classes))


    def forward(self, x):
        # 1st layer
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = x.view(-1, 7 * 7 * 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        x = x.view(-1, 7, 7, 5 * self.bboxes + self.classes)
        return x


if __name__ == '__main__':
    from dataset import DarknetDataset
    from torchvision import transforms
    from transforms import PadToSquare, Rescale, SampleToYoloTensor
    from torch.utils.data import DataLoader

    train_path = "data/train.txt"

    composed = transforms.Compose([PadToSquare(), Rescale(448), SampleToYoloTensor(7, 4)])
    image_dataset = DarknetDataset(train_path, transform=composed)

    dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=4)

    classes = 4
    bboxes = 2
    net = YOLO(classes, bboxes)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['boxes'].size())
        output = net(sample_batched['image'].float())
        print(output.shape)
        # check output tensor size, should be [1, 7, 7, 14]
        break
