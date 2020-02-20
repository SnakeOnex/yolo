import torch
import torch.nn as nn

class YOLOLoss(nn.Module):

    def __init__(self, classes, bboxes):
        self.C = classes
        self.B = bboxes

    def compute_iou(self, box1, box2):
        pass

    def forward(self, pred, target):
        """
        pred: S x S x (B * 5 + C)
        target: S x S x (5 + C)
        """
        print(f"pred: {pred.shape}")
        print(f"target: {target.shape}")

        coord_mask = target[:, :, :, 0] # -1 x S x S
        noobj_mask = target[:, :, :, 0] # -1 x S x S

        print(f"coord_mask: {coord_mask.shape}")
        print(f"noobj_mask: {noobj_mask.shape}")

        target_class = target[:, :, :, 5:(5+self.C)]
        target_box = target[:, :, :, 1:5]

        print(f"target_class: {target_class.shape}")
        print(f"target_box: {target_box.shape}")

        pred_class = pred[:, :, :, (self.B*5):(self.B*5+self.C)] # -1 x S x S x C
        pred_conf_box = pred[:, :, :, 0:(self.B*5)].view(-1, 7, 7, self.B, 5) # -1 x S x S x B x 5

        print(f"pred_class: {pred_class.shape}")
        print(f"pred_conf_box: {pred_conf_box.shape}")

        pred_box = pred_conf_box[:, :, :, :, 1:5] # -1 x S x S x B x 4

        print(f"pred_box: {pred_box.shape}")


        pass


if __name__ == '__main__':
    from dataset import DarknetDataset
    from torchvision import transforms
    from transforms import PadToSquare, Rescale, SampleToYoloTensor
    from torch.utils.data import DataLoader

    from model import YOLO

    train_path = "data/train.txt"

    composed = transforms.Compose([PadToSquare(), Rescale(448), SampleToYoloTensor(7, 4)])
    image_dataset = DarknetDataset(train_path, transform=composed)

    dataloader = DataLoader(image_dataset, batch_size=2, shuffle=False, num_workers=4)

    classes = 4
    bboxes = 2
    net = YOLO(classes, bboxes)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['boxes'].size())
        output = net(sample_batched['image'].float())
        print(output.shape)

        loss_fn = YOLOLoss(4, 2)
        loss = loss_fn.forward(output, sample_batched['boxes'])
        print(f"loss: {loss}")
        
        # check output tensor size, should be [1, 7, 7, 14]
        break
