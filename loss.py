import torch
import torch.nn as nn

class YOLOLoss(nn.Module):

    def __init__(self, classes, bboxes):
        self.C = classes
        self.B = bboxes

    def compute_iou(self, pred_xy_min, pred_xy_max, target_xy_min, target_xy_max):
        """
        ret: iou -1 x S x S x B
        """
        intersect_mins = torch.max(pred_xy_min, target_xy_min)
        print(f"intersect_mins: {intersect_mins.shape}")
        intersect_maxes = torch.min(pred_xy_max, target_xy_max)
        intersect_wh = torch.clamp(intersect_maxes - intersect_mins, min=0.) # -1 x 7 x 7 x B x 2
        intersect_areas = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

        pred_wh = pred_xy_max -pred_xy_min
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        target_wh = target_xy_max - target_xy_min
        target_areas = target_wh[..., 0] * target_wh[..., 1]

        union_areas = pred_areas + target_areas - intersect_areas

        iou = intersect_areas / union_areas

        return iou

    def xywh_to_minmax(self, xy, wh):
        """
        args:
        xy: -1 x S x S x 2
        wh: -1 x S x S x 2
        """

        xy_min = xy - (wh / 2)
        xy_max = xy + (wh / 2)

        return xy_min, xy_max

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
        target_box = target_box.unsqueeze(3)

        print(f"pred_box: {pred_box.shape}")
        print(f"target_box: {target_box.shape}")

        target_xy, target_wh = target_box[:, :, :, :, 0:2], target_box[:, :, :, :, 2:4]
        pred_xy, pred_wh = pred_box[:, :, :, :, 0:2], pred_box[:, :, :, :, 2:4]

        target_xy_min, target_xy_max = self.xywh_to_minmax(target_xy, target_wh)
        pred_xy_min, pred_xy_max = self.xywh_to_minmax(pred_xy, pred_wh)

        print(f"target_xy_min: {target_xy_min.shape}")
        print(f"pred_xy_min: {pred_xy.shape}")

        iou = self.compute_iou(pred_xy_min, pred_xy_max, target_xy_min, target_xy_max) # -1 x S x S x 2



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
