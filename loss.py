import torch
import torch.nn as nn

class YOLOLoss(nn.Module):

    def __init__(self, classes, bboxes):
        self.C = classes
        self.B = bboxes
        self.lamb_coord = 5.
        self.lamb_noobj = 0.5

    def compute_iou(self, pred_xy_min, pred_xy_max, target_xy_min, target_xy_max):
        """
        ret: iou -1 x S x S x B
        """
        intersect_mins = torch.max(pred_xy_min, target_xy_min)
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
        # print(f"pred: {pred.shape}")
        # print(f"target: {target.shape}")

        coord_mask = target[:, :, :, 0] # -1 x S x S
        coord_mask = torch.unsqueeze(coord_mask, 3)
        noobj_mask = (1 - coord_mask) # -1 x S x S

        # print(f"coord_mask: {coord_mask.shape}")
        # print(f"noobj_mask: {noobj_mask.shape}")

        target_class = target[:, :, :, 5:(5+self.C)] # -1 x S x S x C
        target_box = target[:, :, :, 1:5] # -1 x S x S x 4
        target_box = target_box.unsqueeze(3) # -1 x S x S x 1 x 4

        # print(f"target_class: {target_class.shape}")
        # print(f"target_box: {target_box.shape}")

        pred_class = pred[:, :, :, (self.B*5):(self.B*5+self.C)] # -1 x S x S x C
        pred_conf_box = pred[:, :, :, 0:(self.B*5)].view(-1, 7, 7, self.B, 5) # -1 x S x S x B x 5
        pred_conf = pred_conf_box[:, :, :, :, 0]

        # print(f"pred_class: {pred_class.shape}")
        # print(f"pred_conf_box: {pred_conf_box.shape}")
        # print(f"pred_conf: {pred_conf.shape}")

        pred_box = pred_conf_box[:, :, :, :, 1:5] # -1 x S x S x B x 4

        # print(f"pred_box: {pred_box.shape}")

        target_xy, target_wh = target_box[:, :, :, :, 0:2], target_box[:, :, :, :, 2:4]
        pred_xy, pred_wh = pred_box[:, :, :, :, 0:2], pred_box[:, :, :, :, 2:4]

        target_xy_min, target_xy_max = self.xywh_to_minmax(target_xy, target_wh)
        pred_xy_min, pred_xy_max = self.xywh_to_minmax(pred_xy, pred_wh)

        # print(f"target_xy_min: {target_xy_min.shape}")
        # print(f"pred_xy_min: {pred_xy.shape}")

        iou = self.compute_iou(pred_xy_min, pred_xy_max, target_xy_min, target_xy_max) # -1 x S x S x 2

        # print(f"iou: {iou.shape}")

        box_val, box_mask = torch.max(iou, axis=3, keepdim=True)
        # print(f"box_val: {box_val.shape}")
        # print(f"box_mask: {box_mask.shape}")

        # confidence loss
        noobj_loss = self.lamb_noobj * (noobj_mask * box_mask) * torch.pow(pred_conf, 2)
        obj_loss = (coord_mask * box_mask) * torch.pow(1 - pred_conf, 2)
        conf_loss = torch.sum(noobj_loss + obj_loss)

        # print(f"noobj_loss: {noobj_loss.shape}")
        # print(f"obj_loss: {obj_loss.shape}")
        # print(conf_loss)

        # class loss
        class_loss = coord_mask * torch.pow(target_class - pred_class, 2)
        class_loss = torch.sum(class_loss)
        # print(class_loss)

        # box loss
        box_loss_mask = (coord_mask * box_mask).unsqueeze(4)
        box_loss = self.lamb_coord * box_loss_mask * torch.pow(target_xy - pred_xy, 2)
        box_loss += self.lamb_coord * box_loss_mask * torch.pow(torch.sqrt(target_wh) - torch.sqrt(pred_wh), 2)
        box_loss = torch.sum(box_loss)
        # print("box_loss: ", box_loss)

        # print(f"conf: {conf_loss}, clas: {class_loss}, box: {box_loss}")
        loss = conf_loss + class_loss + box_loss
        return loss


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
