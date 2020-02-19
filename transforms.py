import torch
import torch.nn.functional as F
from skimage import transform

class PadToSquare():

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']
        c, h, w = image.shape

        dim_diff = abs(h - w)

        # coords before padding
        x1 = w * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h * (boxes[:, 2] + boxes[:, 4] / 2)

        # padding
        pad1 = dim_diff // 2
        pad2 = dim_diff - dim_diff // 2

        pad = (0, 0, 0, 0)
        if w > h:
            pad = (0, 0, pad1, pad2)
        elif w < h:
            pad = (pad1, pad2, 0, 0)

        image = F.pad(image, pad)

        c, padded_h, padded_w = image.shape
        # print(f"padded shape: {image.shape}")

        # adjust for padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[0]
        y2 += pad[2]

        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w / padded_w
        boxes[:, 4] *= h / padded_h

        return {'image': image, 'boxes': boxes}

class Rescale():

    def __init__(self, new_res):
        self.new_res = new_res

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']

        c, h, w = image.shape
        # print(f"rescale shape: {boxes.shape}")

        image = transform.resize(image, (3, self.new_res, self.new_res))

        return {'image': torch.from_numpy(image), 'boxes': boxes}

class SampleToYoloTensor():

    def __init__(self, grid_size, class_count):
        self.S = grid_size
        self.C = class_count

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']

        label = torch.zeros((self.S, self.S, 5 + self.C))
        print(label.shape)

        cell_size = 1.0 / 7.0
        print(f'cell size: {cell_size}')

        # iterate over all bounding boxes and fit them in respective cells
        ## if two fall into same cell I guess override the first one? (maybe 7x7 is not good enough)
        for i in range(0, boxes.shape[0]):

            box_class_idx = int(boxes[i][0]) + 5
            box_xy = boxes[i][1:3]
            box_wh = boxes[i][3:5]

            ij = (box_xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1]) # box indexes
            top_left = ij * cell_size
            xy_normalized = (box_xy - top_left) / cell_size

            # insert into label tensor
            label[i][j][0] = 1.0
            label[i][j][1:3] = xy_normalized
            label[i][j][3:5] = box_wh
            label[i][j][box_class_idx] = 1.0

        return {'image': image, 'boxes': label}
        










