import torch.nn.functional as F

class PadToSquare():

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']
        c, h, w = image.shape
        print(image.shape)

        print("hello")
        print("hello")
        print("hello")

        dim_diff = abs(h - w)
        print(dim_diff)

        # coords before padding
        x1 = w * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h * (boxes[:, 2] + boxes[:, 4] / 2)

        print(x1)
        print(y1)

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
        print(f"padded shape: {image.shape}")

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

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        pass
