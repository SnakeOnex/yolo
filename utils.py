import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_boxes(image, boxes):

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    # image = image.transpose((1, 2, 0))
    image = image.permute(1, 2, 0)
    print(image.shape)

    plt.imshow(image)

    # center coords
    y, x, _ = image.shape

    for i in range(len(boxes)):

        # bottom left
        x_coord = boxes[i][1] * x - (boxes[i][3] * x) / 2
        y_coord = boxes[i][2] * y - (boxes[i][4] * y) / 2

        # rect dims
        x_rect = x * boxes[i][3]
        y_rect = y * boxes[i][4]

        # color
        color = 'r'
        if boxes[i][0] == 0:
            color = 'b'
        elif boxes[i][0] == 1:
            color = 'y'
        elif boxes[i][0] == 2:
            color = 'o'


        rect = patches.Rectangle((x_coord, y_coord), x_rect, y_rect, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    

    plt.show()
