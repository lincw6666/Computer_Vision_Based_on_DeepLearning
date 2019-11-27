import matplotlib.pyplot as plt
import numpy as np
import mmcv
from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val


def imshow_bboxes(fname, bboxes):
    img = plt.imread(fname)
    plt.figure(figsize=(10.24, 7.2))
    plt.imshow(img)
    for bbox in bboxes:
        bbox = bbox.astype(np.int32)
        left_top = (bbox[0], bbox[1])
        right_bottom = (bbox[0]+bbox[2], bbox[1]+bbox[3])
        plt.gca().add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                fill=False, edgecolor='g', linewidth=2
            )
        )
    plt.show()
