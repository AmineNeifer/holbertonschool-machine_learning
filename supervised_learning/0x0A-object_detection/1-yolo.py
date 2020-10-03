#!/usr/bin/env python3


""" contains Yolo class"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """Class that uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            r = f.read().split("\n")[:-1]
        self.class_names = r
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid function """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """f sogmoid(x):
           return 1 / (1 + np.exp(-x))"""
        inp_w = self.model.input.shape[1].value
        inp_h = self.model.input.shape[2].value

        boxes = []
        box_c = []
        box_c_b = []
        for output in outputs:
            boxes.append(output[..., :4])
            box_c.append(self.sigmoid(output[..., 4:5]))
            box_c_b.append(self.sigmoid(output[..., 5:]))

        img_h = image_size[0]
        img_w = image_size[1]
        for i in range(len(outputs)):

            anchor = self.anchors[i]
            a = anchor.shape[0]
            anchor_w = anchor[:, 0]
            anchor_h = anchor[:, 1]
            grid_h = boxes[i].shape[0]
            grid_w = boxes[i].shape[1]

            tx = boxes[i][..., 0]
            ty = boxes[i][..., 1]
            tw = boxes[i][..., 2]
            th = boxes[i][..., 3]

            anchor_boxes = self.anchors
            bounding_boxes = anchor_boxes[i]

            cx = np.indices((grid_h, grid_w, a))[1]
            cy = np.indices((grid_h, grid_w, a))[0]

            bx = self.sigmoid(tx) + cx
            by = self.sigmoid(ty) + cy
            bw = anchor_w * np.exp(tw)
            bh = anchor_h * np.exp(th)

            x1 = bx - bw / 2
            x2 = x1 + bw
            y1 = by - bh / 2
            y2 = y1 + bh
            boxes[i][..., 0] = x1 * image_size[1]
            boxes[i][..., 1] = y1 * image_size[0]
            boxes[i][..., 2] = x2 * image_size[1]
            boxes[i][..., 3] = y2 * image_size[0]

        return boxes, box_c, box_c_b

        """ for cy in range(grid_height):
                for cx in range(grid_width):
                    for j in range(len(bounding_boxes)):
                        a = bounding_boxes[j]

                        tx = boxes[i][cx, cy, j, 0]
                        ty = boxes[i][cx, cy, j, 1]
                        tw = boxes[i][cx, cy, j, 2]
                        th = boxes[i][cx, cy, j, 3]

                        bx = (self.sigmoid(tx) + cx) / grid_width
                        by = (self.sigmoid(ty) + cy) / grid_height

                        bw = a[0] * np.exp(tw) / inp_w
                        bh = a[1] * np.exp(th) / inp_h

                        boxes[i][cx, cy, j, 0] = (bx - bw / 2) * img_w
                        boxes[i][cx, cy, j, 1] = (by - bh / 2) * img_h
                        boxes[i][cx, cy, j, 2] = (bx + bw / 2) * img_w
                        boxes[i][cx, cy, j, 3] = (by + bh / 2) * img_h
        """
