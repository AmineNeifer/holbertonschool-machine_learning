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
        inp_size = int(
            self.model.input.shape[1])  # darknet gets squared pictures

        boxes = []
        box_c = []
        box_c_b = []
        img_h = image_size[0]
        img_w = image_size[1]
        for i in range(len(outputs)):

            output = outputs[i]
            grid_height = output.shape[0]
            grid_width = output.shape[1]
            anchor_boxes = self.anchors
            box = output[..., :4]
            for cy in range(grid_height):
                for cx in range(grid_width):
                    bounding_boxes = anchor_boxes[i]
                    for j in range(len(bounding_boxes)):
                        a = bounding_boxes[j]
                        tx = box[cx, cy, j, 0]
                        ty = box[cx, cy, j, 1]
                        tw = box[cx, cy, j, 2]
                        th = box[cx, cy, j, 3]
                        bx = (self.sigmoid(tx) + cx) / grid_width
                        by = (self.sigmoid(ty) + cy) / grid_height

                        bw = a[0] * np.exp(tw) / inp_size
                        bh = a[1] * np.exp(th) / inp_size
                        box[cx, cy, j, 0] = (bx - bw / 2) * img_w
                        box[cx, cy, j, 1] = (by - bh / 2) * img_h
                        box[cx, cy, j, 2] = (bx + bw / 2) * img_w
                        box[cx, cy, j, 3] = (by + bh / 2) * img_h
            boxes.append(box)
            box_c.append(self.sigmoid(output[:, :, :, 4:5]))
            box_c_b.append(self.sigmoid(output[:, :, :, 5:]))
        return boxes, box_c, box_c_b
