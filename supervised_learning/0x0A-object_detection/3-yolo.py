#!/usr/bin/env python3


""" contains Yolo class"""
import tensorflow.keras as K
import tensorflow as tf
import numpy as np


class Yolo():
    """Class that uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initialization"""
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
        """it processes output"""
        inp_w = self.model.input.shape[1].value
        inp_h = self.model.input.shape[2].value
        boxes = []
        box_c = []
        box_c_p = []
        for output in outputs:
            boxes.append(output[..., 0:4])
            box_c.append(self.sigmoid(output[..., 4:5]))
            box_c_p.append(self.sigmoid(output[..., 5:]))
        img_w = image_size[1]
        img_h = image_size[0]
        for i in range(len(outputs)):
            grid_h = boxes[i].shape[0]
            grid_w = boxes[i].shape[1]
            a = boxes[i].shape[2]
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            tx = boxes[i][..., 0]
            ty = boxes[i][..., 1]
            tw = boxes[i][..., 2]
            th = boxes[i][..., 3]
            cx = np.indices((grid_h, grid_w, a))[1]
            cy = np.indices((grid_h, grid_w, a))[0]
            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h
            input_w = self.model.input.shape[1].value
            input_h = self.model.input.shape[2].value
            bw = anchor_w * np.exp(tw) / input_w
            bh = anchor_h * np.exp(th) / input_h
            x1 = bx - bw / 2
            x2 = x1 + bw
            y1 = by - bh / 2
            y2 = y1 + bh
            boxes[i][..., 0] = x1 * img_w
            boxes[i][..., 1] = y1 * img_h
            boxes[i][..., 2] = x2 * img_w
            boxes[i][..., 3] = y2 * img_h
        return boxes, box_c, box_c_p

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ filter boxes """
        box_scores = []
        box_classes = []
        box_class_scores = []
        scores = []
        classes = []
        boxis = []
        for i in range(len(boxes)):
            box_scores.append(box_confidences[i] * box_class_probs[i])
            box_classes.append(np.argmax(box_scores[i], axis=-1))
            box_class_scores.append(np.max(box_scores[i], axis=-1))
            filtering_mask = box_class_scores[i] >= self.class_t
            scores += (box_class_scores[i][filtering_mask].tolist())
            boxis += (boxes[i][filtering_mask].tolist())
            classes += (box_classes[i][filtering_mask].tolist())
        scores = np.array(scores)
        boxis = np.array(boxis)
        classes = np.array(classes)
        return boxis, classes, scores

    def iou(self, box1, box2):
        """ calculates intersection over union of two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
        if w_intersection <= 0 or h_intersection <= 0:  # No overlap
            return 0
        I = w_intersection * h_intersection
        U = w1 * h1 + w2 * h2 - I  # Union = Total Area - I
        return I / U

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ non max suppression, it deletes the box that we've no need for"""
        pick = []

        idxs = np.lexsort((box_scores, -box_classes))

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]
            for pos in range(last):
                j = idxs[pos]
                if box_classes[i] == box_classes[j] and (
                        self.iou(filtered_boxes[i], filtered_boxes[j]) > self.nms_t):
                    suppress.append(pos)
            idxs = np.delete(idxs, suppress)
        return filtered_boxes[pick], box_classes[pick], box_scores[pick]
