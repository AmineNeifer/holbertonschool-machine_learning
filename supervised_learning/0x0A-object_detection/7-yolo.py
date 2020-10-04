#!/usr/bin/env python3


""" contains Yolo class"""
import tensorflow.keras as K
import tensorflow as tf
import numpy as np
from glob import glob
import cv2
import os


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

    def load_images(self, folder_path):
        """ loads images using cv2"""
        images = []
        images_p = []
        for path in glob(folder_path + '/*'):
            images.append(cv2.imread(path))
            images_p.append(path)
        return images, images_p

    def preprocess_images(self, images):
        """ preprocess images"""
        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value
        ni = len(images)
        nc = 3
        img_shapes = []
        r = []

        for a in images:
            img_shapes.append([a.shape[0], a.shape[1]])
            r.append(
                cv2.resize(a, dsize=(input_w,
                                     input_h),
                           interpolation=cv2.INTER_CUBIC)
            )
        r = [x / 255 for x in r]
        return np.array(r), np.array(img_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """ displays boxes in the real pictures"""
        c_names = self.class_names
        for i in range(boxes.shape[0]):
            # draw rectangles
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            r_color = (255, 0, 0)
            image = cv2.rectangle(image, pt1, pt2, color=r_color, thickness=2)
            # draw text
            b_score = np.around(box_scores, 2)[i]
            class_name = c_names[box_classes[i]]
            text = str(class_name) + " " + str(b_score)
            pt3 = (x1, y1 - 5)
            b_color = (0, 0, 255)
            image = cv2.putText(
                image,
                text,
                pt3,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                b_color,
                1,
                cv2.LINE_AA)
        cv2.imshow(file_name, image)
        k = cv2.waitKey(0)
        if k == ord('s'):
            os.mkdir("detections")
            cv2.imwrite("detections/" + file_name, image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """ all functions above"""
        output1 = np.random.randn(13, 13, 3, 85)
        output2 = np.random.randn(26, 26, 3, 85)
        output3 = np.random.randn(52, 52, 3, 85)
        for i in range(len(images)):
            boxes, box_confidences, box_class_probs = yolo.process_outputs(
                [output1, output2, output3], np.array([500, 700]))
            boxes, box_classes, box_scores = yolo.filter_boxes(
                boxes, box_confidences, box_class_probs)
            boxes, box_classes, box_scores = yolo.non_max_suppression(
                boxes, box_classes, box_scores)
            images, image_paths = yolo.load_images(folder_path)
            yolo.show_boxes(images[i], boxes, box_classes, box_scores, )
