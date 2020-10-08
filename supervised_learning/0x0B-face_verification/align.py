#!/usr/bin/env python3

""" class FaceAlign"""

import dlib
import numpy as np
import cv2


class FaceAlign:
    """ class for align faces for a better face averaging"""

    def __init__(self, shape_predictor_path):
        """ class constructor"""
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """ detecs a face in an image"""
        dets = self.detector(image, 1)
        init_area = 0
        if not dets:
            return dlib.rectangle(0, 0, image.shape[1], image.shape[0])
        for det in dets:
            area = (int(det.left()) - int(det.right())) * \
                (int(det.top()) - int(det.bottom()))
            if area >= init_area:
                result = det
                init_area = area
        return result

    def rect_to_bb(self, rect):
        """ convert x1 x2 y1 y2 to x y w h"""
        x = rect.left()
        y = rect.top()
        x1 = rect.right()
        y1 = rect.bottom()
        return (x, y, x1, y1)

    def shape_to_np(self, shape, dtype="int"):
        """ converts shape to a numpy array"""
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def find_landmarks(self, image, detection):
        """ finds facial landmarks"""
        try:
            x1, y1, x2, y2 = self.rect_to_bb(detection)
            rec = dlib.rectangle(
                left=int(x1),
                top=int(y1),
                right=int(x2),
                bottom=int(y2))
            dlib_landmarks = self.shape_predictor(image, rec)
            shape = self.shape_to_np(dlib_landmarks)
            return shape
        except BaseException:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """ aligns an image for face verification"""
        landmarks = self.find_landmarks(image, self.detect(image))
        if landmarks is None:
            return None
        inp = np.array(landmarks[landmark_indices], dtype=np.float32)
        anchor = anchor_points * size
        warp_mat = cv2.getAffineTransform(inp, anchor)
        warp_dst = cv2.warpAffine(image, warp_mat, (size, size))
        return warp_dst
