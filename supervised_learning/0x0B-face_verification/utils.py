#!/usr/bin/env python3

""" tools to use for our face verif program"""
import cv2
from glob import glob
import numpy as np


def load_images(images_path, as_array=True):
    """ loads images using cv2 in RGB"""
    images = []
    filenames = []
    for path in sorted(glob(images_path + '/*')):
        images.append(cv2.cvtColor(cv2.imread(path), code=cv2.COLOR_BGR2RGB))
        filenames.append(path.split("/")[-1])
    if as_array is True:
        images = np.array(images)
    return images, filenames


def load_csv(csv_path, params={}):
    with open(csv_path, **params) as f:
        triplets = [x.split(',') for x in f.read().split('\n')[:-1]]
    return triplets
