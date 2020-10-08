#!/usr/bin/env python3

""" tools to use for our face verif program"""
import cv2
from glob import glob
import numpy as np
import csv


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
    """ loads triplets of names from csv file"""
    """with open(csv_path, **params) as f:
        triplets = [x.split(',') for x in f.read().split('\n')[:-1]]
    return triplets"""
    triplets = []
    with open(csv_path, 'r') as f:
        csv_r = csv.reader(f, **params)
        for line in csv_r:
            triplets.append(line)
    return triplets


def save_images(path, images, filenames):
    """ saves images to a specific path"""
    try:
        for filename, image in zip(filenames, images):
            name = path + "/" + filename
            stat = cv2.imwrite(
                name, cv2.cvtColor(
                    image, code=cv2.COLOR_RGB2BGR))
    except BaseException:
        return False
    return stat