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
        images = np.asarray(images)
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
    except Exception as e:
        return False
    return stat


def generate_triplets(images, filenames, triplet_names):
    """ generate triplets from triplets of names"""
    i = 0
    m = len(triplet_names)
    n = images.shape[1]
    A = np.zeros((m, n, n, 3))
    P = np.zeros((m, n, n, 3))
    N = np.zeros((m, n, n, 3))
    for i in range(m):
        idx = filenames.index(triplet_names[i][0] + ".jpg")
        idx1 = filenames.index(triplet_names[i][1] + ".jpg")
        idx2 = filenames.index(triplet_names[i][2] + ".jpg")
        A[i] = np.array(images[idx] / 255)
        P[i] = np.array(images[idx1] / 255)
        N[i] = np.array(images[idx2] / 255)
    return [A, P, N]
