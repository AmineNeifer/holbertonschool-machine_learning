#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid
if __name__ == '__main__':
    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    #print(images)
    #print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    a = np.array([[[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]],
                  [[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]],
                  [[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]]])
    images_conv = convolve_grayscale_valid(images, kernel)
    plt.imshow(images[1], cmap='gray')
    plt.show()
    plt.imshow(images_conv[1], cmap='gray')
    plt.show()
