import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from numpy.core.defchararray import title

img = cv.imread('morpho.png', 0)
kernel = np.ones((5,5), np.uint8)
erosion = cv.erode(img, kernel, iterations=1)
dilation = cv.dilate(img, kernel, iterations=1)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

image = [img, erosion, dilation, opening, closing, gradient, tophat, blackhat]
title = ['Original', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Tophat', 'Blackhat']

for i in range(8):
    plt.subplot(3,3,i+1),plt.imshow(image[i]),plt.title(title[i])
    plt.xticks([]), plt.yticks([])

plt.show()