import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('android-studio.jpeg',0)
hist = cv.calcHist([img], [0], None, [256], [0,256])
# Numpy way

hist, bins = np.histogram(img.ravel(), 256, [0,256])
# hist = np.bincount(img.ravel(), minlength=256)

# Grayscale image
# plt.hist(img.ravel(), 256, [0,256])
# plt.show()

color_img = cv.imread('android-studio.jpeg')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([color_img], [i], None, [256], [0,256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])
plt.show()