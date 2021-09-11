import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread('sunflower.jpg')
lower = cv.pyrDown(img)
lower_lower = cv.pyrDown(lower)

higher_img = cv.pyrUp(img)
higher_higher = cv.pyrUp(lower)
higher_higher_reso = cv.pyrUp(lower_lower)

image = [img, lower, lower_lower, higher_img, higher_higher, higher_higher_reso]
title = ['img', 'lower', 'lower_lower', 'higher_img', 'higher_higher', 'higher_higher_reso']
for i in range(6):
    plt.subplot(2, 3, i+1),plt.imshow(image[i])
    plt.title(title[i]),plt.xticks([]), plt.yticks([])
plt.show()

