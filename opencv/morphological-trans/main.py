import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('i.png',0)
kernel = np.ones((5,5),np.uint8)

erosion = cv.erode(img,kernel,iterations = 1)
dilation = cv.dilate(img, kernel, iterations=1)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

kernel1 = np.ones((9, 9),np.uint8)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel1)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel1)

cross = cv.morphologyEx(img,cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS,(5,5)))

titles = ['Original','Erosion','Dilation','Opening','Closing','Gradient','Tophat','Backhat', 'Cross']
images = [img, erosion, dilation, opening, closing, gradient, tophat, blackhat, cross]

for i in range(9):
    plt.subplot(3,3,i+1),plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()