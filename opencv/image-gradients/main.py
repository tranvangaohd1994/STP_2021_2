import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('sudoku.jpg',0)

laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

img = cv.imread('solid.png', 0)
# Output dtype = cv.CV_8U
sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)

# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

image = [img, laplacian, sobelx, sobely, sobelx8u, sobel_8u]
title = ['Original', 'Laplacian', 'Sobelx', 'Sobely', 'Sobelx8u', 'Sobel_8u(64f)']

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(image[i], cmap = 'gray')
    plt.title(title[i]), plt.xticks([]), plt.yticks([])
plt.show()