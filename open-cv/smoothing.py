import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('momo.jpg')

kernel = np.ones((5,5),np.float32)/25
averaging = cv.filter2D(img,-1,kernel)

blur = cv.blur(img, (5,5))
gaussian_blur = cv.GaussianBlur(img, (5,5), 0)
median = cv.medianBlur(img, 5)
bilateral_blur = cv.bilateralFilter(img, 9 ,75, 75)

image = [img, averaging, blur, gaussian_blur, median, bilateral_blur]
title = ['Original', 'Averaging', 'Blur', 'Gaussian', 'Median', 'Bilateral']

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(image[i]),plt.title(title[i])
    plt.xticks([]), plt.yticks([])

plt.show()
