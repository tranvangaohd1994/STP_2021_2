import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread('sunflower.jpg')
imgb = cv.imread('sunflower.jpg', 0)
hist = cv.calcHist([imgb],[0],None,[256],[0,256])
# calculation in numpy
# hist,bins = np.histogram(img.ravel(),256,[0,256])

# using matplotlib
plt.hist(imgb.ravel(),256,[0,256]); plt.show()
# using matplotlib with rgb
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

# using OpenCV
# create a mask
mask = np.zeros(imgb.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(imgb,imgb,mask = mask)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv.calcHist([imgb],[0],None,[256],[0,256])
hist_mask = cv.calcHist([imgb],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(imgb, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()