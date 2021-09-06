import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('water_coins.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv.dilate(opening, kernel, iterations=3)

dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = np.subtract(sure_bg, sure_fg)

ret, markers = cv.connectedComponents(sure_fg)
markers += 1
markers[unknown==255] = 0

markers = cv.watershed(img, markers)
img[markers==-1] = [255,0,0]

plt.subplot(1,1,1), plt.imshow(img), plt.title('Final')
plt.xticks([]), plt.yticks([])
plt.show()