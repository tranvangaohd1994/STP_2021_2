import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('histogram.jpg',0)
equ = cv.equalizeHist(img)
res = np.hstack((img, equ))
cv.imwrite('global.png', res)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv.imwrite('clahe.png', cl1)