import cv2 as cv
import numpy as np

img = cv.imread("sunflower.jpg")
rows, cols = img.shape[:2]

center = ((cols-1)/2.0 , (rows-1)/2.0)
# center = (0,0)
angle = -60
scale = 1.5
M = cv.getRotationMatrix2D(center, angle, scale)
dst = cv.warpAffine(img, M, (cols, rows))
cv.imshow('img', img)
cv.imshow('rotate', dst)
cv.waitKey(0)
cv.destroyAllWindows()