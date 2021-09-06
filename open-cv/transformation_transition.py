import cv2 as cv
import numpy as np

img = cv.imread('momo.jpg', 0)
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
transition = cv.warpAffine(img, M, (cols, rows))

M = cv.getRotationMatrix2D(((cols-1)/2, (rows-1)/2), 90, 1)
rotation = cv.warpAffine(img, M,(cols, rows))

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
aff_transform = cv.warpAffine(img, M, (cols, rows))

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
per_transform = cv.warpPerspective(img,M,(30,30))

cv.imshow('image', img)
cv.imshow('result', res)
cv.imshow('transition', transition)
cv.imshow('rotation', rotation)
cv.imshow('Affine transform', aff_transform)
cv.imshow('Perspective transform', per_transform)
cv.waitKey(0)
cv.destroyAllWindows()