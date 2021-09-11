import numpy as np
import cv2 as cv

img = cv.imread("sunflower.jpg", 0)
rows,cols = img.shape
# M = np.float32([[1,0,100],[0,1,50]])
M = np.float32([[1, 0, -100], [0, 1, -50]])

# tx, ty = (100, -150)
# M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img', img)
cv.imshow('trans',dst)
cv.waitKey(0)
cv.destroyAllWindows()