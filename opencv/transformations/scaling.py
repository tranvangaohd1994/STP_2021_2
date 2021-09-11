import cv2 as cv
import numpy as np

img = cv.imread("sunflower.jpg")
height, width = img.shape[:2]
# default INTER_LINEAR
# res = cv.resize(img, (0,0), fx = 2, fy = 2)
# res = cv.resize(img, (0,0), fx = 0.5, fy = 0.5)

# phong to
res = cv.resize(img, (2*width, 2*height), interpolation=cv.INTER_CUBIC)
# res = cv.resize(img, None, fx = 0.5, fy = 0.5, interpolation=cv.INTER_CUBIC)

# thu nho
# res = cv.resize(img, None, fx = 0.5, fy = 0.5, interpolation=cv.INTER_AREA)

cv.imshow("img", img)
cv.imshow("resize", res)
cv.waitKey(0)
cv.destroyAllWindows()