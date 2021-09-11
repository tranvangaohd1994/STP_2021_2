import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("sunflower.jpg")
rows, cols, ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
for (x, y) in pts1:
  plt.scatter(x, y, s=50, c='black', marker='x')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
for (x, y) in pts2:
  plt.scatter(x, y, s=50, c='black', marker='x')

plt.show()