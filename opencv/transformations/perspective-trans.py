import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('sudoku.jpg')
rows,cols,ch = img.shape
pts1 = np.float32([[40,100],[280,50],[120,330],[400,250]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
for (x, y) in pts1:
    plt.scatter(x, y, s = 50, c = "red", marker = "x")

plt.subplot(122),plt.imshow(dst),plt.title('Output')
for (x, y) in pts2:
    plt.scatter(x, y, s = 50, c = "red", marker = "x")
plt.show()