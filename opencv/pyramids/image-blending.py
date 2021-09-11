import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

A = cv.imread('apple.jpg')[...,::-1]
B = cv.imread('orange.jpg')[...,::-1]

A=cv.resize(A, (400,400))
B=cv.resize(B, (400,400))
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)
# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)
# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    size = (gpA[i-1].shape[1],gpA[i-1].shape[0])
    GE = cv.pyrUp(gpA[i], dstsize=size)
    L = cv.subtract(gpA[i-1],GE)
    lpA.append(L)
# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    size = (gpB[i-1].shape[1],gpB[i-1].shape[0])
    GE = cv.pyrUp(gpB[i], dstsize=size)
    L = cv.subtract(gpB[i-1],GE)
    lpB.append(L)
# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):]))
    LS.append(ls)
# now reconstruct
ls_ = LS[0]
for i in range(1,6):
    size = (LS[i].shape[1],LS[i].shape[0])
    ls_ = cv.pyrUp(ls_, dstsize=size)
    ls_ = cv.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):]))

image = [A, B, real, ls_]
title = ['A', 'B', 'A_B', 'A_B_reconstruct']
for i in range(4):
    plt.subplot(2, 2, i+1),plt.imshow(image[i])
    plt.title(title[i]),plt.xticks([]), plt.yticks([])
plt.show()