import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

A = cv.imread('apple.jpg')[...,::-1]
B = cv.imread('orange.jpg')[...,::-1]
A = cv.resize(A, (400,400))
B = cv.resize(B, (400,400))

#to attach to images side by side
A_B = np.hstack((B[:,:200],A[:,210:]))
#now we have to blur in the middle so that the image looks like one image
#copy image1 in a new variable
copy_A = A.copy()
gp_A = [copy_A]
#making a list for image 1 and its first element is image itself
#G pyramid for A image
for i in range(6):
    copy_A = cv.pyrDown(copy_A)
    gp_A.append(copy_A)
#appending the pyr down image to the list of first image
#copy image2 in a new variable
copy_B = B.copy()
gp_B = [copy_B]
#making a list for image 2 and its first element is image itself
#G pyramid for B image
for i in range(6):
    copy_B = cv.pyrDown(copy_B)
    gp_B.append(copy_B)
    #appending the pyr down image to the list of second image
copy_A=gp_A[5]
#assigning the first image list last element to copied variable
lp_A=[copy_A]
#again making alist for image one whose first element is the last element of previous list
#L pyramid for A image
for i in range(5,0,-1):
    size = (gp_A[i-1].shape[1],gp_A[i-1].shape[0])
    G = cv.pyrUp(gp_A[i],dstsize=size)
    L_A = cv.subtract(gp_A[i-1],G)
    lp_A.append(L_A)
#same process with B image
copy_B=gp_B[5]
#assigning the first image list last element to copied variable
lp_B=[copy_B]
#again making alist for image one whose first element is the last element of previous list
#L pyramid for A image
for i in range(5,0,-1):
    size = (gp_B[i-1].shape[1],gp_B[i-1].shape[0])
    G = cv.pyrUp(gp_B[i],dstsize=size)
    L_B = cv.subtract(gp_B[i-1],G)
    lp_B.append(L_B)
#now add left and right halves of the images in each level of pyramid
A_B_pyramid=[]
#an empty list
for A_lap,B_lap in zip(lp_A,lp_B):
    cols,rows,ch = A_lap.shape
    L = np.hstack((B_lap[:,0:int(cols/2)],A_lap[:,int(cols/2):]))
    A_B_pyramid.append(L)
#now reconstruct
A_B_reconstruct = A_B_pyramid[0]
for i in range(1,6):
    size = (A_B_pyramid[i].shape[1],A_B_pyramid[i].shape[0])
    A_B_reconstruct = cv.pyrUp(A_B_reconstruct,dstsize=size)
    A_B_reconstruct = cv.add(A_B_pyramid[i],A_B_reconstruct)

image = [A, B, A_B, A_B_reconstruct]
title = ['A', 'B', 'A_B', 'A_B_reconstruct']
for i in range(4):
    plt.subplot(2, 2, i+1),plt.imshow(image[i])
    plt.title(title[i]),plt.xticks([]), plt.yticks([])
plt.show()
