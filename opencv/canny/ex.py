import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def nothing(x):
    pass

img = cv2.imread('noisy_leaf.jpg', 0) #read image as grayscale


canny = cv2.Canny(img, 85, 255) 

cv2.namedWindow('image') # make a window with name 'image'
cv2.createTrackbar('L', 'image', 0, 255, nothing) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 255, nothing) #upper threshold trackbar for window 'image

while(1):
    numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')

    canny = cv2.Canny(img, l, u)

cv2.destroyAllWindows()