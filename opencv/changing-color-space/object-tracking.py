import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()

    # frame = cv.imread("sunflower.jpg")
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # blue = np.uint8([255, 0, 0])
    # hsv_blue = cv.cvtColor(blue, cv.COLOR_BGR2HSV)
    # print(hsv_blue)


    
    # define range of blue color in HSV
    lower_blue = np.array([90,100,100])
    upper_blue = np.array([110,255,255])
    # define range of yellow color in HSV
    lower_yellow = np.array([15, 0, 0])
    upper_yellow = np.array([36, 255, 255])
    # define range of green color in HSV
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([70, 255, 255])

    # Threshold the HSV image to get only one colors
    mask1 = cv.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv.inRange(hsv, lower_green, upper_green)
    mask3 = cv.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv.bitwise_or(cv.bitwise_or(mask1, mask2), mask3)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()