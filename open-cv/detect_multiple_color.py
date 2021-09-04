import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while 1:
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    lower_green = np.array([50,100,100])
    upper_green = np.array([70,255,255])

    lower_red = np.array([-10,100,100])
    upper_red = np.array([10,255,255])

    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv.inRange(hsv, lower_green, upper_green)
    red_mask = cv.inRange(hsv, lower_red, upper_red)

    kernal = np.ones((8,8), "uint8")
    # for red
    red_mask = cv.dilate(red_mask, kernal)
    res_red = cv.bitwise_and(frame, frame, mask=red_mask)

    # for green
    green_mask = cv.dilate(green_mask, kernal)
    res_green = cv.bitwise_and(frame, frame, mask=green_mask)

    # for blue
    blue_mask = cv.dilate(blue_mask, kernal)
    res_blue = cv.bitwise_and(frame, frame, mask=blue_mask)

    # Detect red colour
    contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 300):
            x,y,w,h = cv.boundingRect(contour)
            frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            cv.putText(frame, "Red", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0,255))

    # Detect green colour
    contours, hierarchy = cv.findContours(green_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 300):
            x,y,w,h = cv.boundingRect(contour)
            frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv.putText(frame, "Green", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

    # Detect red colour
    contours, hierarchy = cv.findContours(blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 300):
            x,y,w,h = cv.boundingRect(contour)
            frame = cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv.putText(frame, "Blue", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))

    cv.imshow('frame', frame)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()