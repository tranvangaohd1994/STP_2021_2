import cv2
import numpy as np


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            # cv2.drawContours(imgContour, cnt, -1, (255, 255, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05*peri, True)
            # cv2.drawContours(imgContour, [approx], -1, (0,255,255), 10)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest
