import numpy as np
import cv2 as cv

img = cv.imread('morpho.png', 0)
ret, thresh = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv.moments(cnt)
print(M)

area = cv.contourArea(cnt)
perimeter = cv.arcLength(cnt, True)

epsilon = 0.1*perimeter
approx = cv.approxPolyDP(cnt, epsilon, True)

hull = cv.convexHull(cnt)

k = cv.isContourConvex(cnt)

x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img, [box], 0, (0,0,255), 2)

(x,y), radius = cv.minEnclosingCircle(cnt)
center = (int(x), (int(y)))
radius = int(radius)
cv.circle(img, center, radius, (0,0,255), 2)

ellipse = cv.fitEllipse(cnt)
cv.ellipse(img, ellipse, (255,0,0), 2)

row, col = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx)+y)
righty = int(((col-x)*vy/vx)+y)
cv.line(img, (col-1, righty), (0, lefty), (255,0,0), 2)