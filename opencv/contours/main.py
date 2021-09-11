import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('contour.png')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv.Canny(imgray, 127, 255)

ret, thresh = cv.threshold(imgray, 127, 255, 0)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print( contours)
print (hierarchy)
cv.drawContours(img, contours, -1, (0,255,0), 3)
# cv.drawContours(img, contours, 0, (0,255,0), 3)
# cnt = contours[1]
# cv.drawContours(img, [cnt], 0, (0,255,0), 3)

cnt = contours[0]
M = cv.moments(cnt)
print( M )

area = cv.contourArea(cnt)
print("S = ", area)

perimeter = cv.arcLength(cnt,True)
print("P = ", perimeter)

epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)
# cv.drawContours(img, [approx], -1, (0,255,0), 3)

hull = cv.convexHull(cnt)
# cv.drawContours(img, [hull], -1, (0,255,0), 3)

k = cv.isContourConvex(cnt)
print("k = ", k)

x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,(0,0,255),2)

(x,y),radius = cv.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv.circle(img,center,radius,(0,0,255),2)

ellipse = cv.fitEllipse(cnt)
cv.ellipse(img,ellipse,(0,255,255),2)

rows,cols = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img,(cols-1,righty),(0,lefty),(255,255,0),2)

image = [img, thresh, edged]
title = ['A', 'B', 'C']
for i in range(3):
    plt.subplot(2, 2, i+1),plt.imshow(image[i], cmap='gray')
    plt.title(title[i]),
    # plt.xticks([]), plt.yticks([])
plt.show()