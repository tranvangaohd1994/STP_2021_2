import cv2 as cv
import numpy as np

img = cv.imread(cv.samples.findFile('sudoku.jpg'))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLines(edges,1,np.pi/180,155)

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# cv.imwrite('./hough-line/houghlines3.jpg',img)


# 
img2 = cv.imread(cv.samples.findFile('sudoku.jpg'))
gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
lines = cv.HoughLinesP(edges,1,np.pi/180,80,minLineLength=50,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img2,(x1,y1),(x2,y2),(0,255,0),2)

cv.imshow('image', img2)
cv.waitKey(0)
cv.destroyAllWindows()