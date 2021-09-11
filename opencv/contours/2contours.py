import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('world.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale
blur = cv.blur(gray, (3, 3)) # blur the image
ret, thresh = cv.threshold(blur, 50, 255, cv.THRESH_BINARY)

# Finding contours for the thresholded image
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# create hull array for convex hull points
hull = []
# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv.convexHull(contours[i], False))
# create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 0, 255) # red - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    cv.imshow("drawing", drawing)
    # draw ith convex hull object
    cv.drawContours(drawing, hull, i, color, 1, 8)
    cv.imshow("drawing", drawing)
cv.imshow("drawing", drawing)


k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()