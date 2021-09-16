from util import getContours
import cv2
import numpy as np

image = cv2.imread("imageCMT/6.jpg")

image_resized = cv2.resize(image, (500, 500))


# Fitler with hsv

hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)


lower_red = np.array([40, 30, 1])
upper_red = np.array([100, 255, 255])


mask = cv2.inRange(hsv, lower_red, upper_red)
image_color_hsv_filter = cv2.bitwise_and(
    image_resized, image_resized, mask=mask)

_, _, image_gray = cv2.split(image_color_hsv_filter)

rect_contours = cv2.cvtColor(
    np.zeros((500, 500, 3), np.uint8), cv2.COLOR_BGR2GRAY)


# Fill rectangular contours
cnts = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(rect_contours, [c], -1, (255, 255, 255), -1)
rect_contours = cv2.erode(rect_contours, (5, 5), iterations=5)

rect_line = cv2.cvtColor(
    np.zeros((500, 500, 3), np.uint8), cv2.COLOR_BGR2GRAY)

# Draw all line black image
edges = cv2.Canny(rect_contours, 75, 200, apertureSize=3)
lines = cv2.HoughLinesP(edges, 3, np.pi/180, 5,
                        minLineLength=50, maxLineGap=8000)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(rect_line, (x1, y1), (x2, y2), 255, 2)

# Dilation image
closed_image = cv2.dilate(
    rect_line, (12, 12), iterations=5)

# Draw rectangles
biggest = getContours(closed_image)
print(biggest)
cv2.drawContours(image_resized, biggest, -1, (255, 0, 0), 20)

cv2.imshow("gray", image_gray)
cv2.imshow("rect contours", rect_contours)
cv2.imshow("rect", rect_line)
cv2.imshow("Dilation", closed_image)
cv2.imshow('image', image_resized)

cv2.waitKey()
