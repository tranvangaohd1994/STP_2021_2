import numpy as np
import cv2 as cv

def text_skew(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = np.bitwise_not(image_gray)

    thresh = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w,h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    cv.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return rotated