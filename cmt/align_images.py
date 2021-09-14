import numpy as np
import cv2 as cv
import imutils

def align_images(image, template, max_feature=500, keep_percent=0.2, debug=False):
    scale = int(template.shape[1]*100/image.shape[1])
    w = int(image.shape[1]*scale/100)
    h = int(image.shape[0]*scale/100)
    image = cv.resize(image, (w,h), interpolation=cv.INTER_AREA)
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(max_feature)
    (key_point1, descs1) = orb.detectAndCompute(template_gray, None)
    (key_point2, descs2) = orb.detectAndCompute(img_gray, None)

    method = cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv.DescriptorMatcher_create(method)
    matches = matcher.match(descs1, descs2, None)

    matches = sorted(matches, key=lambda x:x.distance)

    keep = int(len(matches)*keep_percent)
    matches = matches[:keep]

    if debug:
        matched_vis = cv.drawMatches(image, key_point1, template, key_point2, matches, None)
        matched_vis = imutils.resize(matched_vis, width=1000)
        cv.imshow("Matched Keypoints", matched_vis)
        cv.waitKey(0)

    point1 = np.zeros((len(matches), 2), dtype="float")
    point2 = np.zeros((len(matches), 2), dtype="float")

    for (i,m) in enumerate(matches):
        point1[i] = key_point1[m.queryIdx].pt
        point2[i] = key_point2[m.trainIdx].pt
    
    (H, mask) = cv.findHomography(point2, point1, method=cv.RANSAC)

    (h, w) = template.shape[:2]
    aligned = cv.warpPerspective(image, H, (w, h))

    return aligned