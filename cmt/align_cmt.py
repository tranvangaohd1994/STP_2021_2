from align_images import align_images
from text_skew import text_skew
import cv2 as cv
from matplotlib import pyplot as plt

print("[INFO] loading images...")
image = cv.imread('imageCMT/7.jpg')
template = cv.imread('template.jpg')

print("[INFO] Aligning images...")
aligned = align_images(image, template, debug=False)
# aligned = text_skew(image)

plt.figure(figsize=[20,10])
plt.subplot(121), plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original')
plt.subplot(122), plt.imshow(cv.cvtColor(aligned, cv.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Aligned')
plt.show()