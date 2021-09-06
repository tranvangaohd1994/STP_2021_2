import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('momo.jpg', 0)
canny = cv.Canny(img, 100, 100)

images = [img, canny]
titles = ['Original', 'Canny Edge']

for i in range(2):
    plt.subplot(2,2,i+1), plt.imshow(images[i]), plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()