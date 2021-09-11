import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('sunflower.jpg',0)
cv.imshow('img', img)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

# equalization using numpy
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
cv.imshow('img2', img2)
hist,bins = np.histogram(img2.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


# equalization using opencv but it is not good for all
img3 = cv.imread('sunflower.jpg',0)
equ = cv.equalizeHist(img3)
res = np.hstack((img3,equ)) #stacking images side-by-side
cv.imwrite('./histograms/equalHis.jpg',res)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img3)
cv.imwrite('./histograms/Clahe.jpg',cl1)


k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
