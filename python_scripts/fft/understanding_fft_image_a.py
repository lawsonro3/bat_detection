import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

file = '2016-07-30_014634'
frameno = '50'
extension = '.jpg'
readpath = '/Users/icunitz/Desktop/bat_detection/frames/' + file + '/frame' + frameno + extension

img = cv2.imread(readpath, 0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))

dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

rows, cols = img.shape
# crow, ccol = int(rows/2) , int(cols/2.66666666666666667)
crow, ccol = int(rows/2) , int(cols/2)


# Circular HPF mask, center circle is 0, remaining all ones

# mask = np.ones((rows, cols, 2), np.uint8)
# r = 2000
# center = [crow, ccol]
# x, y = np.ogrid[:rows, :cols]
# mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
# mask[mask_area] = 1


### Band pass filter
# mask = np.zeros((rows, cols, 2), np.uint8)
# r_out = 150
# r_in = 100
# center = [crow, ccol]
# x, y = np.ogrid[:rows, :cols]
# mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
#                            ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
# mask[mask_area] = 1

## LOw pass filter
mask = np.zeros((rows, cols, 2), np.uint8)
r = 70
n = 60
center = [crow, ccol]
print(center)
x, y = np.ogrid[:rows, :cols]
mask_area = [crow+n, ccol+n]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

## Different filter
# rows, cols = img_gray.shape
#
# crow, ccol = int(rows/2) , int(cols/2)     # center
#
#
# n = 10
# # create a mask first, center square is 1, remaining all zeros
# mask = np.zeros((rows, cols, 2), np.uint8)
# mask[crow-n:crow+n, ccol-n:ccol+n] = 0

### HIgh pass filter
# # Circular HPF mask, center circle is 0, remaining all ones
# rows, cols = img.shape
# crow, ccol = int(rows / 2), int(cols / 2)
#
# mask = np.ones((rows, cols, 2), np.uint8)
# r = 70
# center = [crow, ccol]
# x, y = np.ogrid[:rows, :cols]
# mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
# mask[mask_area] = 0

# apply mask and inverse DFT
fshift = dft_shift * mask
fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


# sub_image = img - img_back
sub_image = (2*img_back)
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 2)
plt.imshow(img_back, cmap='gray')
plt.title('Invers')
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 3)
plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask')
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 4)
plt.imshow(sub_image, cmap='gray')
plt.title('Image subtraction')
plt.xticks([])
plt.yticks([])
plt.show()
