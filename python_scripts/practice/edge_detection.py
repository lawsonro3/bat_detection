# Import useful libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.destroyAllWindows()

img1 = cv2.imread('/Users/icunitz/Desktop/bat_detection/frames/clear_background/bats/close/2016-07-30_014634/frame55.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img1_gray, 90, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Set up variables for click event 1
drawLine1 = False
points_ = []

# Define mouse callback function (click event 1)
def click_event1(event, x, y, flags, param):
        global points_, drawLine1
        if event == cv2.EVENT_LBUTTONDOWN:
                drawLine1 = True
                points_.append((x, y))
        if event == cv2.EVENT_LBUTTONUP:
                if drawLine1:
                        drawLine1 = False
                        points_.append((x, y))

# Define FFT function
def takedft(img_name):
        dft = cv2.dft(np.float32(img_name), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return dft, dft_shift, magnitude_spectrum

rows, cols = img1_gray.shape
crow, ccol = int(rows/2) , int(cols/2)
thresh_contours = cv2.drawContours(thresh, contours, -1, (0, 0, 255), 3)

img1_gray_takedft = takedft(img1_gray)
magnitude_spectrum = img1_gray_takedft[2]
shift = img1_gray_takedft[1]

cv2.imshow('Grayscale Image', img1_gray)

cv2.namedWindow('magnitude_spectrum')
cv2.setMouseCallback('magnitude_spectrum', click_event1)
cv2.imshow('magnitude_spectrum', magnitude_spectrum)
cv2.waitKey(0) & 0xFF

print(points_)

cv2.imshow('Thresholded Image', thresh)
# cv2.imshow('thresh_contours', thresh_contours)

for contour in contours:
    img1_contours = cv2.drawContours(img1, contour, -1, (0, 0, 255), 3)

    cv2.imshow('img1_contours', img1_contours)

    peri = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    print (len(approx))

    cv2.waitKey(20) & 0xFF

mask = np.zeros((rows, cols, 2), np.uint8)

# Low pass filter mask
r = 70
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

fshift = shift * mask
filtered_magnitude_spectrum = np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img1_back = cv2.idft(f_ishift)
img1_back = cv2.magnitude(img1_back[:, :, 0], img1_back[:, :, 1])

plt.close()
plt.figure(1)

plt.subplot(1, 3, 1)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude_spectrum')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(filtered_magnitude_spectrum, cmap='gray')
plt.title('filtered_magnitude_spectrum')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 3)
plt.imshow(img1_back, cmap='gray')
plt.title('img1_back')
plt.xticks([])
plt.yticks([])

plt.show()