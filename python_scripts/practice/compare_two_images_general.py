# Import useful libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.measure import compare_ssim as ssim
import matplotlib as mpl
from mpl_toolkits import mplot3d

plt.close() # Close any previous matplotlib.pyplot windows

n = 20
s = n * 2 + 1 # Length of square sides

squarethickness = 3

# Set up read/write paths
readpath1 = '/Users/icunitz/Desktop/frame11.jpg'
readpath2 = '/Users/icunitz/Desktop/frame13.jpg'

# Set up window names
window1Name = 'Image 1'
window2Name = 'Image 2'

# Define FFT function
def takedft(img_name):
        dft = cv2.dft(np.float32(img_name), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return dft, dft_shift, magnitude_spectrum

# Show original frame 1
img1 = cv2.imread(readpath1)
cv2.imshow(window1Name, img1)
cv2.waitKey(0) & 0xFF

# Show grayscale frame 1
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow(window1Name, img1_gray)
cv2.waitKey(0) & 0xFF

# Show original frame 2
img2 = cv2.imread(readpath2)
cv2.imshow(window2Name, img2)
cv2.waitKey(0) & 0xFF

# Show grayscale frame 2
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow(window2Name, img2_gray)
cv2.waitKey(0) & 0xFF

cv2.destroyAllWindows()

## Read rows and columns
rows1, cols1 = img1_gray.shape
rows2, cols2 = img2_gray.shape

if rows1 == rows2 and cols1 == cols2:
    rows, cols = rows1, cols1
else:
    print('Error: images not of same size')

# Take FFTs of frames 1 and 2
img1_mag_spect = takedft(img1_gray)[2]
img2_mag_spect = takedft(img2_gray)[2]

# Calculate SSIM of images
avg_ssim, ssim_image = ssim(img1_gray, img2_gray, full=True)
ssim_value = ssim_image[int(rows/2)][int(cols/2)]

# Use template matching to find normalized cross-correlation of images
template_matched = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCORR_NORMED)
nccorr_value = np.amax(template_matched)

# Calculate SSIM of FFTs
avg_ssim_fft, ssim_image_fft = ssim(img1_mag_spect, img2_mag_spect, full=True)
ssim_value_fft = ssim_image_fft[int(rows/2)][int(cols/2)]

# Use template matching to find normalized cross-correlation of FFTs
template_matched_fft = cv2.matchTemplate(img1_mag_spect, img2_mag_spect, cv2.TM_CCORR_NORMED)
nccorr_value_fft = np.amax(template_matched_fft)

## Set up grayscale normalization conditions
Normalization1 = True
Normalization2 = True

if Normalization1:
        norm1 = mpl.colors.Normalize(vmin = 0, vmax = 255)
else:
        norm1 = None

if Normalization2:
        plottop = 14.0
        norm2 = mpl.colors.Normalize(vmin = 0, vmax = plottop)
else:
        plottop = None
        norm2 = None

## Plot results
plt.figure(1)
plt.suptitle('Comparison')

plt.subplot(3, 2, 1)
plt.cla()
plt.imshow(img1_gray, cmap='gray', norm=norm1)
plt.title('Image 1')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 2, 2)
plt.cla()
plt.imshow(img1_mag_spect, cmap='gray', norm=norm2)
plt.title('Image 1 FFT')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 2, 3)
plt.cla()
plt.imshow(img2_gray, cmap='gray', norm=norm1)
plt.title('Image 2')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 2, 4)
plt.cla()
plt.imshow(img2_mag_spect, cmap='gray', norm=norm2)
plt.title('Image 2 FFT')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 2, 5)
plt.cla()
plt.text(0.05, 0.6, 'Image nccorr = %s' % round(nccorr_value, 4))
plt.text(0.05, 0.4, 'Image SSIM = %s' % round(ssim_value, 4))
plt.xticks([])
plt.yticks([])

plt.subplot(3, 2, 6)
plt.cla()
plt.text(0.05, 0.6, 'FFT nccorr = %s' % round(nccorr_value_fft, 4))
plt.text(0.05, 0.4, 'FFT SSIM = %s' % round(ssim_value_fft, 4))
plt.xticks([])
plt.yticks([])

plt.show()
