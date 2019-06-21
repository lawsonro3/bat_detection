# Import useful libraries
import cv2
import numpy as np
from scipy import signal
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
import os # Not currently using - figure out how to effectively use os.path.join

plt.close() # Close any previous matplotlib.pyplot windows

# Set up read paths
file = input('Video Name: ') # Try 2016-07-30_014634
frameno_withbat = input('Frame Number With Bat: ') # Try 65
frameno_wobat = input('Frame Number Without Bat: ') # Try 60
frameextension = '.jpg'
readpath_withbat = '/Users/icunitz/Desktop/bat_detection/frames/' + file + '/frame' + frameno_withbat + frameextension # Customize this based on directories in computer
readpath_wobat = '/Users/icunitz/Desktop/bat_detection/frames/' + file + '/frame' + frameno_wobat + frameextension

# Set window names
frameTitle_withbat = 'Video ' + file + ', Frame ' + frameno_withbat
window1Name = frameTitle_withbat
frameTitle_wobat = 'Video ' + file + ', Frame ' + frameno_wobat
window2Name = frameTitle_wobat

n = 20
s = n * 2 + 1 # Length of square sides

ref_location = [] # Empty list to hold click locations

# Read original image and create grayscale copy
img_withbat = cv2.imread(readpath_withbat)
clone_withbat = cv2.cvtColor(img_withbat, cv2.COLOR_BGR2GRAY)

# Click event
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_location.append((x, y))
        ## Later - get squares to appear as you click?
        #cv2.rectangle(img_withbat, (x - n, y - n), (x + n, y + n), (0, 0, 0), 2)

# Set up original image window and callback function
cv2.namedWindow(window1Name)
cv2.setMouseCallback(window1Name, click_event)

print('\nClick in the middle of the bat and press any key to progress. The region of interest coordinates will be saved as your last click.\n')

# Show original image
cv2.imshow(window1Name, img_withbat)
cv2.waitKey(0) & 0xFF

# Coordinates of last clicked region
roi_x = ref_location[-1][0]
roi_y = ref_location[-1][1]

print ('Location of Interest: (' + str(roi_x) + ', ' + str(roi_y) + ')')

print('\nPress any keys to progress.\n')

# Convert original image to grayscale to show later
img_withbat = cv2.cvtColor(img_withbat, cv2.COLOR_BGR2GRAY)

# Draw square
cv2.rectangle(img_withbat, (roi_x - n, roi_y - n), (roi_x + n, roi_y + n), (0, 0, 0), 2)

# Show grayscale image with square
cv2.imshow(window1Name, img_withbat)
cv2.waitKey(0) & 0xFF

# Crop image around last clicked location
roi_withbat = clone_withbat[(roi_y - n):(roi_y + n + 1), (roi_x - n):(roi_x + n + 1)]

# Read original image w/o bat and create grayscale copy
img_wobat = cv2.imread(readpath_wobat)
clone_wobat = cv2.cvtColor(img_wobat, cv2.COLOR_BGR2GRAY)

# Convert original image w/o bat to grayscale to show later
img_wobat = cv2.cvtColor(img_wobat, cv2.COLOR_BGR2GRAY)

# Draw square around region of interest in image w/o bat
cv2.rectangle(img_wobat, (roi_x - n, roi_y - n), (roi_x + n, roi_y + n), (0, 0, 0), 2)

# Show grayscale image w/o bat with square
cv2.imshow(window2Name, img_wobat)
cv2.waitKey(0) & 0xFF

# Crop image w/o bat around same location
roi_wobat = clone_wobat[(roi_y - n):(roi_y + n + 1), (roi_x - n):(roi_x + n + 1)]

cv2.destroyAllWindows()

# Define FFT function
def takedft(img_name):
        dft = cv2.dft(np.float32(img_name), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return dft, dft_shift, magnitude_spectrum

# Take FFT and extract magnitude spectrum of cropped images
takedft_roi_withbat = takedft(roi_withbat)
mag_spect_roi_withbat = takedft_roi_withbat[2]
takedft_roi_wobat = takedft(roi_wobat)
mag_spect_roi_wobat = takedft_roi_wobat[2]

# Correlate FFTs
correlation = signal.correlate2d(mag_spect_roi_withbat, mag_spect_roi_wobat, fillvalue=0)

# Calculate SSIM of FFTs
avg_ssim, ssim_image = ssim(mag_spect_roi_withbat, mag_spect_roi_wobat, full=True)

# Image similarity results (at center, where FFTs are aligned)
ssim_value = ssim_image[n][n]
correlation_value = correlation[2*n][2*n]

## Show results

titlefontsize = 12
subtitlefontsize = 10
figrows = 3
figcolumns = 4

# Set up grayscale normalization condition
Normalization = True

if Normalization:
        norm = mpl.colors.Normalize(vmin = 0, vmax = 255)
else:
        norm = None

plt.figure(1, figsize=(figcolumns*3, figrows*3))
plt.suptitle(file + ', Same Location Comparison; ROI Center: (%s, %s)' % (roi_x, roi_y), fontsize = titlefontsize)

plt.subplot(figrows, figcolumns, 1)
plt.cla()
plt.imshow(img_withbat, cmap='gray', norm=norm)
plt.title(frameTitle_withbat, fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 2)
plt.cla()
plt.imshow(roi_withbat, cmap='gray', norm=norm)
plt.title('ROI w/ Bat', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 3)
plt.cla()
plt.imshow(mag_spect_roi_withbat, cmap='gray', norm=norm)
plt.title('FFT of ROI w/ Bat', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax1 = plt.subplot(figrows, figcolumns, 4, projection='3d')
plt.cla()
X1, Y1 = np.meshgrid(range(s), range(s))
Z1 = mag_spect_roi_withbat
mplot3d.Axes3D.plot_surface(ax1, X1, Y1, Z1, cmap='gray', norm=norm)
plt.title('FFT of ROI w/ Bat, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax1, bottom=0.0, top=255.0)
mplot3d.Axes3D.set_zticks(ax1, [])
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 5)
plt.cla()
plt.imshow(img_wobat, cmap='gray', norm=norm)
plt.title(frameTitle_wobat, fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 6)
plt.cla()
plt.imshow(roi_wobat, cmap='gray', norm=norm)
plt.title('ROI w/o Bat', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 7)
plt.cla()
plt.imshow(mag_spect_roi_wobat, cmap='gray', norm=norm)
plt.title('FFT of ROI w/o Bat', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax2 = plt.subplot(figrows, figcolumns, 8, projection='3d')
plt.cla()
X2, Y2 = np.meshgrid(range(s), range(s))
Z2 = mag_spect_roi_wobat
mplot3d.Axes3D.plot_surface(ax2, X2, Y2, Z2, cmap='gray', norm=norm)
plt.title('FFT of ROI w/o Bat, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax2, bottom=0.0, top=255.0)
mplot3d.Axes3D.set_zticks(ax2, [])
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 9)
plt.cla()
plt.imshow(correlation, cmap='gray')
plt.title('Correlation Image of FFTs', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 10)
plt.cla()
plt.text(0.05, 0.5, 'Correlation = %s' % round(correlation_value, 4))
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 11)
plt.cla()
plt.imshow(ssim_image, cmap='gray')
plt.title('SSIM Image of FFTs', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 12)
plt.cla()
plt.text(0.25, 0.5, 'SSIM = %s' % round(ssim_value, 4))
plt.xticks([])
plt.yticks([])

plt.show()

print ("Done")