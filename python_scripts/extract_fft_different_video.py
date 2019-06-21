# Import useful libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.measure import compare_ssim as ssim
import matplotlib as mpl
from mpl_toolkits import mplot3d
import os # Not currently using - figure out how to effectively use os.path.join

plt.close() # Close any previous matplotlib.pyplot windows

# Set up read paths
file1 = input('Video 1 Name: ') # Try 2016-07-30_014634
frame1 = input('Video 1 Frame Number: ')
file2 = input('Video 2 Name: ') # Try something else
frame2 = input('Video 2 Frame Number: ')
frameextension = '.jpg'
readpath1 = '/Users/icunitz/Desktop/bat_detection/frames/' + file1 + '/frame' + frame1 + frameextension # Customize this based on directories in computer
readpath2 = '/Users/icunitz/Desktop/bat_detection/frames/' + file2 + '/frame' + frame2 + frameextension

# Set window names
frameTitle1 = 'Video ' + file1 + ', Frame ' + frame1
window1Name = frameTitle1
frameTitle2 = 'Video ' + file2 + ', Frame ' + frame2
window2Name = frameTitle2

n = 20
s = n * 2 + 1 # Length of square sides

ref_location1 = [] # Empty list to hold click locations on Frame 1

# Read original frame 1 and create grayscale copy
img1 = cv2.imread(readpath1)
clone1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Click event 1
def click_event1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_location1.append((x, y))

# Set up original frame 1 window and callback function
cv2.namedWindow(window1Name)
cv2.setMouseCallback(window1Name, click_event1)

print('\nClick in the middle of the bat and press any key to progress. The region of interest coordinates will be saved as your last click.\n')

# Show original frame 1
cv2.imshow(window1Name, img1)
cv2.waitKey(0) & 0xFF

# Coordinates of last clicked region
roi1_x = ref_location1[-1][0]
roi1_y = ref_location1[-1][1]

print ('Location of Interest, Video %s, Frame %s: (%s, %s)' % (file1, frame1, roi1_x, roi1_y))

print('\nPress any keys to progress.\n')

# Convert original frame 1 to grayscale to show later
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Draw square
cv2.rectangle(img1, (roi1_x - n, roi1_y - n), (roi1_x + n, roi1_y + n), (0, 0, 0), 2)

# Show grayscale frame 1 with square
cv2.imshow(window1Name, img1)
cv2.waitKey(0) & 0xFF

# Crop frame 1 around last clicked location
roi1 = clone1[(roi1_y - n):(roi1_y + n + 1), (roi1_x - n):(roi1_x + n + 1)]

ref_location2 = [] # Empty list to hold click locations on Frame 2

# Read original frame 2 and create grayscale copy
img2 = cv2.imread(readpath2)
clone2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.namedWindow(window2Name)

# Click event 2 - fix later
def click_event2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_location2.append((x, y))

# Set up original image window and callback function
cv2.setMouseCallback(window2Name, click_event2)

print('\nClick in the middle of the bat and press any key to progress. The region of interest coordinates will be saved as your last click.\n')

# Show original image
cv2.imshow(window2Name, img2)
cv2.waitKey(0) & 0xFF

# Coordinates of last clicked region
roi2_x = ref_location2[-1][0]
roi2_y = ref_location2[-1][1]

print ('Location of Interest, Video %s, Frame %s: (%s, %s)' % (file1, frame2, roi2_x, roi2_y))

print('\nPress any keys to progress.\n')

# Convert original frame 2 to grayscale to show later
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Draw square around region of interest in frame 2
cv2.rectangle(img2, (roi2_x - n, roi2_y - n), (roi2_x + n, roi2_y + n), (0, 0, 0), 2)

# Show grayscale frame 2 with square
cv2.imshow(window2Name, img2)
cv2.waitKey(0) & 0xFF

# Crop frame 2 around same location
roi2 = clone2[(roi2_y - n):(roi2_y + n + 1), (roi2_x - n):(roi2_x + n + 1)]

cv2.destroyAllWindows()

# Define FFT function
def takedft(img_name):
        dft = cv2.dft(np.float32(img_name), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return dft, dft_shift, magnitude_spectrum

# Take FFT and extract magnitude spectrum of cropped images
takedft_roi1 = takedft(roi1)
mag_spect_roi1 = takedft_roi1[2]
takedft_roi2 = takedft(roi2)
mag_spect_roi2 = takedft_roi2[2]

# Correlate FFTs
correlation = signal.correlate2d(mag_spect_roi1, mag_spect_roi2, fillvalue=0)

# Calculate SSIM of FFTs
avg_ssim, ssim_image = ssim(mag_spect_roi1, mag_spect_roi2, full=True)

# Image similarity results (at center, where FFTs are aligned)
ssim_value = ssim_image[n][n]
correlation_value = correlation[2*n][2*n]

## Show results

titlefontsize = 12
subtitlefontsize = 10
figrows = 3
figcolumns = 4

# Set up grayscale normalization condition
Normalization = False

if Normalization:
        norm = mpl.colors.Normalize(vmin = 0, vmax = 255)
else:
        norm = None

plt.figure(1, figsize=(figcolumns*3, figrows*3))
plt.suptitle('%s vs. %s Comparison' % (file1, file2), fontsize = titlefontsize)

plt.subplot(figrows, figcolumns, 1)
plt.cla()
plt.imshow(img1, cmap='gray', norm=norm)
plt.title(frameTitle1, fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 2)
plt.cla()
plt.imshow(roi1, cmap='gray', norm=norm)
plt.title('ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 3)
plt.cla()
plt.imshow(mag_spect_roi1, cmap='gray', norm=norm)
plt.title('FFT of ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax1 = plt.subplot(figrows, figcolumns, 4, projection='3d')
plt.cla()
X1, Y1 = np.meshgrid(range(s), range(s))
Z1 = mag_spect_roi1
mplot3d.Axes3D.plot_surface(ax1, X1, Y1, Z1, cmap='gray', norm=norm)
plt.title('FFT of ROI, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax1, bottom=0.0, top=200.0)
mplot3d.Axes3D.set_zticks(ax1, [])
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 5)
plt.cla()
plt.imshow(img2, cmap='gray', norm=norm)
plt.title(frameTitle2, fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 6)
plt.cla()
plt.imshow(roi2, cmap='gray', norm=norm)
plt.title('ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 7)
plt.cla()
plt.imshow(mag_spect_roi2, cmap='gray', norm=norm)
plt.title('FFT of ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax2 = plt.subplot(figrows, figcolumns, 8, projection='3d')
plt.cla()
X2, Y2 = np.meshgrid(range(s), range(s))
Z2 = mag_spect_roi2
mplot3d.Axes3D.plot_surface(ax2, X2, Y2, Z2, cmap='gray', norm=norm)
plt.title('FFT of ROI, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax2, bottom=0.0, top=200.0)
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