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
readpath_withbat = '/Users/icunitz/Desktop/bat_detection/frames/clear_background/bats/close/' + file + '/frame' + frameno_withbat + frameextension # Customize this based on directories in computer
readpath_wobat = '/Users/icunitz/Desktop/bat_detection/frames/clear_background/bats/close/' + file + '/frame' + frameno_wobat + frameextension

n = 20
s = n * 2 + 1 # Length of square sides

# Set window names
frameTitle_withbat = 'Video ' + file + ', Frame ' + frameno_withbat
window1Name = frameTitle_withbat
window2Name = frameTitle_withbat + ', Rotated'
frameTitle_wobat = 'Video ' + file + ', Frame ' + frameno_wobat
window3Name = frameTitle_wobat + ', Rotated'

# Set up variables for click event 1
drawLine = False
xi, yi = 0, 0
xf, yf = 0, 0

ref_location = [] # Empty list to hold click event 2 locations

# Click event 1 - to find rotation angle
def click_event1(event, x, y, flags, param):
        global xi, yi, xf, yf, drawLine
        if event == cv2.EVENT_LBUTTONDOWN:
                drawLine = True
                xi, yi = x, y
        if event == cv2.EVENT_LBUTTONUP:
                if drawLine:
                        drawLine = False
                        xf, yf = x, y

# Click event 2 - to find region of interest
def click_event2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_location.append((x, y))

# Define angle finding function
def find_angle(xi, yi, xf, yf):
        if xi == xf:
                angle_deg = 90.0
        else:
                det = (yf - yi) / (xf - xi)
                angle_rad = np.arctan(det)
                angle_deg = angle_rad * (180 / np.pi)
        return angle_deg

# Define image rotating function
def rotate(image_name, a_found):
        global yi, yf
        
        Acute = True
        sign = 1.0

        rows, cols = image_name.shape[:2]
        
        a_rad = a_found * (np.pi / 180)
        rot_angle_rad = np.pi/2 + a_rad

        if yi < yf:
                Acute = False

        if a_found < 0:
                if not Acute:
                        sign = -1.0
                        rot_angle_rad = np.pi/2 - a_rad
        else:
                if Acute:
                        sign = -1.0
                        rot_angle_rad = np.pi/2 - a_rad
        
        if Acute:
                r = int(rows*np.cos(rot_angle_rad) + cols*np.sin(rot_angle_rad))
                c = int(cols*np.cos(rot_angle_rad) + rows*np.sin(rot_angle_rad))
        else:
                r = int(cols*np.cos(rot_angle_rad - np.pi/2) + rows*np.sin(rot_angle_rad - np.pi/2))
                c = int(rows*np.cos(rot_angle_rad - np.pi/2) + cols*np.sin(rot_angle_rad - np.pi/2))
        
        rot_angle_deg = rot_angle_rad * (180 / np.pi)

        M = cv2.getRotationMatrix2D((cols//2, rows//2), sign * rot_angle_deg, 1)
        M[0,2] += (c - cols) / 2
        M[1,2] += (r - rows) / 2
        
        return cv2.warpAffine(image_name, M, (c, r))

# Define FFT function
def takedft(img_name):
        dft = cv2.dft(np.float32(img_name), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return dft, dft_shift, magnitude_spectrum

# Show original image
img_withbat = cv2.imread(readpath_withbat)
cv2.namedWindow(window1Name)
cv2.setMouseCallback(window1Name, click_event1)
cv2.imshow(window1Name, img_withbat)
cv2.waitKey(0) & 0xFF

# Find rotation angle from clicked points
found_angle = find_angle(xi, yi, xf, yf)

# print('\nClick in the middle of the bat and press any key to progress. The region of interest coordinates will be saved as your last click.\n')

# Rotate image and show rotated version
img_withbat_rotated = rotate(img_withbat, found_angle)
cv2.namedWindow(window2Name)
cv2.setMouseCallback(window2Name, click_event2)
cv2.imshow(window2Name, img_withbat_rotated)
cv2.waitKey(0) & 0xFF

# Coordinates of region of interest
roi_x = ref_location[-1][0]
roi_y = ref_location[-1][1]

# print ('\nLocation of Interest: (%s, %s)' % (roi_x, roi_y))

# print('\nPress any keys to progress.\n')

# Convert rotated image to grayscale and clone
clone_img_withbat_rotated_gray = cv2.cvtColor(img_withbat_rotated, cv2.COLOR_BGR2GRAY)
img_withbat_rotated_gray = cv2.cvtColor(img_withbat_rotated, cv2.COLOR_BGR2GRAY)

# Show grayscale rotated image with square
cv2.rectangle(img_withbat_rotated_gray, (roi_x - n, roi_y - n), (roi_x + n, roi_y + n), (0, 0, 0), 2)
cv2.imshow(window2Name, img_withbat_rotated_gray)
cv2.waitKey(0) & 0xFF

# Crop image around last clicked location
roi_withbat = clone_img_withbat_rotated_gray[(roi_y - n):(roi_y + n + 1), (roi_x - n):(roi_x + n + 1)]

# Read in original image w/o bat
img_wobat = cv2.imread(readpath_wobat)

# Rotate image, convert to grayscale, and clone
img_wobat_rotated = rotate(img_wobat, found_angle)
clone_img_wobat_rotated = cv2.cvtColor(img_wobat_rotated, cv2.COLOR_BGR2GRAY)
img_wobat_rotated_gray = cv2.cvtColor(img_wobat_rotated, cv2.COLOR_BGR2GRAY)

# Show grayscale rotated image w/o bat with square
cv2.rectangle(img_wobat_rotated_gray, (roi_x - n, roi_y - n), (roi_x + n, roi_y + n), (0, 0, 0), 2)
cv2.imshow(window3Name, img_wobat_rotated_gray)
cv2.waitKey(0) & 0xFF

# Crop image w/o bat around same location
roi_wobat = clone_img_wobat_rotated[(roi_y - n):(roi_y + n + 1), (roi_x - n):(roi_x + n + 1)]

cv2.destroyAllWindows()

# Take FFT and extract magnitude spectrum of cropped images
takedft_roi_withbat = takedft(roi_withbat)
mag_spect_roi_withbat = takedft_roi_withbat[2]
takedft_roi_wobat = takedft(roi_wobat)
mag_spect_roi_wobat = takedft_roi_wobat[2]

# Correlate FFTs
correlation = signal.correlate2d(mag_spect_roi_withbat, mag_spect_roi_wobat, fillvalue=0)

#Correlate FFTs to themselves and extract value results
auto1 = signal.correlate2d(mag_spect_roi_withbat, mag_spect_roi_withbat, fillvalue=0)
auto2 = signal.correlate2d(mag_spect_roi_wobat, mag_spect_roi_wobat, fillvalue=0)

auto1_value = auto1[2*n][2*n]
auto2_value = auto2[2*n][2*n]

# Calculate SSIM of FFTs
avg_ssim, ssim_image = ssim(mag_spect_roi_withbat, mag_spect_roi_wobat, full=True)

# Image similarity results (at center, where FFTs are aligned)
ssim_value = ssim_image[n][n]
correlation_value = correlation[2*n][2*n]

# Normalized correlation value (to average of auto-correlation values)
correlation_value_normalized = correlation_value / ((auto1_value + auto2_value) / 2.0)

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
plt.suptitle(file + ', Same Location Comparison', fontsize = titlefontsize)

plt.subplot(figrows, figcolumns, 1)
plt.cla()
plt.imshow(img_withbat_rotated_gray, cmap='gray', norm=norm)
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
plt.imshow(img_wobat_rotated_gray, cmap='gray', norm=norm)
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
plt.text(0.05, 0.6, 'Correlation = %s' % round(correlation_value, 4))
plt.text(0.05, 0.3, 'Normalized = %s' % round(correlation_value_normalized, 4))
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