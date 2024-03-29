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
frame1 = input('Frame 1 Number: ')
frame2 = input('Frame 2 Number: ')
frameextension = '.jpg'
readpath1 = '/Users/icunitz/Desktop/bat_detection/frames/' + file + '/frame' + frame1 + frameextension # Customize this based on directories in computer
readpath2 = '/Users/icunitz/Desktop/bat_detection/frames/' + file + '/frame' + frame2 + frameextension

n = 20
s = n * 2 + 1 # Length of square sides

# Set window names
frameTitle1 = 'Video ' + file + ', Frame ' + frame1
window1Name = frameTitle1
window2Name = frameTitle1 + ', Rotated'
frameTitle2 = 'Video ' + file + ', Frame ' + frame2
window3Name = frameTitle2
window4Name = frameTitle2 + ', Rotated'

# Set up variables for click event 1
drawLine1 = False
xi1, yi1 = 0, 0
xf1, yf1 = 0, 0

# Set up variables for click event 3
drawLine2 = False
xi2, yi2 = 0, 0
xf2, yf2 = 0, 0

ref_location1 = [] # Empty list to hold click locations on Frame 1
ref_location2 = [] # Empty list to hold click locations on Frame 2

# Click event 1 - to find rotation angle
def click_event1(event, x, y, flags, param):
        global xi1, yi1, xf1, yf1, drawLine1
        if event == cv2.EVENT_LBUTTONDOWN:
                drawLine1 = True
                xi1, yi1 = x, y
        if event == cv2.EVENT_LBUTTONUP:
                if drawLine1:
                        drawLine1 = False
                        xf1, yf1 = x, y

# Click event 2 - to find region of interest
def click_event2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_location1.append((x, y))

# Click event 3 - to find rotation angle
def click_event3(event, x, y, flags, param):
        global xi2, yi2, xf2, yf2, drawLine2
        if event == cv2.EVENT_LBUTTONDOWN:
                drawLine2 = True
                xi2, yi2 = x, y
        if event == cv2.EVENT_LBUTTONUP:
                if drawLine2:
                        drawLine2 = False
                        xf2, yf2 = x, y

# Click event 3 - to find region of interest
def click_event4(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_location2.append((x, y))

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
def rotate(image_name, a_found, instance):
        global yi1, yf1, yi2, yf2

        if instance == 1:
                yi = yi1
                yf = yf1
        elif instance == 2:
                yi = yi2
                yf = yf2
        else:
                print('\nError: Unexpected image rotation instance.\n')
        
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

# Show original frame 1
img1 = cv2.imread(readpath1)
cv2.namedWindow(window1Name)
cv2.setMouseCallback(window1Name, click_event1)
cv2.imshow(window1Name, img1)
cv2.waitKey(0) & 0xFF

# Find rotation angle from clicked points
found_angle1 = find_angle(xi1, yi1, xf1, yf1)

# print('\nClick in the middle of the bat and press any key to progress. The region of interest coordinates will be saved as your last click.\n')

# Rotate frame 1 and show rotated version
img1_rotated = rotate(img1, found_angle1, 1)
cv2.namedWindow(window2Name)
cv2.setMouseCallback(window2Name, click_event2)
cv2.imshow(window2Name, img1_rotated)
cv2.waitKey(0) & 0xFF

# Coordinates of region of interest
roi1_x = ref_location1[-1][0]
roi1_y = ref_location1[-1][1]

# print ('\nLocation of Interest, Frame %s: (%s, %s)' % (frame1, roi1_x, roi1_y))

# print('\nPress any keys to progress.\n')

# Convert rotated frame 1 to grayscale and clone
clone_img1_rotated_gray = cv2.cvtColor(img1_rotated, cv2.COLOR_BGR2GRAY)
img1_rotated_gray = cv2.cvtColor(img1_rotated, cv2.COLOR_BGR2GRAY)

# Show grayscale rotated frame 1 with square
cv2.rectangle(img1_rotated_gray, (roi1_x - n, roi1_y - n), (roi1_x + n, roi1_y + n), (0, 0, 0), 2)
cv2.imshow(window2Name, img1_rotated_gray)
cv2.waitKey(0) & 0xFF

# Crop frame 1 around last clicked location
roi1 = clone_img1_rotated_gray[(roi1_y - n):(roi1_y + n + 1), (roi1_x - n):(roi1_x + n + 1)]

# Read in original frame 2 
img2 = cv2.imread(readpath2)
cv2.namedWindow(window3Name)
cv2.setMouseCallback(window3Name, click_event3)
cv2.imshow(window3Name, img2)
cv2.waitKey(0) & 0xFF

# print('\nClick in the middle of the bat and press any key to progress. The region of interest coordinates will be saved as your last click.\n')

# Find rotation angle from clicked points
found_angle2 = find_angle(xi2, yi2, xf2, yf2)

# Rotate image and show rotated version
img2_rotated = rotate(img2, found_angle2, 2)
cv2.namedWindow(window4Name)
cv2.setMouseCallback(window4Name, click_event4)
cv2.imshow(window4Name, img2_rotated)
cv2.waitKey(0) & 0xFF

# Coordinates of last clicked region
roi2_x = ref_location2[-1][0]
roi2_y = ref_location2[-1][1]

# print ('\nLocation of Interest, Frame %s: (%s, %s)' % (frame2, roi2_x, roi2_y))

# print('\nPress any keys to progress.\n')

# Convert rotated frame 2 to grayscale and clone
clone_img2_rotated_gray = cv2.cvtColor(img2_rotated, cv2.COLOR_BGR2GRAY)
img2_rotated_gray = cv2.cvtColor(img2_rotated, cv2.COLOR_BGR2GRAY)

# Show grayscale rotated frame 2 with square
cv2.rectangle(img2_rotated_gray, (roi2_x - n, roi2_y - n), (roi2_x + n, roi2_y + n), (0, 0, 0), 2)
cv2.imshow(window4Name, img2_rotated_gray)
cv2.waitKey(0) & 0xFF

# Crop frame 2 around last clicked location
roi2 = clone_img2_rotated_gray[(roi2_y - n):(roi2_y + n + 1), (roi2_x - n):(roi2_x + n + 1)]

cv2.destroyAllWindows()

# Take FFT and extract magnitude spectrum of cropped images
takedft_roi1 = takedft(roi1)
mag_spect_roi1 = takedft_roi1[2]
takedft_roi2 = takedft(roi2)
mag_spect_roi2 = takedft_roi2[2]

# Correlate FFTs
correlation = signal.correlate2d(mag_spect_roi1, mag_spect_roi2, fillvalue=0)

#Correlate FFTs to themselves and extract value results
auto1 = signal.correlate2d(mag_spect_roi1, mag_spect_roi1, fillvalue=0)
auto2 = signal.correlate2d(mag_spect_roi2, mag_spect_roi2, fillvalue=0)

auto1_value = auto1[2*n][2*n]
auto2_value = auto2[2*n][2*n]

# Calculate SSIM of FFTs
avg_ssim, ssim_image = ssim(mag_spect_roi1, mag_spect_roi2, full=True)

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
plt.suptitle(file + ', Same Video Comparison', fontsize = titlefontsize)

plt.subplot(figrows, figcolumns, 1)
plt.cla()
plt.imshow(img1_rotated_gray, cmap='gray', norm=norm)
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
plt.imshow(img2_rotated_gray, cmap='gray', norm=norm)
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