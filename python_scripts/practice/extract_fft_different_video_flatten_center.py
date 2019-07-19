# Import useful libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.measure import compare_ssim as ssim
import matplotlib as mpl
from mpl_toolkits import mplot3d
import csv
import os
import datetime

stamp = datetime.datetime.now().microsecond # Stamp for figure title

plt.close() # Close any previous matplotlib.pyplot windows

n = 20
s = n * 2 + 1 # Length of square sides

squarethickness = 3

# Set up folder structure and read/write paths
cwd = os.getcwd()
scriptfolder = 'python_scripts'
framefolder1 = 'frames'
framefolder2 = 'clear_background'
homefolder = cwd[:-len(scriptfolder)]
readlocation = os.path.join(homefolder, framefolder1, framefolder2)
readlocation_input = homefolder
inputfile = 'input.csv'
outputfolder1 = 'output'
outputfolder2 = 'figs'
writelocation = os.path.join(homefolder, outputfolder1, outputfolder2)
writelocation_output = os.path.join(homefolder, outputfolder1)
outputfile = 'output.csv'

# Define find element function for column
def findelements(inputlist, Duplicate=False):
        outputlist = []
        for element in inputlist:
                if element.startswith('#'):
                        continue
                elif element == '':
                        continue
                else:
                        outputlist.append(element)
        if not Duplicate:
                if len(outputlist) == 1:
                        return outputlist[0]
                else:
                        print('Error: Elements in input column =/= 1')
        else:
                if len(outputlist) == 2:
                        return outputlist
                elif len(outputlist) == 1:
                        return outputlist[:1]
                else:
                        print('Error: Elements in input column =/= 2 or 1')

# Define find file name column function
def findfilelist(objecttype, distance):
        filelist = ''
        if objecttype == 'airplanes':
                if distance == 'close':
                        filelist = row[4]
                else:
                        filelist = row[5]
        elif objecttype == 'bats':
                if distance == 'close':
                        filelist = row[6]
                else:
                        filelist = row[7]
        elif objecttype == 'birds':
                if distance == 'close':
                        filelist = row[8]
                else:
                        filelist.append(row[9])
        elif objecttype == 'insects':
                if distance == 'close':
                        filelist = row[10]
        return filelist

# Set up empty lists to hold data from input file
objecttype1_ = []
objecttype2_ = []
distance1_ = []
distance2_ = []
filename1_ = []
filename2_ = []
frame1_ = []
frame2_ = []

# Read object type, distance, and frame columns
with open(os.path.join(readlocation_input, inputfile), newline='') as csvfile:
        inputreader = csv.reader(csvfile)
        for row in inputreader:
                objecttype1_.append(row[0])
                distance1_.append(row[1])
                objecttype2_.append(row[2])
                distance2_.append(row[3])

                frame1_.append(row[11])
                frame2_.append(row[12])

# Find object type, distance, and frame from columns
objecttype1 = findelements(objecttype1_)
objecttype2 = findelements(objecttype2_)
distance1 = findelements(distance1_)
distance2 = findelements(distance2_)
frame1 = findelements(frame1_)
frame2 = findelements(frame2_)

# Read file name columns
with open(os.path.join(readlocation_input, inputfile), newline='') as csvfile:
        inputreader = csv.reader(csvfile)
        for row in inputreader:
                filename1_.append(findfilelist(objecttype1, distance1))
                filename2_.append(findfilelist(objecttype2, distance2))

# Find file names from columns
if filename1_ == filename2_:
        filenames_ = findelements(filename1_, Duplicate=True)
        if len(filenames_) == 2:
                filename1 = filenames_[0]
                filename2 = filenames_[1]
        else:
                filename1 = filenames_[0]
                filename2 = filenames_[0]
else:
        filename1 = findelements(filename1_)
        filename2 = findelements(filename2_)

## Print results
# print(objecttype1, distance1, filename1, frame1, objecttype2, distance2, filename2,  frame2)

extension = '.jpg'

# Set up frame read paths
readpath1 = os.path.join(readlocation, objecttype1, distance1, filename1, 'frame%s%s' % (frame1, extension))
readpath2 = os.path.join(readlocation, objecttype2, distance2, filename2, 'frame%s%s' % (frame2, extension))

# Set window names
frameTitle1 = 'Video ' + filename1 + ', Frame ' + frame1
window1Name = frameTitle1
window2Name = frameTitle1 + ', Rotated'
frameTitle2 = 'Video ' + filename2 + ', Frame ' + frame2
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
        
        return cv2.warpAffine(image_name, M, (c, r)), sign * rot_angle_deg

# Define FFT function
def takedft(img_name):
        dft = cv2.dft(np.float32(img_name), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
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
img1_rotated = rotate(img1, found_angle1, 1)[0]
img1_rotangle = rotate(img1, found_angle1, 1)[1]
cv2.namedWindow(window2Name)
cv2.setMouseCallback(window2Name, click_event2)
cv2.imshow(window2Name, img1_rotated)
cv2.waitKey(0) & 0xFF

# Coordinates of last clicked region
roi1_x = ref_location1[-1][0]
roi1_y = ref_location1[-1][1]

# print('\nPress any keys to progress.\n')

# Convert rotated frame 1 to grayscale and clone
clone_img1_rotated_gray = cv2.cvtColor(img1_rotated, cv2.COLOR_BGR2GRAY)
img1_rotated_gray = cv2.cvtColor(img1_rotated, cv2.COLOR_BGR2GRAY)

# Show grayscale rotated frame 1 with square
cv2.rectangle(img1_rotated_gray, (roi1_x - n, roi1_y - n), (roi1_x + n, roi1_y + n), (0, 0, 0), squarethickness)
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

# Find rotation angle from clicked points
found_angle2 = find_angle(xi2, yi2, xf2, yf2)

# Rotate image and show rotated version
img2_rotated = rotate(img2, found_angle2, 2)[0]
img2_rotangle = rotate(img2, found_angle2, 2)[1]
cv2.namedWindow(window4Name)
cv2.setMouseCallback(window4Name, click_event4)
cv2.imshow(window4Name, img2_rotated)
cv2.waitKey(0) & 0xFF

# print('\nClick in the middle of the bat and press any key to progress. The region of interest coordinates will be saved as your last click.\n')

# Coordinates of last clicked region
roi2_x = ref_location2[-1][0]
roi2_y = ref_location2[-1][1]

# print('\nPress any keys to progress.\n')

# Convert rotated frame 2 to grayscale and clone
clone_img2_rotated_gray = cv2.cvtColor(img2_rotated, cv2.COLOR_BGR2GRAY)
img2_rotated_gray = cv2.cvtColor(img2_rotated, cv2.COLOR_BGR2GRAY)

# Show grayscale rotated frame 2 with square
cv2.rectangle(img2_rotated_gray, (roi2_x - n, roi2_y - n), (roi2_x + n, roi2_y + n), (0, 0, 0), squarethickness)
cv2.imshow(window4Name, img2_rotated_gray)
cv2.waitKey(0) & 0xFF

# Crop frame 2 around same location
roi2 = clone_img2_rotated_gray[(roi2_y - n):(roi2_y + n + 1), (roi2_x - n):(roi2_x + n + 1)]

cv2.destroyAllWindows()

# Take FFT and extract magnitude spectrum of cropped images
takedft_roi1 = takedft(roi1)
mag_spect_roi1 = takedft_roi1[2]
takedft_roi2 = takedft(roi2)
mag_spect_roi2 = takedft_roi2[2]

# Version 2: set center points of FFTs equal to zero
mag_spect_roi1_v2 = mag_spect_roi1.copy()
mag_spect_roi1_v2[n][n] = 0.0
mag_spect_roi2_v2 = mag_spect_roi2.copy()
mag_spect_roi2_v2[n][n] = 0.0

# Version 3: set center nine squares of FFTs equal to zero
mag_spect_roi1_v3 = mag_spect_roi1.copy()
mag_spect_roi1_v3[n-1:n+2,n-1:n+2] = np.zeros((3,3))
mag_spect_roi2_v3 = mag_spect_roi2.copy()
mag_spect_roi2_v3[n-1:n+2,n-1:n+2] = np.zeros((3,3))

# Correlate FFTs
correlation = signal.correlate2d(mag_spect_roi1, mag_spect_roi2, fillvalue=0)

# Try template matching
template_matched = cv2.matchTemplate(mag_spect_roi1, mag_spect_roi2, cv2.TM_CCORR_NORMED)

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

## Do the same with the v2s

# Correlate FFTs
correlation_v2 = signal.correlate2d(mag_spect_roi1_v2, mag_spect_roi2_v2, fillvalue=0)

# Try template matching
template_matched_v2 = cv2.matchTemplate(mag_spect_roi1_v2, mag_spect_roi2_v2, cv2.TM_CCORR_NORMED)

#Correlate FFTs to themselves and extract value results
auto1_v2 = signal.correlate2d(mag_spect_roi1_v2, mag_spect_roi1_v2, fillvalue=0)
auto2_v2 = signal.correlate2d(mag_spect_roi2_v2, mag_spect_roi2_v2, fillvalue=0)

auto1_value_v2 = auto1_v2[2*n][2*n]
auto2_value_v2 = auto2_v2[2*n][2*n]

# Calculate SSIM of FFTs
avg_ssim_v2, ssim_image_v2 = ssim(mag_spect_roi1_v2, mag_spect_roi2_v2, full=True)

# Image similarity results (at center, where FFTs are aligned)
ssim_value_v2 = ssim_image_v2[n][n]
correlation_value_v2 = correlation_v2[2*n][2*n]

# Normalized correlation value (to average of auto-correlation values)
correlation_value_normalized_v2 = correlation_value_v2 / ((auto1_value_v2 + auto2_value_v2) / 2.0)

## Do the same with the v3s

# Correlate FFTs
correlation_v3 = signal.correlate2d(mag_spect_roi1_v3, mag_spect_roi2_v3, fillvalue=0)

# Try template matching
template_matched_v3 = cv2.matchTemplate(mag_spect_roi1_v3, mag_spect_roi2_v3, cv2.TM_CCORR_NORMED)

#Correlate FFTs to themselves and extract value results
auto1_v3 = signal.correlate2d(mag_spect_roi1_v3, mag_spect_roi1_v3, fillvalue=0)
auto2_v3 = signal.correlate2d(mag_spect_roi2_v3, mag_spect_roi2_v3, fillvalue=0)

auto1_value_v3 = auto1_v3[2*n][2*n]
auto2_value_v3 = auto2_v3[2*n][2*n]

# Calculate SSIM of FFTs
avg_ssim_v3, ssim_image_v3 = ssim(mag_spect_roi1_v3, mag_spect_roi2_v3, full=True)

# Image similarity results (at center, where FFTs are aligned)
ssim_value_v3 = ssim_image_v3[n][n]
correlation_value_v3 = correlation_v3[2*n][2*n]

# Normalized correlation value (to average of auto-correlation values)
correlation_value_normalized_v3 = correlation_value_v3 / ((auto1_value_v3 + auto2_value_v3) / 2.0)

## Show results

titlefontsize = 12
subtitlefontsize = 10
figrows = 3
figcolumns = 4

# Set up grayscale normalization conditions
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

plt.figure(1, figsize=(figcolumns*3, figrows*3))
plt.suptitle('%s, Frame %s vs. %s, Frame %s\n%s' % (filename1, frame1, filename2, frame2, stamp), fontsize = titlefontsize)

plt.subplot(figrows, figcolumns, 1)
plt.cla()
plt.imshow(img1_rotated_gray, cmap='gray', norm=norm1)
plt.title('%s, Frame %s' % (filename1, frame1), fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 2)
plt.cla()
plt.imshow(roi1, cmap='gray', norm=norm1)
plt.title('ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 3)
plt.cla()
plt.imshow(mag_spect_roi1, cmap='gray', norm=norm2)
plt.title('FFT of ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax1 = plt.subplot(figrows, figcolumns, 4, projection='3d')
plt.cla()
X1, Y1 = np.meshgrid(range(s), range(s))
Z1 = mag_spect_roi1
mplot3d.Axes3D.plot_surface(ax1, X1, Y1, Z1, cmap='gray', norm=norm2)
plt.title('FFT of ROI, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax1, bottom=0.0, top=plottop)
mplot3d.Axes3D.set_zticks(ax1, [])
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 5)
plt.cla()
plt.imshow(img2_rotated_gray, cmap='gray', norm=norm1)
plt.title('%s, Frame %s' % (filename2, frame2), fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 6)
plt.cla()
plt.imshow(roi2, cmap='gray', norm=norm1)
plt.title('ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 7)
plt.cla()
plt.imshow(mag_spect_roi2, cmap='gray', norm=norm2)
plt.title('FFT of ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax2 = plt.subplot(figrows, figcolumns, 8, projection='3d')
plt.cla()
X2, Y2 = np.meshgrid(range(s), range(s))
Z2 = mag_spect_roi2
mplot3d.Axes3D.plot_surface(ax2, X2, Y2, Z2, cmap='gray', norm=norm2)
plt.title('FFT of ROI, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax2, bottom=0.0, top=plottop)
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
plt.text(0.05, 0.75, 'Correlation = %s' % round(correlation_value, 4))
plt.text(0.05, 0.5, '"Normalized" = %s' % round(correlation_value_normalized, 4))
plt.text(0.05, 0.25, 'Normalized = %s' % round(np.amax(template_matched), 4))
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

## Do the same with the v2s

plt.figure(2, figsize=(figcolumns*3, figrows*3))
plt.suptitle('%s, Frame %s vs. %s, Frame %s\nCenter Value Set to 0\n%s' % (filename1, frame1, filename2, frame2, stamp), fontsize = titlefontsize)

plt.subplot(figrows, figcolumns, 1)
plt.cla()
plt.imshow(img1_rotated_gray, cmap='gray', norm=norm1)
plt.title('%s, Frame %s' % (filename1, frame1), fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 2)
plt.cla()
plt.imshow(roi1, cmap='gray', norm=norm1)
plt.title('ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 3)
plt.cla()
plt.imshow(mag_spect_roi1_v2, cmap='gray', norm=norm2)
plt.title('FFT of ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax12 = plt.subplot(figrows, figcolumns, 4, projection='3d')
plt.cla()
X12, Y12 = np.meshgrid(range(s), range(s))
Z12 = mag_spect_roi1_v2
mplot3d.Axes3D.plot_surface(ax12, X12, Y12, Z12, cmap='gray', norm=norm2)
plt.title('FFT of ROI, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax12, bottom=0.0, top=plottop)
mplot3d.Axes3D.set_zticks(ax12, [])
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 5)
plt.cla()
plt.imshow(img2_rotated_gray, cmap='gray', norm=norm1)
plt.title('%s, Frame %s' % (filename2, frame2), fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 6)
plt.cla()
plt.imshow(roi2, cmap='gray', norm=norm1)
plt.title('ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 7)
plt.cla()
plt.imshow(mag_spect_roi2_v2, cmap='gray', norm=norm2)
plt.title('FFT of ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax22 = plt.subplot(figrows, figcolumns, 8, projection='3d')
plt.cla()
X22, Y22 = np.meshgrid(range(s), range(s))
Z22 = mag_spect_roi2_v2
mplot3d.Axes3D.plot_surface(ax22, X22, Y22, Z22, cmap='gray', norm=norm2)
plt.title('FFT of ROI, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax22, bottom=0.0, top=plottop)
mplot3d.Axes3D.set_zticks(ax22, [])
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 9)
plt.cla()
plt.imshow(correlation_v2, cmap='gray')
plt.title('Correlation Image of FFTs', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 10)
plt.cla()
plt.text(0.05, 0.75, 'Correlation = %s' % round(correlation_value_v2, 4))
plt.text(0.05, 0.5, '"Normalized" = %s' % round(correlation_value_normalized_v2, 4))
plt.text(0.05, 0.25, 'Normalized = %s' % round(np.amax(template_matched_v2), 4))
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 11)
plt.cla()
plt.imshow(ssim_image_v2, cmap='gray')
plt.title('SSIM Image of FFTs', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 12)
plt.cla()
plt.text(0.25, 0.5, 'SSIM = %s' % round(ssim_value_v2, 4))
plt.xticks([])
plt.yticks([])

## Do the same with the v3s

plt.figure(3, figsize=(figcolumns*3, figrows*3))
plt.suptitle('%s, Frame %s vs. %s, Frame %s\nCenter 9 Values Set to 0\n%s' % (filename1, frame1, filename2, frame2, stamp), fontsize = titlefontsize)

plt.subplot(figrows, figcolumns, 1)
plt.cla()
plt.imshow(img1_rotated_gray, cmap='gray', norm=norm1)
plt.title('%s, Frame %s' % (filename1, frame1), fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 2)
plt.cla()
plt.imshow(roi1, cmap='gray', norm=norm1)
plt.title('ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 3)
plt.cla()
plt.imshow(mag_spect_roi1_v3, cmap='gray', norm=norm2)
plt.title('FFT of ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax13 = plt.subplot(figrows, figcolumns, 4, projection='3d')
plt.cla()
X13, Y13 = np.meshgrid(range(s), range(s))
Z13 = mag_spect_roi1_v3
mplot3d.Axes3D.plot_surface(ax13, X13, Y13, Z13, cmap='gray', norm=norm2)
plt.title('FFT of ROI, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax13, bottom=0.0, top=plottop)
mplot3d.Axes3D.set_zticks(ax13, [])
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 5)
plt.cla()
plt.imshow(img2_rotated_gray, cmap='gray', norm=norm1)
plt.title('%s, Frame %s' % (filename2, frame2), fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 6)
plt.cla()
plt.imshow(roi2, cmap='gray', norm=norm1)
plt.title('ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 7)
plt.cla()
plt.imshow(mag_spect_roi2_v3, cmap='gray', norm=norm2)
plt.title('FFT of ROI', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

ax23 = plt.subplot(figrows, figcolumns, 8, projection='3d')
plt.cla()
X23, Y23 = np.meshgrid(range(s), range(s))
Z23 = mag_spect_roi2_v3
mplot3d.Axes3D.plot_surface(ax23, X23, Y23, Z23, cmap='gray', norm=norm2)
plt.title('FFT of ROI, 3D', fontsize = subtitlefontsize)
mplot3d.Axes3D.set_zlim3d(ax23, bottom=0.0, top=plottop)
mplot3d.Axes3D.set_zticks(ax23, [])
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 9)
plt.cla()
plt.imshow(correlation_v3, cmap='gray')
plt.title('Correlation Image of FFTs', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 10)
plt.cla()
plt.text(0.05, 0.75, 'Correlation = %s' % round(correlation_value_v3, 4))
plt.text(0.05, 0.5, '"Normalized" = %s' % round(correlation_value_normalized_v3, 4))
plt.text(0.05, 0.25, 'Normalized = %s' % round(np.amax(template_matched_v3), 4))
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 11)
plt.cla()
plt.imshow(ssim_image_v3, cmap='gray')
plt.title('SSIM Image of FFTs', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(figrows, figcolumns, 12)
plt.cla()
plt.text(0.25, 0.5, 'SSIM = %s' % round(ssim_value_v3, 4))
plt.xticks([])
plt.yticks([])

plt.show()

# Create output figure title
figextension = '.jpg'
figtitle = '%s_%s_%s_%s_%s%s' % (filename1, frame1, filename2, frame2, stamp, figextension)

WriteFile = False

## Write results to csv output file
if WriteFile:
        with open(os.path.join(writelocation_output, outputfile), mode='a', newline='') as csvfile:
                outputwriter = csv.writer(csvfile)
                outputwriter.writerow([figtitle, '', objecttype1, distance1, filename1, frame1,
                                        objecttype2, distance2, filename2, frame2, '',
                                        correlation_value, correlation_value_normalized, ssim_value, '',
                                        img1_rotangle, roi1_x, roi1_y, img2_rotangle, roi2_x, roi2_y])
        plt.savefig(os.path.join(writelocation, figtitle))
        print (figtitle)
else:
        print ('Figure not saved')

print ("Done")