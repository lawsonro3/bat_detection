import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.measure import compare_ssim as ssim
import matplotlib as mpl
from mpl_toolkits import mplot3d
import os
import csv

plt.close() # Close any previous matplotlib.pyplot windows

n = 20
s = n * 2 + 1 # Length of square sides

category1 = 'insects'

file1 = 'insect_127862_frame0069.jpg'

readlocation_frames = '/Users/icunitz/Desktop/bat_detection/frames/final'
readpath1 = os.path.join(readlocation_frames, category1, file1)
readpath_roi = '/Users/icunitz/Desktop/bat_detection/output/final/roi_locations_%s.csv' % category1

writelocation = '/Users/icunitz/Desktop'

window1Name = file1
window2Name = file1 + ' ROI'
window3Name = file1 + ' ROI mag spect'

# Define image rotating function
def rotate(image_name, angle_deg, x, y):
    rows, cols = image_name[:,:,0].shape
    M = cv2.getRotationMatrix2D((x, y), angle_deg, 1)
    return cv2.warpAffine(image_name, M, (cols, rows))

# Define take ROI function
def roi(image_name, x, y):
    global n
    roi = image_name[y-n : y+n+1, x-n : x+n+1, :]
    return roi

# Define FFT function
def takedft(img_name):
        dft = cv2.dft(np.float32(img_name), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return dft, dft_shift, magnitude_spectrum

# Set up variables for click event
drawLine = False
xi, yi = 0, 0
xf, yf = 0, 0

# Click event - to find rotation angle
def click_event(event, x, y, flags, param):
        global xi, yi, xf, yf, drawLine
        if event == cv2.EVENT_LBUTTONDOWN:
                drawLine = True
                xi, yi = x, y
        if event == cv2.EVENT_LBUTTONUP:
                if drawLine:
                        drawLine = False
                        xf, yf = x, y

# Define rotation angle finding function
def find_angle(xi, yi, xf, yf):
    det = (yi - yf) / (xf - xi)
    angle_rad = np.arctan(det)
    angle_deg = angle_rad * (180 / np.pi)

    if yi > yf:
        if det > 0:
            rot_angle_deg = 90.0 - angle_deg
        else:
            rot_angle_deg = -1.0 * (angle_deg - 90.0)
            print('Negative rotation angle')
    else:
        if det > 0:
            rot_angle_deg = -1.0 * (angle_deg + 90.0)
            print('Negative rotation angle')
        else:
            rot_angle_deg = 270.0 - angle_deg

    return rot_angle_deg

# Extract ROI coordinates
roi_dict = {}
with open(readpath_roi, newline = '') as csvfile:
    inputreader = csv.reader(csvfile)
    next(inputreader)
    for row in inputreader:
        filename = row[0]
        roi_dict[filename] = [int(row[1]), int(row[2])]
        print(filename)
roi_x = roi_dict[file1][0]
roi_y = roi_dict[file1][1]

# Show original frame 1
img1 = cv2.imread(readpath1)
cv2.namedWindow(window1Name)
cv2.setMouseCallback(window1Name, click_event)
cv2.imshow(window1Name, img1)
cv2.waitKey(0) & 0xFF

# Find rotation angle from clicked points
angle1 = find_angle(xi, yi, xf, yf)

# Find rotated ROI 1 and mag spect 1
roi1_rot = roi(rotate(img1, angle1, roi_x, roi_y), roi_x, roi_y)
roi1_rot_gray = cv2.cvtColor(roi1_rot, cv2.COLOR_BGR2GRAY)
roi1_rot_mag_spect = takedft(roi1_rot_gray)[2]

cv2.imshow(window2Name, roi1_rot)
cv2.waitKey(0) & 0xFF

cv2.imshow(window3Name, roi1_rot_mag_spect)
cv2.waitKey(0) & 0xFF

cv2.imwrite(os.path.join(writelocation, '%s_ROI.png' % file1), roi1_rot)

Normalization2 = True

if Normalization2:
        plottop = 14.0
        norm2 = mpl.colors.Normalize(vmin = 0, vmax = plottop)
else:
        plottop = None
        norm2 = None

plt.imshow(roi1_rot_mag_spect, cmap='gray', norm = norm2)
plt.xticks([])
plt.yticks([])
plt.show()

plt.savefig(os.path.join(writelocation, 'img'))

cv2.destroyAllWindows()