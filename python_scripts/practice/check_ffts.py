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

category1 = 'bats'
category2 = 'airplanes'

readpath_input = '/Users/icunitz/Desktop/bat_detection/output/final/%s_%s.csv' % (category1, category2)

readlocation_frames = '/Users/icunitz/Desktop/bat_detection/frames/final'
readpath1 = os.path.join(readlocation_frames, category1)
readpath2 = os.path.join(readlocation_frames, category2)

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

comp_dict = {}
count = 1

# Read file names, SSIM, nccorr, etc. from output .csv file
with open(readpath_input, newline='') as csvfile:
    inputreader = csv.reader(csvfile)
    next(inputreader)
    for row in inputreader:
        comp_dict[count] = {}
        comp_dict[count]['file1'] = row[0]
        comp_dict[count]['file2'] = row[1]
        comp_dict[count]['ssim_read'] = float(row[2])
        comp_dict[count]['nccorr_read'] = float(row[3])
        comp_dict[count]['ssim_angle'] = float(row[4])
        comp_dict[count]['nccorr_angle'] = float(row[5])
        comp_dict[count]['roi1_x'] = int(row[6])
        comp_dict[count]['roi1_y'] = int(row[7])
        comp_dict[count]['roi2_x'] = int(row[8])
        comp_dict[count]['roi2_y'] = int(row[9])

        count += 1

x = 996

readlocation1 = os.path.join(readpath1, comp_dict[x]['file1'])
readlocation2 = os.path.join(readpath2, comp_dict[x]['file2'])

img1 = cv2.imread(readlocation1)
img2 = cv2.imread(readlocation2)

# Take FFT of ROI 2
roi2 = roi(img2, comp_dict[x]['roi2_x'], comp_dict[x]['roi2_y'])
roi2_gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
roi2_mag_spect = takedft(roi2_gray)[2]

## Rotate ROI1 to SSIM angle, take FFT, and calculate SSIM
roi1_rot_s = roi(rotate(img1, comp_dict[x]['ssim_angle'], comp_dict[x]['roi1_x'],
            comp_dict[x]['roi1_y']), comp_dict[x]['roi1_x'], comp_dict[x]['roi1_y'])
roi1_rot_gray_s = cv2.cvtColor(roi1_rot_s, cv2.COLOR_BGR2GRAY)
roi1_rot_mag_spect_s = takedft(roi1_rot_gray_s)[2]

avg_ssim, ssim_image = ssim(roi1_rot_mag_spect_s, roi2_mag_spect, full=True)
ssim_value = float(ssim_image[n][n])

## Rotate ROI1 to nccorr angle, take FFT, and calculate nccorr
roi1_rot_n = roi(rotate(img1, comp_dict[x]['nccorr_angle'], comp_dict[x]['roi1_x'],
            comp_dict[x]['roi1_y']), comp_dict[x]['roi1_x'], comp_dict[x]['roi1_y'])
roi1_rot_gray_n = cv2.cvtColor(roi1_rot_n, cv2.COLOR_BGR2GRAY)
roi1_rot_mag_spect_n = takedft(roi1_rot_gray_n)[2]

template_matched = cv2.matchTemplate(roi1_rot_mag_spect_n, roi2_mag_spect, cv2.TM_CCORR_NORMED)
nccorr_value = float(np.amax(template_matched))

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
plt.suptitle('%s vs. %s\nSSIM = %s' % (comp_dict[x]['file1'], comp_dict[x]['file2'], round(ssim_value, 4)))

plt.subplot(2, 3, 1)
plt.cla()
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Frame 1')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 2)
plt.cla()
plt.imshow(cv2.cvtColor(roi1_rot_s, cv2.COLOR_BGR2RGB))
plt.title('Object 1: %s, rotated %s deg' % (category1[:-1], int(comp_dict[x]['ssim_angle'])))
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 3)
plt.cla()
plt.imshow(roi1_rot_mag_spect_s, cmap='gray', norm=norm2)
plt.title('Object 1 Magnitude Spectrum')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 4)
plt.cla()
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Frame 2')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 5)
plt.cla()
plt.imshow(cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))
plt.title('Object 2: %s' % category2[:-1])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 6)
plt.cla()
plt.imshow(roi2_mag_spect, cmap='gray', norm=norm2)
plt.title('Object 2 Magnitude Spectrum')
plt.xticks([])
plt.yticks([])

plt.show()



# errorlist_ = []
# 
# # Loop over comparisons
# for x in comp_dict:
#     readlocation1 = os.path.join(readpath1, comp_dict[x]['file1'])
#     readlocation2 = os.path.join(readpath2, comp_dict[x]['file2'])
# 
#     img1 = cv2.imread(readlocation1)
#     img2 = cv2.imread(readlocation2)
# 
#     # Take FFT of ROI 2
#     roi2 = roi(img2, comp_dict[x]['roi2_x'], comp_dict[x]['roi2_y'])
#     roi2_gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
#     roi2_mag_spect = takedft(roi2_gray)[2]
# 
#     ## Rotate ROI1 to SSIM angle, take FFT, and calculate SSIM
#     roi1_rot_s = roi(rotate(img1, comp_dict[x]['ssim_angle'], comp_dict[x]['roi1_x'],
#                 comp_dict[x]['roi1_y']), comp_dict[x]['roi1_x'], comp_dict[x]['roi1_y'])
#     roi1_rot_gray_s = cv2.cvtColor(roi1_rot_s, cv2.COLOR_BGR2GRAY)
#     roi1_rot_mag_spect_s = takedft(roi1_rot_gray_s)[2]
# 
#     avg_ssim, ssim_image = ssim(roi1_rot_mag_spect_s, roi2_mag_spect, full=True)
#     ssim_value = float(ssim_image[n][n])
# 
#     ## Rotate ROI1 to nccorr angle, take FFT, and calculate nccorr
#     roi1_rot_n = roi(rotate(img1, comp_dict[x]['nccorr_angle'], comp_dict[x]['roi1_x'],
#                 comp_dict[x]['roi1_y']), comp_dict[x]['roi1_x'], comp_dict[x]['roi1_y'])
#     roi1_rot_gray_n = cv2.cvtColor(roi1_rot_n, cv2.COLOR_BGR2GRAY)
#     roi1_rot_mag_spect_n = takedft(roi1_rot_gray_n)[2]
# 
#     template_matched = cv2.matchTemplate(roi1_rot_mag_spect_n, roi2_mag_spect, cv2.TM_CCORR_NORMED)
#     nccorr_value = float(np.amax(template_matched))
# 
#     # Check if newly calculated values are the same as old values
#     ssim_check = round(comp_dict[x]['ssim_read'], 4) == round(ssim_value, 4)
#     nccorr_check = round(comp_dict[x]['nccorr_read'], 4) == round(nccorr_value, 4)
# 
#     # print(ssim_check)
#     # print(nccorr_check)
# 
#     if ssim_check==False:
#         errorlist_.append(['%s_%s: SSIM difference' % (comp_dict[x]['file1'], comp_dict[x]['file2']), comp_dict[x]['ssim_read'], ssim_value])
# 
#     if nccorr_check==False:
#        errorlist_.append(['%s_%s: nccorr difference' % (comp_dict[x]['file1'], comp_dict[x]['file2']), comp_dict[x]['nccorr_read'], nccorr_value])
# 
# if errorlist_ == []:
#     print('Comparisons check out: %s vs. %s' % (category1, category2))
# else:
#     print(errorlist_)