# Import useful libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations as comb
from scipy import signal
from skimage.measure import compare_ssim as ssim
import matplotlib as mpl
from mpl_toolkits import mplot3d
import csv
import os

n = 20
s = n * 2 + 1 # Length of square sides

category = 'bats'

# Set up .csv input/output file paths
input_output_location = '/Users/icunitz/Desktop/bat_detection/output/final'
input_file = 'roi_locations_%s.csv' % category
input_path = os.path.join(input_output_location, input_file)
output_file = '%s_%s.csv' % (category, category)
output_path = os.path.join(input_output_location, output_file)

# Set up frame reading paths
frame_location = '/Users/icunitz/Desktop/bat_detection/frames/final'
frame_path = os.path.join(frame_location, category)
frame_extension = '.jpg'

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
def takedft(img1_name):
        dft = cv2.dft(np.float32(img1_name), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return dft, dft_shift, magnitude_spectrum

# Define function to read ROI info, ROI, and FFT of ROI from input file
def readroi(input_path, save_roi_fft=False, frame_path=''):
    #global frame_extension
    roi_dict = {}
    with open(input_path, newline='') as csvfile:
        inputreader = csv.reader(csvfile)
        next(inputreader)
        for row in inputreader:
            roi_filename = row[0]
            roi_x = int(row[1])
            roi_y = int(row[2])
            roi_dict[roi_filename] = [roi_x, roi_y, 0, 0]

            # Save ROIs and FFTs of ROIs
            if save_roi_fft:
                img = cv2.imread(os.path.join(frame_path, roi_filename))
                roi_square = roi(img, roi_x, roi_y)
                roi_gray = cv2.cvtColor(roi_square, cv2.COLOR_BGR2GRAY)
                roi_mag_spect = takedft(roi_gray)[2]
                roi_dict[roi_filename][2] = [roi_gray]
                roi_dict[roi_filename][3] = [roi_mag_spect]
    return roi_dict

# Read ROI info for categories 1 and 2
rois = readroi(input_path, save_roi_fft=True, frame_path=frame_path)

# Set up comparison nested dictionary
comparison_dict = {}
for combination in comb(sorted(os.listdir(frame_path)), 2):
    file1 = combination[0]
    file2 = combination[1]
    if file1.endswith(frame_extension) and file2.endswith(frame_extension):
        comparison_dict['%s_%s' % (file1, file2)] = {}
        for angle in np.arange(0, 361, 4):
            comparison_dict['%s_%s' % (file1, file2)][angle] = {}

plt.close() # Close any previous plt plots

# Iterate through files
for combination in comb(sorted(os.listdir(frame_path)), 2):
    file1 = combination[0]
    file2 = combination[1]

    if file1.endswith(frame_extension) and file2.endswith(frame_extension):

        # Load image 1
        img1 = cv2.imread(os.path.join(frame_path, file1))

        # Extract ROI1 data
        roi1_x, roi1_y = rois[file1][0], rois[file1][1]

        # Take and rotate ROI1
        for angle in np.arange(0, 361, 4):
            # Take rotated ROI 1 and FFT 1
            roi1_rot = roi(rotate(img1, angle, roi1_x, roi1_y), roi1_x, roi1_y)
            roi1_rot_gray = cv2.cvtColor(roi1_rot, cv2.COLOR_BGR2GRAY)
            roi1_rot_mag_spect = takedft(roi1_rot_gray)[2]

            # Compare rotated ROI1 to ROI2s
            roi2_x = rois[file2][0]
            roi2_y = rois[file2][1]

            roi2_mag_spect = np.asarray(rois[file2][3])[0,:,:]

            # Calculate SSIM of FFTs
            avg_ssim, ssim_image = ssim(roi1_rot_mag_spect, roi2_mag_spect, full=True)
            ssim_value = ssim_image[n][n]

            # Use template matching to calculate normalized cross-correlation of FFTs
            template_matched = cv2.matchTemplate(roi1_rot_mag_spect, roi2_mag_spect, cv2.TM_CCORR_NORMED)
            nccorr_value = np.amax(template_matched)

            comparison_dict['%s_%s' % (file1, file2)][angle] = [ssim_value, nccorr_value, roi1_x, roi1_y, roi2_x, roi2_y]
            
            print(file1, file2, angle)

# Create dictionary of max angles and values (and ROI locations)
max_dict = {}
for comparison in comparison_dict:
    max_angle_ssim = max(comparison_dict[comparison], key = lambda x: comparison_dict[comparison][x][0])
    max_angle_nccorr = max(comparison_dict[comparison], key = lambda x: comparison_dict[comparison][x][1])
    
    max_ssim = comparison_dict[comparison][max_angle_ssim][0]
    max_nccorr = comparison_dict[comparison][max_angle_nccorr][1]

    roi1_x = comparison_dict[comparison][max_angle_ssim][2]
    roi1_y = comparison_dict[comparison][max_angle_ssim][3]
    roi2_x = comparison_dict[comparison][max_angle_ssim][4]
    roi2_y = comparison_dict[comparison][max_angle_ssim][5]

    max_dict[comparison] = [max_ssim, max_nccorr, max_angle_ssim, max_angle_nccorr, roi1_x, roi1_y, roi2_x, roi2_y]

# Write max angles and values to Excel sheet
for comparison in max_dict:
    file1 = comparison[:comparison.find('_' + category[:-1])]
    file2 = comparison[(1 + comparison.find('_' + category[:-1])):]

    with open(output_path, mode='a', newline='') as csvfile:
        outputwriter = csv.writer(csvfile)
        outputwriter.writerow([file1, file2,
                                max_dict[comparison][0], max_dict[comparison][1],
                                max_dict[comparison][2], max_dict[comparison][3],
                                max_dict[comparison][4], max_dict[comparison][5],
                                max_dict[comparison][6], max_dict[comparison][7]])
