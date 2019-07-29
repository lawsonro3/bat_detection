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

n = 20
s = n * 2 + 1 # Length of square sides

category1 = 'bats'
category2 = 'insects'

# Set up .csv input/output file paths
input_output_location = '/Users/icunitz/Desktop/bat_detection/output/final'
input1_file = 'roi_locations_%s.csv' % category1
input2_file = 'roi_locations_%s.csv' % category2
input1_path = os.path.join(input_output_location, input1_file)
input2_path = os.path.join(input_output_location, input2_file)
output_file = '%s_%s.csv' % (category1, category2)
output_path = os.path.join(input_output_location, output_file)

# Set up frame reading paths
frame_location = '/Users/icunitz/Desktop/bat_detection/frames/final'
frame1_path = os.path.join(frame_location, category1)
frame2_path = os.path.join(frame_location, category2)
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
rois1 = readroi(input1_path)
rois2 = readroi(input2_path, save_roi_fft=True, frame_path=frame2_path)

# Set up comparison nested dictionary
comparison_dict = {}
for file1 in sorted(os.listdir(frame1_path)):
    for file2 in sorted(os.listdir(frame2_path)):
        comparison_dict['%s_%s' % (file1, file2)] = {}
        for angle in np.arange(0, 361, 4):
            comparison_dict['%s_%s' % (file1, file2)][angle] = {}

plt.close() # Close any previous plt plots

# Iterate through files
for file1 in sorted(os.listdir(frame1_path)):
    # Load image 1
    img1 = cv2.imread(os.path.join(frame1_path, file1))

    # Extract ROI1 data
    roi1_x, roi1_y = rois1[file1][0], rois1[file1][1]

    # Take and rotate ROI1
    for angle in np.arange(0, 361, 4):
        # Take rotated ROI 1 and FFT 1
        roi1_rot = roi(rotate(img1, angle, roi1_x, roi1_y), roi1_x, roi1_y)
        roi1_rot_gray = cv2.cvtColor(roi1_rot, cv2.COLOR_BGR2GRAY)
        roi1_rot_mag_spect = takedft(roi1_rot_gray)[2]

        # Compare rotated ROI1 to ROI2s
        for file2 in sorted(os.listdir(frame2_path)):
            roi2_x = rois2[file2][0]
            roi2_y = rois2[file2][1]

            roi2_mag_spect = np.asarray(rois2[file2][3])[0,:,:]

            # Calculate SSIM of FFTs
            avg_ssim, ssim_image = ssim(roi1_rot_mag_spect, roi2_mag_spect, full=True)
            ssim_value = ssim_image[n][n]

            # Use template matching to calculate normalized cross-correlation of FFTs
            template_matched = cv2.matchTemplate(roi1_rot_mag_spect, roi2_mag_spect, cv2.TM_CCORR_NORMED)
            nccorr_value = np.amax(template_matched)

            comparison_dict['%s_%s' % (file1, file2)][angle] = [ssim_value, nccorr_value, roi1_x, roi1_y, roi2_x, roi2_y]

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
    file1 = comparison[:comparison.find('_' + category2[:-1])]
    file2 = comparison[comparison.find(category2[:-1]):]

    with open(output_path, mode='a', newline='') as csvfile:
        outputwriter = csv.writer(csvfile)
        outputwriter.writerow([file1, file2,
                                max_dict[comparison][0], max_dict[comparison][1],
                                max_dict[comparison][2], max_dict[comparison][3],
                                max_dict[comparison][4], max_dict[comparison][5],
                                max_dict[comparison][6], max_dict[comparison][7]])

'''
# Define function to extract ROIs and FFTs
def readroifft(input_path, roi_dict):
    global frame_extension
    roi_fft_dict = {}
    with open(input_path, newline='') as csvfile:
        inputreader = csv.reader(csvfile)
        next(inputreader)
        for row in inputreader:
            roi_filename = row[0]
            roi_x, roi_y = roi_dict[roi_filename][0], roi_dict[roi_filename][1]
            img = cv2.imread(os.path.join(input_path, roi_filename, frame_extension))
            roi = roi(img, roi_x, roi_y)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_mag_spect = takedft(roi_gray)[2]
            roi_fft_dict[roi_filename] = [roi_gray, roi_mag_spect]
    return roi_fft_dict

# Show frame 1
img1 = cv2.imread('/Users/icunitz/Desktop/bat_detection/frames/final/bats/bat_111511_frame216.jpg')
cv2.imshow('Image', img1)
cv2.waitKey(0) & 0xFF

# Show frame 2
img2 = cv2.imread('/Users/icunitz/Desktop/bat_detection/frames/final/bats/bat_111511_frame125.jpg')
cv2.imshow('Image', img2)
cv2.waitKey(0) & 0xFF

    optimize.append([angle, ssim_value, nccorr_value])

    ## Show results

    #plt.figure(1)
    #plt.cla()

    #ax11 = plt.subplot(2, 4, 1)
    #ax11.imshow(roi1_rot_gray, cmap='gray')
    #ax11.set_xticks([])
    #ax11.set_yticks([])

    #ax12 = plt.subplot(2, 4, 2)
    #ax12.imshow(roi1_rot_mag_spect, cmap='gray')
    #ax12.set_xticks([])
    #ax12.set_yticks([])

    #ax13 = plt.subplot(2, 4, 3)
    #ax13.imshow(roi2_gray, cmap='gray')
    #ax13.set_xticks([])
    #ax13.set_yticks([])

    #ax14 = plt.subplot(2, 4, 4)
    #ax14.imshow(roi2_mag_spect, cmap='gray')
    #ax14.set_xticks([])
    #ax14.set_yticks([])

    #ax16 = plt.subplot(2, 4, 6)
    #ax16.cla()
    #ax16.text(0.25, 0.5, 'NCCORR = %s' % round(nccorr_value, 4))
    #ax16.set_xticks([])
    #ax16.set_yticks([])

    #ax18 = plt.subplot(2, 4, 8)
    #ax18.cla()
    #ax18.text(0.25, 0.5, 'SSIM = %s' % round(ssim_value, 4))
    #ax18.set_xticks([])
    #ax18.set_yticks([])

    #plt.pause(0.005)
    #plt.show()

    # cv2.imshow('Test', roi1_rot_gray)
    # cv2.imshow('Test FFT', roi1_rot_mag_spect)

    #if cv2.waitKey(50) & 0xFF == ord('q'):
    #    break

# Convert optimize list to np array and find angle with maximized SSIM
optimize = np.asarray(optimize)
max_index_s = np.where(optimize[:,1] == np.amax(optimize[:,1]))
max_angle_s = optimize[max_index_s, 0]
max_ssim_value = optimize[max_index_s, 1]

# Find angle with maximized nccorr
max_index_n = np.where(optimize[:,2] == np.amax(optimize[:,2]))
max_angle_n = optimize[max_index_n, 0]
max_nccorr_value = optimize[max_index_n, 2]

# Repeat earlier rotation, but for maximized SSIM only
roi1_rot_s = roi(rotate(img1, max_angle_s, x1, y1), x1, y1)
roi1_rot_gray_s = cv2.cvtColor(roi1_rot_s, cv2.COLOR_BGR2GRAY)
roi1_rot_mag_spect_s = takedft(roi1_rot_gray_s)[2]

# Calculate SSIM of FFTs
avg_ssim_s, ssim_image_s = ssim(roi1_rot_mag_spect_s, roi2_mag_spect, full=True)
ssim_value_s = ssim_image_s[n][n]

# Use template matching to calculate normalized cross-correlation of FFTs
template_matched_s = cv2.matchTemplate(roi1_rot_mag_spect_s, roi2_mag_spect, cv2.TM_CCORR_NORMED)
nccorr_value_s = np.amax(template_matched_s)

## Show figure with maxized SSIM
plt.figure(2)
plt.cla()

ax21 = plt.subplot(2, 4, 1)
ax21.imshow(roi1_rot_gray_s, cmap='gray')
ax21.set_xticks([])
ax21.set_yticks([])

ax22 = plt.subplot(2, 4, 2)
ax22.imshow(roi1_rot_mag_spect_s, cmap='gray')
ax22.set_xticks([])
ax22.set_yticks([])

ax23 = plt.subplot(2, 4, 3)
ax23.imshow(roi2_gray, cmap='gray')
ax23.set_xticks([])
ax23.set_yticks([])

ax24 = plt.subplot(2, 4, 4)
ax24.imshow(roi2_mag_spect, cmap='gray')
ax24.set_xticks([])
ax24.set_yticks([])

ax26 = plt.subplot(2, 4, 6)
ax26.cla()
ax26.text(0.25, 0.5, 'NCCORR = %s' % round(nccorr_value_s, 4))
ax26.set_xticks([])
ax26.set_yticks([])

ax28 = plt.subplot(2, 4, 8)
ax28.cla()
ax28.text(0.25, 0.5, 'SSIM = %s' % round(ssim_value_s, 4))
ax28.set_xticks([])
ax28.set_yticks([])

plt.show()

# Repeat earlier rotation, but for maximized nccorr only
roi1_rot_n = roi(rotate(img1, max_angle_n, x1, y1), x1, y1)
roi1_rot_gray_n = cv2.cvtColor(roi1_rot_n, cv2.COLOR_BGR2GRAY)
roi1_rot_mag_spect_n = takedft(roi1_rot_gray_n)[2]

# Calculate SSIM of FFTs
avg_ssim_n, ssim_image_n = ssim(roi1_rot_mag_spect_n, roi2_mag_spect, full=True)
ssim_value_n = ssim_image_n[n][n]

# Use template matching to calculate normalized cross-correlation of FFTs
template_matched_n = cv2.matchTemplate(roi1_rot_mag_spect_n, roi2_mag_spect, cv2.TM_CCORR_NORMED)
nccorr_value_n = np.amax(template_matched_n)

## Show figure with maxized nccorr
plt.figure(3)
plt.cla()

ax31 = plt.subplot(2, 4, 1)
ax31.imshow(roi1_rot_gray_n, cmap='gray')
ax31.set_xticks([])
ax31.set_yticks([])

ax32 = plt.subplot(2, 4, 2)
ax32.imshow(roi1_rot_mag_spect_n, cmap='gray')
ax32.set_xticks([])
ax32.set_yticks([])

ax33 = plt.subplot(2, 4, 3)
ax33.imshow(roi2_gray, cmap='gray')
ax33.set_xticks([])
ax33.set_yticks([])

ax34 = plt.subplot(2, 4, 4)
ax34.imshow(roi2_mag_spect, cmap='gray')
ax34.set_xticks([])
ax34.set_yticks([])

ax36 = plt.subplot(2, 4, 6)
ax36.cla()
ax36.text(0.25, 0.5, 'NCCORR = %s' % round(nccorr_value_n, 4))
ax36.set_xticks([])
ax36.set_yticks([])

ax38 = plt.subplot(2, 4, 8)
ax38.cla()
ax38.text(0.25, 0.5, 'SSIM = %s' % round(ssim_value_n, 4))
ax38.set_xticks([])
ax38.set_yticks([])

plt.show()

# # Take ROI 2 and FFT 2
# roi2 = roi(img2, x2, y2)
# roi2_gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
# roi2_mag_spect = takedft(roi2_gray)[2]

# Set up empty list to hold angle and SSIM values
'''