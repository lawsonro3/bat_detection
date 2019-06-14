# Import useful libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os # Not currently using - figure out how to effectively use os.path.join

# Set up read paths
file = input('Video Name: ') # Try 2016-07-30_014634
frameno_withbat = input('Frame Number With Bat: ') # Try 65
frameno_wobat = input('Frame Number Without Bat: ') # Try 60
frameextension = '.jpg'
readpath_withbat = '/Users/icunitz/Desktop/bat_detection/frames/' + file + '/frame' + frameno_withbat + frameextension # Customize this based on directories in computer
readpath_wobat = '/Users/icunitz/Desktop/bat_detection/frames/' + file + '/frame' + frameno_wobat + frameextension

# Set window names
frameTitle_withbat = 'Frame ' + frameno_withbat
window1Name = frameTitle_withbat + ' (w/ Bat)'
window2Name = frameTitle_withbat + " Region of Interest"
frameTitle_wobat = 'Frame ' + frameno_wobat
window3Name = frameTitle_wobat + ' (w/o Bat)'

n = 20 # Half of length of square sides

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

plt.close() # Close any previous matplotlib.pyplot windows

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
roi_withbat = clone_withbat[(roi_y - n):(roi_y + n), (roi_x - n):(roi_x + n)]

# Read original image w/o bat and create grayscale copy
img_wobat = cv2.imread(readpath_wobat)
clone_wobat = cv2.cvtColor(img_wobat, cv2.COLOR_BGR2GRAY)

# Convert original image w/o bat to grayscale to show later
img_wobat = cv2.cvtColor(img_wobat, cv2.COLOR_BGR2GRAY)

# Draw square around region of interest in image w/o bat
cv2.rectangle(img_wobat, (roi_x - n, roi_y - n), (roi_x + n, roi_y + n), (0, 0, 0), 2)

# Show grayscale image w/o bat with square
cv2.imshow(window3Name, img_wobat)
cv2.waitKey(0) & 0xFF

# Crop image w/o bat around same location
roi_wobat = clone_wobat[(roi_y - n):(roi_y + n), (roi_x - n):(roi_x + n)]

## Show ROI in new window
# cv2.imshow(window2Name, roi_withbat)
# cv2.waitKey(0) & 0xFF

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

# Show results
titlefontsize = 12
subtitlefontsize = 10

plt.figure(1)
plt.suptitle(file + '; Region of Interest (ROI) Center: (%s, %s)' % (roi_x, roi_y), fontsize = titlefontsize)

plt.subplot(2, 3, 1)
plt.cla()
plt.imshow(img_withbat, cmap='gray')
plt.title(frameTitle_withbat + ' (w/ Bat)', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 2)
plt.cla()
plt.imshow(roi_withbat, cmap='gray')
plt.title('ROI w/ Bat', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 3)
plt.cla()
plt.imshow(mag_spect_roi_withbat, cmap='gray')
plt.title('FFT of ROI w/ Bat', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 4)
plt.cla()
plt.imshow(img_wobat, cmap='gray')
plt.title(frameTitle_wobat + ' (w/o Bat)', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 5)
plt.cla()
plt.imshow(roi_wobat, cmap='gray')
plt.title('ROI w/o Bat', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 6)
plt.cla()
plt.imshow(mag_spect_roi_wobat, cmap='gray')
plt.title('FFT of ROI w/o Bat', fontsize = subtitlefontsize)
plt.xticks([])
plt.yticks([])

plt.show()

print ("Done")