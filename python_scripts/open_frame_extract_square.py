# Import useful libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set up read path
file = '2016-07-30_014634'
frameno = '54'
extension = '.jpg'
readpath = '/Users/icunitz/Desktop/bat_detection/frames/' + file + '/frame' + frameno + extension

# Set window names
frameTitle = 'Frame ' + frameno
window1Name = frameTitle + " Original"
window2Name = frameTitle + " Region of Interest"

n = 20 # Half of length of square sides

ref_location = [] # Empty list to hold click locations

# Read original image and create grayscale copy
img = cv2.imread(readpath)
clone = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Click event
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global ref_location
        print (x, y)
        ref_location.append((x, y))
        ## Get rectangle to work later
        #cv2.rectangle(img, (x - n, y - n), (x + n, y + n), (0, 0, 0), 2)

plt.close()

# Set up original image window and callback function
cv2.namedWindow(window1Name)
cv2.setMouseCallback(window1Name, click_event)

# Show original image
cv2.imshow(window1Name, img)
cv2.waitKey(0) & 0xFF

# Convert original image to grayscale to show later
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Alternate waitKey setup - seens to be no difference in use, investigate later
# while(1):
#         cv2.imshow(window1Name, img)
#         if cv2.waitKey(0) & 0xFF:
#                 # cv2.waitKey(1)
#                 # print (ref_location)
#                 break

# Record ROI coordinates and draw square around region of interest
roi_x = ref_location[-1][0]
roi_y = ref_location[-1][1]
cv2.rectangle(img, (roi_x - n, roi_y - n), (roi_x + n, roi_y + n), (0, 0, 0), 2)

cv2.imshow(window1Name, img)
cv2.waitKey(0) & 0xFF

# Crop image around last clicked location and show cropped image
print ('Location of Interest: (' + str(roi_x) + ', ' + str(roi_y) + ')')
roi = clone[(ref_location[-1][1] - n):(ref_location[-1][1] + n), (ref_location[-1][0] - n):(ref_location[-1][0] + n)]
cv2.imshow(window2Name, roi)
cv2.waitKey(0) & 0xFF

def dft(img_name):
        dft = cv2.dft(np.float32(roi), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return dft, dft_shift, magnitude_spectrum

# Take FFT and extract magnitude spectrum of cropped image
dft_roi = dft(roi)[0]
dft_shift_roi = dft(roi)[1]
magnitude_spectrum_roi = dft(roi)[2]

plt.figure(1)

plt.subplot(2, 2, 1)
plt.cla()
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.cla()
plt.imshow(roi, cmap='gray')
plt.title('Region of Interest (ROI)')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.cla()
plt.imshow(magnitude_spectrum_roi, cmap='gray')
plt.title('FFT Spectrum of ROI')
plt.xticks([])
plt.yticks([])

plt.show()

print ("Done")

cv2.destroyAllWindows()
