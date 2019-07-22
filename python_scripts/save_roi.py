# Import useful libraries
import cv2
import numpy as np
import csv
import os

category = 'bats'

n = 20
s = n * 2 + 1 # Length of square sides
r = int((np.sqrt(2) / 2) * s) # Radius of circle shown

circlethickness = 2

## Set up folder structure and read/write locations
cwd = os.getcwd()
scriptfolder = 'python_scripts'
framefolder1 = 'frames'
framefolder2 = 'final'
homefolder = cwd[:-len(scriptfolder)]
readlocation = os.path.join(homefolder, framefolder1, framefolder2, category)

outputfolder1 = 'output'
outputfolder2 = 'final'
writelocation = os.path.join(homefolder, outputfolder1, outputfolder2)
outputfile = 'roi_locations_%s.csv' % category

for file in os.listdir(readlocation):
        readpath = os.path.join(readlocation, file)

        ref_location = [(40, 40)] # List to hold click locations, starting with generic location
        # Click event - to find region of interest
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                ref_location.append((x, y))

        img = cv2.imread(readpath)
        windowName = file

        # Show image and circle around clicked locations
        while True:
            # Show frame
            img_clone = img.copy()

            cv2.namedWindow(windowName)
            cv2.setMouseCallback(windowName, click_event)

            cv2.circle(img_clone, (ref_location[-1][0], ref_location[-1][1]), r, (0, 0, 0), circlethickness)
            cv2.imshow(windowName, img_clone)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save last clicked coordinates
        roi_x = ref_location[-1][0]
        roi_y = ref_location[-1][1]

        WriteFile = True

        ## Write results to csv output file
        if WriteFile:
                with open(os.path.join(writelocation, outputfile), mode='a', newline='') as csvfile:
                        outputwriter = csv.writer(csvfile)
                        outputwriter.writerow([file, roi_x, roi_y])
        else:
                print ('ROI coordinates not saved')

        print ('Saved ROI coordinates: %s' % file)




# # Define image rotating function
# def rotate(image_name, angle_deg, ref_location_list):
#     global s
# 
#     x = ref_location_list[-1][0]
#     y = ref_location_list[-1][1]
# 
#     M = cv2.getRotationMatrix2D((x, y), angle_deg, 1)
#     return cv2.warpAffine(image_name, M, (s, s))
# 
# #for angle in np.arange(0, 360, 4):
#    roi1_rotated = rotate(roi1, angle)
#    # take FFT of roi_rotated
#
