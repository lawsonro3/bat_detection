import cv2
import numpy as np
import os
import imutils
print('WORKIND')

current_directory = '/Users/isabellecunitz/Desktop/bat_detection/bat_vids'
print(current_directory)
count = 0
filename = 'bat_123245.avi'
# for file in os.listdir(current_directory):
#    if file.endswith(".avi"):

original_filename = filename
print(original_filename)
filename, separator, extension = original_filename.partition('.')
cap = cv2.VideoCapture(os.path.join(current_directory, file))
#back_sub = cv2.bgsegm.createBackgroundSubtractorMOG(history = 200, nmixtures = 5)

back_sub = cv2.createBackgroundSubtractorMOG2(history = 15,varThreshold = 1750) ## Works for Bats
#back_sub = cv2.createBackgroundSubtractorMOG2(history = 15,varThreshold = 200) ## Works for Birds

#back_sub = cv2.createBackgroundSubtractorKNN(history = 40,dist2Threshold = 19000)
#back_sub = cv2.bgsegm.createBackgroundSubtractorGMG()

while True:
    ret, frame = cap.read()

    if ret is False:
        break
    else:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #f, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #gray_gaussian = cv2.GaussianBlurr(gray, {5,5},0)
        #back_su_mask = back_sub.apply(frame)
        back_su_mask = back_sub.apply(frame)
        blur = cv2.medianBlur(back_su_mask,7)

        #(_,contours,_) = cv2.findContours(blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = cv2.findContours(back_su_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts=imutils.grab_contours(contours)

        for i in cnts:
            # i = np.asarray(i)
            if type(i) is not np.ndarray:
                print(type(i))
            #path = '/Users/jyarbrou/Computer_vision/Bat_project/bat_project/Bat_images/mixed_targets/mixed_targets_image'
            #the_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            #contour_area = cv2.contourArea(i)
    

            x,y,w,h = cv2.boundingRect(i)
            print (x, y, w, h)
            cv2.rectangle(frame, (x - 50, y - 50), (x + 50, y + 50), (0, 0, 0), 0)
            count = count + 1
            roi = frame[y-50: y +50, x-50:x+50 ]
            
    

        cv2.imshow("bats",frame)
        #cv2.imshow("blur",blur)
        if cv2.waitKey(1) == ord('q'):
            break
