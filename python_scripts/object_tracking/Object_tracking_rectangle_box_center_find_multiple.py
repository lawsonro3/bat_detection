import cv2
import numpy as np
import os
print('WORKIND')

os.chdir('/Users/jyarbrou/Computer_vision/Bat_project/bat_project//Bat_images/mixed_targets')
current_directory = os.getcwd()
print(current_directory)
count = 0
# filename_0 = 'bat_111511.avi'
# filename_1 = 'bat_111603.avi'
# filename_2 = 'bat_112309.avi'
# filename_3 = 'bat_112899.avi'
# filename_4 = 'bat_112970.avi'
# filename_5 = 'bat_114125.avi'
# filename_6 = 'bat_114939.avi'
# filename_7 = 'bat_114972.avi'
# filename_8 = 'bat_115518.avi'
# filename_9 = 'bat_116562.avi'
for file in os.listdir(current_directory):
    if file.endswith(".avi"):
        original_filename = file
        print(original_filename)
        filename, separator, extension = original_filename.partition('.')
        cap = cv2.VideoCapture(file)
        #back_sub = cv2.bgsegm.createBackgroundSubtractorMOG(history = 200, nmixtures = 5)
        back_sub = cv2.createBackgroundSubtractorMOG2(history = 90,varThreshold = 6000)
        #back_sub = cv2.createBackgroundSubtractorKNN(history = 40,dist2Threshold = 19000)
        #back_sub = cv2.bgsegm.createBackgroundSubtractorGMG()

        while True:
            ret, frame = cap.read()
        #    ret, difference = cv2.threshold(back_sub,25,255,cv2.THRESH_BINARY)

            if ret is False:
                break
            else:
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                #f, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                #gray_gaussian = cv2.GaussianBlurr(gray, {5,5},0)
                #back_su_mask = back_sub.apply(frame)
                back_su_mask = back_sub.apply(frame)
                blur = cv2.medianBlur(back_su_mask,3)
            #    (_,contours,_) = cv2.findContours(back_su_mask_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                contours = cv2.findContours(blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                # for i in contours:
                #     contour_area = cv2.contourArea(i)
                #     if (contour_area > .5) and (contour_area <100):
                #         cv2.drawContours(frame,contours,-1,(0,255,0),2)

                
                for i in contours:
                    path = '/Users/jyarbrou/Computer_vision/Bat_project/bat_project/Bat_images/mixed_targets/mixed_targets_image'
                    the_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    contour_area = cv2.contourArea(i)
                    # M = cv2.moments(contour_area)
                    # cX = int(M["m10"]/M["m00"])
                    # cY = int((M["m01"]/M["m00"]))
                    # print(M)

                    (x,y,w,h) = cv2.boundingRect(i)
                    cv2.rectangle(frame, (x - 50, y - 50), (x + 50, y + 50), (0, 0, 0), 0)
                    count = count + 1
                    roi = frame[y-50: y +50, x-50:x+50 ]
                    #cv2.imwrite(filename + '_frame_number_' + str(int(the_frame)) +'.jpg', roi)
                    #cv2.imwrite(os.path.join(path, filename + '_frame_number_' + str(int(the_frame)) + '_count_' + str(count) + '.jpg' ) , roi)
            
                    


                cv2.imshow("bats",frame)
            #    cv2.imshow("gray", gray)
                #cv2.imshow("thresh",threshold)
            #    cv2.imshow('back_mog',back_su_mask)
                #cv2.imshow("back",back_su_mask)
                #cv2.imshow('blur', blur)
            #    cv2.imshow("back_3",back_su_mask_4)
                #cv2.imshow("back_G",back_gaussian)
                if cv2.waitKey(1) == ord('q'):
                    break
