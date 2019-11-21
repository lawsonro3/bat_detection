import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fftpack as fft
import imutils

directory='/Users/isabellecunitz/Desktop/bat_detection/bat_vids'

filename = 'bat_123245.avi' # bat, stationary blades
# filename = 'bird_185554.avi' # multiple birds, stationary blades
# filename = 'bird_418385.avi' # one bird, moving blade
# filename = 'bird_536808.avi' # raptor, stationary blades
# filename = '2016-07-30_014634.avi' # moving blades
# filanane = '2016-10-02_234015.avi' # moving clouds, slow-moving blades

name, separator, extension = filename.partition('.')
cap = cv2.VideoCapture(os.path.join(directory, filename))

output_directory = '/Users/isabellecunitz/Desktop/bat_detection/output'
output_filename = 'test_size.csv'
output_path = os.path.join(output_directory, output_filename)

fps = 30 # frames per second

#back_sub = cv2.bgsegm.createBackgroundSubtractorMOG(history = 200, nmixtures = 5)
back_sub = cv2.createBackgroundSubtractorMOG2(history = 15,varThreshold = 2000) ## Works for Bats
# back_sub = cv2.createBackgroundSubtractorMOG2(history = 15,varThreshold = 200) ## Works for Birds
#back_sub = cv2.createBackgroundSubtractorKNN(history = 40,dist2Threshold = 19000)
#back_sub = cv2.bgsegm.createBackgroundSubtractorGMG()

endtime = 4.8 # seconds
endframe = endtime * fps

frame_count = 0
contour_count = 0

frames = []
x_locs = []
y_locs = []
areas = []

while True:
    ret, frame = cap.read()

    if ret is False:
        break
    else:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        f, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #gray_gaussian = cv2.GaussianBlurr(gray, {5,5},0)
        back_sub_mask = back_sub.apply(frame)
        blur = cv2.medianBlur(back_sub_mask,5)

        contours = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(contours)
        
        for i in cnts:
            path = '/Users/jyarbrou/Computer_vision/Bat_project/bat_project/Bat_images/blade_not_rotating/frames'
            the_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            M = cv2.moments(i)
            if M["m00"] == 0:
                M["m00"] = 1
            cX = int(M["m10"]/M["m00"])
            cY = int((M["m01"]/M["m00"]))

            cv2.drawContours(frame, cnts, -1, (0, 255, 0), 2)

            area = cv2.contourArea(i)

            (x,y,w,h) = cv2.boundingRect(i)
            cv2.rectangle(frame, (cX - 50 , cY -50), (cX + 50, cY + 50), (0, 0, 0), 0)
            contour_count += 1
            roi = frame[y-50: y +50, x-50:x+50 ]

            if (area > 1.0) and (area < 500) and (cX, cY != 0,0): # filter out very small/large objects and objects at 0,0
                frames.append(frame_count)
                x_locs.append(cX)
                y_locs.append(cY)
                areas.append(area)
                print (cX,cY)
                with open('/Users/isabellecunitz/Desktop/bat_area_dataset.csv', mode='a', newline='') as csvfile:
                    outputwriter = csv.writer(csvfile)
                    outputwriter.writerow([filename, frame_count, cX, cY, area])

            #cv2.imwrite(os.path.join(path, filename + '_frame_number_' + str(int(the_frame)) + '_count_' + str(count) + '.jpg' ) , roi)

        cv2.imshow("bats",frame)
        cv2.imshow("back",back_sub_mask)
        cv2.imshow('blur', blur)

        frame_count +=1

        if cv2.waitKey(1) == ord('q'):
            break

        #if frame_count == endframe:
        #    break


cap.release()
cv2.destroyAllWindows()

frames = np.asarray(frames)
x_locs = np.asarray(x_locs)
y_locs = np.asarray(y_locs)
areas = np.asarray(areas)

times = frames / fps # time, seconds, since begnning of video (30 fps)

avg = np.average(areas)
areas_norm = areas - avg
areas_detrend = signal.detrend(areas_norm)

area_fft = fft.fft(areas_norm) # FFT of  areas
area_psd = np.abs(area_fft)**2 # Power spectrum
area_freq = fft.fftfreq(len(area_psd), 1.0/fps) # Set x axis to be in Hz
i = area_freq > 0

area_freq_plot = area_freq[i]
area_psd_plot = area_psd[i]

area_fft_norm = fft.fft(areas_norm) # FFT of normalized areas
area_psd_norm = np.abs(area_fft_norm)**2 # Power spectrum
area_freq_norm = fft.fftfreq(len(area_psd_norm), 1.0/fps) # Set x axis to be in Hz
i = area_freq_norm > 0

area_freq_plot_norm = area_freq_norm[i]
area_psd_plot_norm = area_psd_norm[i]


p = int(len(area_freq_plot)/40) # filter out initial peak

m = p + np.argmax(area_psd_plot[p:]) # find index of maximum frequency, not including initial peak
max_freq = area_freq_plot[m]

area_fft_bis = area_fft.copy()
area_fft_bis[np.abs(area_freq) > 3.1] = 0
area_clean = np.real(fft.ifft(area_fft_bis))

# Close previous windows
plt.close("all")

# Plot object area vs. time
fig, ax = plt.subplots()
ax.plot(times, areas)
ax.plot(times, areas_norm)
ax.plot(times, areas_detrend)
ax.plot(times, area_clean)
ax.set_title(filename + ", Area vs. Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Object Area (pixels)")
fig.show()

# Plot FFT
fig, ax = plt.subplots()
ax.plot(area_freq_plot, area_psd_plot)
ax.plot(area_freq_plot_norm, area_psd_plot_norm)
ax.axvline(x=max_freq, color='y')
ax.axvline(x=area_freq_plot[p], color='r')
ax.set_title(filename + ", PSD")
ax.set_xlabel("Frequency (Hz)")
fig.show()

#with open(output_path, mode='a', newline='') as csvfile:
#    outputwriter = csv.writer(csvfile)
#    outputwriter.writerow([filename, frame_count, cX, cY, area])
