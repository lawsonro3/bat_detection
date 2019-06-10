import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('/Users/icunitz/Desktop/bat_vids/2016-07-25_221513.avi')
success,image = vidcap.read()
count = 0
success = True
while success:
    cv2.imwrite("frame%d.jpg" % count, image)
    success, image = vidcap.read()
    print ('Read a new frame: ', success)
    count += 1
