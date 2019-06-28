def rotate(image_name, a_found):
        global xi, yi, xf, yf
        
        Acute = True
        sign = 1

        rows, cols = image_name.shape[:2]
        
        a_rad = a_found * (np.pi / 180)
        angle = np.pi/2 + a_rad

        if yi < yf:
                Acute = False

        if ((yf - yi) / (xf - xi)) < 0:
                if not Acute:
                        sign = -1
                        angle = np.pi/2 - a_rad
        else:
                if Acute:
                        sign = -1
                        angle = np.pi/2 - a_rad

        print ('Sign: ' + str(sign))
        print ('Acute: ' + str(Acute))
        
        if Acute:
                r = int(rows*np.cos(angle) + cols*np.sin(angle))
                c = int(cols*np.cos(angle) + rows*np.sin(angle))
        else:
                r = int(cols*np.cos(np.pi/2 - angle) + rows*np.sin(np.pi/2 - angle))
                c = int(rows*np.cos(np.pi/2 - angle) + cols*np.sin(np.pi/2 - angle))

        M = cv2.getRotationMatrix2D((cols/2, rows/2), sign * angle, 1)
        M[0,2] += (c - cols) / 2
        M[1,2] += (r - rows) / 2
        
        return cv2.warpAffine(image_name, M, (c, r))
