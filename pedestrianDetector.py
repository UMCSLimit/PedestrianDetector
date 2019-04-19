from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

import myblob

#Webcam stream
#cap = cv2.VideoCapture(0)

#Prepered videos: 1, 2, 3.mp4 
cap = cv2.VideoCapture("1.mp4")


time.sleep(0.5)
subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=12, detectShadows=False)

while True:
    #Read frame
    _ ,frame = cap.read()
    #sprawdzanie roznicy pomiedzy poprzednimi klatkami
    # kernel = np.ones((3,3),np.float32)/25
    # frame = cv2.filter2D(mask,-1,kernel)
    mask = subtractor.apply(frame)

    # mask = cv2.dilate(mask, None, iterations=2)
    # kernel = np.ones((3,3),np.float32)/25
    # mask = cv2.filter2D(mask,-1,kernel)

    #ODSZUMIACZ 3000
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 50

    #your answer image
    img3000 = np.zeros((output.shape), np.uint8)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img3000[output == i + 1] = 255

    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    dilation = cv2.dilate(img3000,kernel_ellipse,iterations = 1)

    (ret, thresh3000) = cv2.threshold(dilation,127,255,0)
    cnts = cv2.findContours(thresh3000, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #KONIEC ODSZUMIACZA 3000

    ### BLOB DETECTOR 3000 ###
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 255
    # # Filter by Area.
    params.filterByArea = True
    params.minArea = 54
    params.maxArea = 20000
    # # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
    # # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.15 #0.87
    params.maxConvexity = 1
    # # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    # # Filter by Color
    params.filterByColor = 1
    params.blobColor = 255
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh3000)
    # Draw detected blobs as red circles.
    im_with_keypoints = cv2.drawKeypoints(thresh3000, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # # KONIEC BLOB DETECTOR 3000 # # #

    # # # BLOB ANALAJZER 9000 # # #
    myBlobs = []
    for k in keypoints:
        x, y = k.pt
        newBlob = myblob.Blob(x, y)
        myBlobs.append(newBlob)
        print(newBlob.sredniaX(), "  ", newBlob.sredniaY())

    #if any(myBlobs):
        #print(myblob.Blob.srednia())

    # # # KONIEC OF BLOB ANALAJZER 9000 # # #

    # # # CONTURO-KRAJZERKA XD # # #
    index = 0
    for c in cnts:
    # if the contour is too small, ignore it
        carea = cv2.contourArea(c)
        
		# compute the bounding box for the contour, draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)

        if x < 300:
            if carea < 100 or carea > 15000:
                continue
        else:
            if carea < 254 or carea > 20000:
                continue
    
        #solidy
        ratio = carea/(w*h)
        if ratio < 0.45: #0.4
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)        
        index=index+1

        #print(x, y)
        #print(ratio)
    # # # KONIEC KONTURO-KRAJZERKI # # # 

    #upperline
    cv2.line(frame,(130,50),(1200,550),(0,255,0),1)
    
    #downline
    cv2.line(frame,(130,380),(1200,1400),(0,255,0),1)

    # # # SHOW IMAGES # # #
    windowSizeH = 640
    windowSizeW = 400
    img3000 = cv2.resize(img3000, (windowSizeH, windowSizeW))
    cv2.imshow("Conected components 3000", img3000)
    #thresh3000 = cv2.resize(thresh3000, (windowSizeH, windowSizeW))
    # cv2.imshow("Blobs 3000", thresh3000)
    frame = cv2.resize(frame, (windowSizeH, windowSizeW))
    cv2.imshow("Frame", frame)
    mask = cv2.resize(mask, (windowSizeH, windowSizeW))
    cv2.imshow("mask", mask)
    im_with_keypoints = cv2.resize(im_with_keypoints, (windowSizeH, windowSizeW))
    cv2.imshow("im_with_keypoints", im_with_keypoints)
    # # #  END OF SHOW IMAGES # # #

    if cv2.waitKey(33) == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()

