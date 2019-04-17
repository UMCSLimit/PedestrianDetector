from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

#cap = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)
time.sleep(2.0)

#cap = cv2.VideoCapture("highway.mp4")

subtractor = cv2.createBackgroundSubtractorMOG2(history=360, varThreshold=12, detectShadows=False)
while True:
    #Read frame
    _ ,frame = cap.read()
    #sprawdzanie roznicy pomiedzy poprzednimi klatkami
    mask = subtractor.apply(frame)

    #thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
    #kernel = np.ones((1,1),np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #szukanie kontur
    thresh = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
    # if the contour is too small, ignore it
        carea = cv2.contourArea(c)
        if carea < 200 or carea > 20000:
            continue

		# compute the bounding box for the contour, draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)

        #solidy
        ratio = carea/(w*h)
        print(ratio)
        if ratio < 0.4:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print(x, y)

    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
    if cv2.waitKey(33) == ord('a'):
        break
#cap.release()
cap.stop()
cv2.destroyAllWindows()
