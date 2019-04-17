from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

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
    '''
    #BLOBER DETECTOR 4000
    detector = cv2.SimpleBlobDetector()
 
    # Detect blobs.
    keypoints = detector.detect(img3000)
 
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img3000, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imshow(im_with_keypoints)
    #KONIEC BLOBERA DETECORA 4000
    '''
    '''
    kernel_square = np.ones((9,9),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))

    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation = cv2.dilate(mask,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation,5)
    ret,thresh = cv2.threshold(median,127,255,0)

    # thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
    #kernel = np.ones((1,1),np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #szukanie kontur
    thresh = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    '''
    '''
    # BLOBS
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()
    detector = cv2.SimpleBlobDetector_create()
    # Detect blobs.
    keypoints = detector.detect(thresh3000)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(thresh3000, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    '''  
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

    params.filterByColor = 1
    params.blobColor = 255
    
    # # Create a detector with the parameters
    # # ver = (cv2.__version__).split('.')
    # # if int(ver[0]) < 3 :
    # #     detector = cv2.SimpleBlobDetector(params)
    # # else : 
    detector = cv2.SimpleBlobDetector_create(params)
    
    keypoints = detector.detect(thresh3000)
    # # Draw detected blobs as red circles.
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(thresh3000, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("im_with_keypoints", im_with_keypoints)

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
        print(ratio)
        if ratio < 0.45: #0.4
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)        
        index=index+1
        print(x, y)

    #cv2.imshow("Blobs", thresh)
    #upperline
    cv2.line(frame,(130,50),(1200,550),(0,255,0),1)
    
    #downline
    cv2.line(frame,(130,380),(1200,1400),(0,255,0),1)
    cv2.imshow("Conected components 3000", img3000)
    cv2.imshow("Blobs 3000", thresh3000)
    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
    if cv2.waitKey(33) == ord('a'):
        break
cap.release()
#cap.stop()
cv2.destroyAllWindows()