from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

from myblob import Point, Blob



def remove_noise(mask):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 50
    img = np.zeros((output.shape), np.uint8)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img[output == i + 1] = 255
    return img

def dilate_image(img):
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation = cv2.dilate(img,kernel_ellipse,iterations = 1)
    (ret, thresh3000) = cv2.threshold(dilation,127,255,0)
    return thresh3000

def params_for_blobs():
    params = cv2.SimpleBlobDetector_Params()
    # # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 255
    # # Filter by Area.
    params.filterByArea = True
    params.minArea = 54
    params.maxArea = 30000
    # # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.05
    # # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.10 #0.87
    params.maxConvexity = 1
    # # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    # # Filter by Color
    params.filterByColor = 1
    params.blobColor = 255
    return params

def start_stream():
    #Webcam stream
    #cap = cv2.VideoCapture(0)
    #Prepered videos: 1, 2, 3.mp4 
    cap = cv2.VideoCapture("1.mp4")
    time.sleep(0.5)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=12, detectShadows=False)
    myBlobs = []

    while True:
        #Read frame
        _ ,frame = cap.read()
        mask = subtractor.apply(frame)

        img3000 = remove_noise(mask)
        thresh3000 = dilate_image(img3000)
        params = params_for_blobs()
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh3000)
        im_with_keypoints = cv2.drawKeypoints(thresh3000, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #remoove not founded blobs 
        for blob in myBlobs:
            if not blob.founded:
                myBlobs.remove(blob)

        # # # BLOB ANALAJZER 9000 # # #
        print("nr of keys = ", len(keypoints))
        print("nr of blobs = ", len(myBlobs))

        index = 0
        for k in keypoints:
            xK, yK = k.pt
            #print("Point x ",xK, " y ",yK)

            anyBlobFounded = False
            index = index + 1
            for blob in myBlobs:                
                #debug lines and text
                cv2.line(frame,(int(blob.points[-1].x + blob.vectors[-1].x),int(blob.points[-1].y + blob.vectors[-1].y)),(int(blob.points[-1].x),int(blob.points[-1].y)),(0,255,255),2)

                # cv2.putText(frame, str(int(blob.distance(blob.xHistory[-1], blob.yHistory[-1], xK, yK))), (int((blob.xHistory[-1]+xK)/2), int((blob.yHistory[-1]+yK)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                # cv2.putText(frame, str(index), (int(xK), int(yK)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                blob.founded = False
                #print(xK, yK)
                if(blob.isNearToLast(Point(xK, yK))):
                    blob.newPosition(xK, yK)
                    anyBlobFounded = True
                    blob.founded = True
        
            if not anyBlobFounded:
                myBlobs.append(Blob(xK, yK))


        # # # KONIEC OF BLOB ANALAJZER 9000 # # #

        #lines
        #cv2.line(frame,(130,50),(1200,550),(0,255,0),1)
        #cv2.line(frame,(130,380),(1200,1400),(0,255,0),1)

        # # # SHOW IMAGES # # #
        windowSizeH = 640
        windowSizeW = 400
        #img3000 = cv2.resize(img3000, (windowSizeH, windowSizeW))
        #cv2.imshow("Conected components 3000", img3000)
        #thresh3000 = cv2.resize(thresh3000, (windowSizeH, windowSizeW))
        # cv2.imshow("Blobs 3000", thresh3000)
        frame = cv2.resize(frame, (windowSizeH, windowSizeW))
        cv2.imshow("Frame", frame)
        #mask = cv2.resize(mask, (windowSizeH, windowSizeW))
        #cv2.imshow("mask", mask)
        im_with_keypoints = cv2.resize(im_with_keypoints, (windowSizeH, windowSizeW))
        cv2.imshow("im_with_keypoints", im_with_keypoints)
        # # #  END OF SHOW IMAGES # # #

        if cv2.waitKey(33) == ord('a'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_stream()