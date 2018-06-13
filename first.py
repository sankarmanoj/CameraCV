import cv2
feed = cv2.VideoCapture("rtsp://192.168.0.204/11")
print "Opened Feed"
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# for x in range(100):
import numpy as np
previousFrame = None
count = 0
from collections import deque
pastDeltas = deque()
while True:
    _,frame = feed.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if previousFrame is not None:
        frameDelta = cv2.absdiff(previousFrame,gray)
        pastDeltas.append(frameDelta)
        if len(pastDeltas)>5:
            pastDeltas.popleft()
        # print sum(pastDeltas)
        thresh = cv2.threshold(sum(pastDeltas),10,255,cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh,None,iterations = 30)
        mean = np.mean(thresh)
        print mean
        if mean > 0.2:
            m2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # boxes = []
            for contour in contours:
                # if cv2.contourArea(contour)>500:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(frame,(x,y,x+w,y+h),(0.255,0),2)
            cv2.drawContours(frame,contours,-1,(0,255,0),3)
            cv2.imshow("Frame",frame)
        # cv2.imwrite(str(count)+".jpg",thresh)
        cv2.imshow("output",thresh)
    previousFrame = gray
    # laplacian = cv2.Laplacian(frame,cv2.CV_64F)
    cv2.waitKey(1)
cv2.destroyAllWindows()
