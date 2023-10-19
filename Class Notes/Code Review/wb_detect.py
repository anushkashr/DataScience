import numpy as np
import cv2 

cam = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1): 
    ret, frame = cam.read()
    cv2.imshow('frame',frame) 
    fgmask = fgbg.apply(frame)
    cv2.imshow('fgmask', fgmask) 
    cv2.imshow('frame',frame ) 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

cap.release() 
cv2.destroyAllWindows()