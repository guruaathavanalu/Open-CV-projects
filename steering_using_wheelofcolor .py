# -*- coding: utf-8 -*-

import numpy as np
import cv2
# This variable determines if we want to load color range from memory 
# or use the ones defined here. 

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Creating A 5x5 kernel for morphological operations
kernel = np.ones((5,5),np.uint8)

while(1):
    
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip( frame, 1 )

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges 
    # from there
                
    lower_range1  = np.array([24,72,60])
    upper_range1 = np.array([56,229,255])
    
    lower_range2  = np.array([96,121,134])
    upper_range2 = np.array([113,201,242])
    

    mask_y = cv2.inRange(hsv, lower_range1, upper_range1)
    mask_b = cv2.inRange(hsv, lower_range2, upper_range2)
    # Perform the morphological operations to get rid of the noise.
    # Erosion Eats away the white part while dilation expands it.
    mask_y= cv2.erode(mask_y,kernel,iterations = 1)
    mask_y = cv2.dilate(mask_y,kernel,iterations = 2)
    
    mask_b= cv2.erode(mask_b,kernel,iterations = 1)
    mask_b = cv2.dilate(mask_b,kernel,iterations = 2)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # stack all frames and show it
    stacked = np.hstack((mask_3,frame,res))
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()