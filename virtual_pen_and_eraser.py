import numpy as np
import cv2
kernel=np.ones((3,3),np.uint8)
def nothing(x):
    pass
x1,y1=0,0
h=0
t=4
q=0
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
canvas=None
r,g,b=0,255,0

cv2.namedWindow('frame')
cv2.namedWindow('frame1')
cv2.createTrackbar("L - H", "frame1", 0, 179, nothing)
cv2.createTrackbar("L - S", "frame1", 0, 255, nothing)
cv2.createTrackbar("L - V", "frame1", 0, 255, nothing)
cv2.createTrackbar("U - H", "frame1", 179, 179, nothing)
cv2.createTrackbar("U - S", "frame1", 255, 255, nothing)
cv2.createTrackbar("U - V", "frame1", 255, 255, nothing)
 
while(False):

    ret, s = cap.read()
    
    l_h = cv2.getTrackbarPos("L - H", "frame1")
    l_s = cv2.getTrackbarPos("L - S", "frame1")
    l_v = cv2.getTrackbarPos("L - V", "frame1")
    u_h = cv2.getTrackbarPos("U - H", "frame1")
    u_s = cv2.getTrackbarPos("U - S", "frame1")
    u_v = cv2.getTrackbarPos("U - V", "frame1")
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    l_l=np.array([l_h,l_s,l_v])
    u_l=np.array([u_h,u_s,u_v])
    frame_threshold = cv2.inRange(hsv,l_l, u_l)
    
    

    
    cv2.imshow('frame',frame_threshold)
    cv2.imshow('frame1',s)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    if cv2.waitKey(20) == ord('s'):
        np.save('target',[l_l,u_l])
        break


cv2.destroyAllWindows()
l_l=np.array([19,83,219])
u_l=np.array([35,185,255])
while(True):
    
    ret, s = cap.read()
    s= cv2.flip(s, 1 )
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
 #   hsv=cv2.copyMakeBorder(hsv,10,10,10,10,cv2.BORDER_CONSTANT)
    mask = cv2.inRange(hsv,l_l, u_l)
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    c,h_q=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if canvas is None:
        canvas = np.zeros_like(s)
#cur from here        
    if c and cv2.contourArea(max(c,key=cv2.contourArea)) >1000:
        a=max(c,key=cv2.contourArea)
        x2,y2,w,h = cv2.boundingRect(a)
        if x1==0 :
            x1=x2
            y1=y2
            state=0
        else:
            if(q==1):
                canvas = cv2.line(canvas, (x1,y1),(x2,y2), [r,g,b], t)
            state=1
        x1,y1=x2,y2
    else:
        x1=0
        y1=0
        state=3

    print(x1,y1,x2,y2,state)
#    frame = cv2.add(s,canvas)
    frame=s
    cv2.rectangle(mask_3,(x2,y2),(x2+w,y2+h),(r,g,b),2)
    stacked = np.hstack((canvas,mask_3))
    cv2.imshow('frame',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    if cv2.waitKey(20) & 0xFF == ord('1'):
        r=244
        b=0
        g=0
        t=4
        q=1
    if cv2.waitKey(20) & 0xFF == ord('2'):
        r=0
        b=0
        g=244
        t=4
        q=1
    if cv2.waitKey(20) & 0xFF == ord('3'):
        r=0
        b=255
        g=0
        t=4
        q=1
    if cv2.waitKey(20) & 0xFF == ord('4'):
        r=0
        b=0
        g=0
        t=30
        q=1
    if cv2.waitKey(20) & 0xFF == ord('5'):
        q=0
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()