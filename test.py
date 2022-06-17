import numpy as np
import cv2
from sklearn.preprocessing import normalize
import os


def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        print('Distance: '+ str(Distance)+' m')

kernel= np.ones((3,3),np.uint8)

window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)
	
# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

path = r'C:\\Users\maksi\Downloads\1.jpg'
# ‪C:\Users\maksi\Downloads\1.jpg
frameR = cv2.imread(path)

rows,cols,_ = frameR.shape
M = np.float32([[1,0,100],[0,1,50]])
frameL = cv2.warpAffine(frameR,M,(cols,rows))


# frameL = cv2.imread(path)
grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
disp= stereo.compute(grayL,grayR)

dispL= disp
dispR= stereoR.compute(grayR,grayL)
dispL= np.int16(dispL)
dispR= np.int16(dispR)

filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
#cv2.imshow('Disparity Map', filteredImg)
disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp 

dispc= (disp-disp.min())*255
dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN)

cv2.imshow('Filtered Color Depth',filt_Color)
cv2.waitKey(10000)
