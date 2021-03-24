import numpy as np
import cv2
import glob
import argparse
import sys, time, math, yaml
import utlis
import statistics
import os 
from skimage.filters import threshold_local


#---------------------- SET THE PARAMETERS
workingFolder   = "./calibrate_folder_1"
imageType       = 'jpg'
#------------------------------------------

# Find the images files
filename    = workingFolder + "/*." + imageType
images      = glob.glob(filename)

print(len(images))

if len(images) < 1:
    print("Not enough images were found!!!")
    sys.exit()

else:    
    for fname in images:
        #-- Read the file and convert in greyscale
        img     = cv2.imread(fname)
        img     = utlis.image_resize(img, height = 800)
        gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        utlis.field_contour(img.copy())
        gray_edit_color = cv2.imread('./data/gray.jpg')
        gray_edit    = cv2.cvtColor(gray_edit_color,cv2.COLOR_BGR2GRAY)

        print("Reading image ", fname)
        chess_corner = utlis.detect_chessboard(gray_edit_color, gray)
        dist = math.sqrt((chess_corner[20][0]-chess_corner[21][0])**2 + (chess_corner[20][1]-chess_corner[21][1])**2)
        new_chess_corner = utlis.find_edge(chess_corner,6,dist)
        full_chess_corner = utlis.find_edge(new_chess_corner,8,dist)
        temp_chess_corner = utlis.find_edge(new_chess_corner,8,dist/4)              
        check_chess_corner = utlis.define_side(full_chess_corner, temp_chess_corner, gray_edit)
        # define_side(new_chess_corner2)

        image = gray_edit_color.copy()

        for i in check_chess_corner:
            image = cv2.circle(image, (int(i[0]),int(i[1])), 3, (0,0,255), -1)
        # Displaying the image 
        cv2.imshow('new', image)
        cv2.waitKey()
        
        # organized_list = utlis.sort_order(chess_corner)
        # for i in organized_list:
        #     cv2.circle(img,(x,y),3,(0,0,255),-1)
        #     cv2.imshow("Source", img)
        #     cv2.waitKey()

        # #utlis.houghline(gray)
                
cv2.destroyAllWindows()
