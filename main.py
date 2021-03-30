import numpy as np
import cv2
import glob
import argparse
import sys, time, math, yaml
import utlis
import statistics
import os 
from skimage.filters import threshold_local


# Set parameter for images files
workingFolder   = "./calibrate_folder_1"
imageType       = 'jpg'
filename    = workingFolder + "/*." + imageType
images      = glob.glob(filename)
#------------------------------------------

if len(images) < 1:
    print("No image found")
    sys.exit()

else:    
    for fname in images:
        # Read the file and convert file
        img     = cv2.imread(fname)
        img     = utlis.image_resize(img, height = 500)
        gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        utlis.field_contour(img.copy(), './data/gray.jpg')
        gray_edit_color = cv2.imread('./data/gray.jpg')
        gray_edit    = cv2.cvtColor(gray_edit_color,cv2.COLOR_BGR2GRAY)

        # Detect Chessboard corner
        print("Reading image ", fname)
        chess_corner = utlis.detect_chessboard(gray_edit_color.copy(), gray.copy())
        dist = math.sqrt((chess_corner[20][0]-chess_corner[21][0])**2 + (chess_corner[20][1]-chess_corner[21][1])**2)
        new_chess_corner = utlis.find_edge(chess_corner,6,1)
        full_chess_corner = utlis.find_edge(new_chess_corner,8,1)
        temp_chess_corner = utlis.find_edge(new_chess_corner,8,1/8)

        # Defineside and warp chessboard             
        warped, test, warp_coor = utlis.define_side(full_chess_corner, temp_chess_corner, gray_edit.copy(), gray.copy())

        # Draw line on chessboard
        cv2.imwrite('./data/warpped.jpg', warped) 
        warped = cv2.imread('./data/warpped.jpg')
        utlis.field_contour(warped, './data/warpped.jpg')
        warped = cv2.imread('./data/warpped.jpg')
    
        # Displaying the image 
        image = gray_edit_color.copy()
        for i in temp_chess_corner  :
            image = cv2.circle(image, (int(i[0]),int(i[1])), 3, (0,0,255), -1)
        
        withLine = utlis.houghline(warped.copy())
        withLine2 = utlis.draw_line(warped.copy())
        utlis.crop_img(withLine2)

        cv2.imshow('new', image)
        cv2.imshow('warp', warped)
        cv2.imshow('withLine', withLine2)
        cv2.waitKey()
                
cv2.destroyAllWindows()
