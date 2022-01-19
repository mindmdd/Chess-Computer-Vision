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



def main():
    compare = False
    if len(images) < 1:
        print("No image found")
        sys.exit()

    else:    
        for index in range(len(images)):
            fname = images[index]
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
            extended_chess_corner = utlis.find_edge(new_chess_corner,8,1/8)

            # Define side and warp chessboard             
            warped, test, warp_coor = utlis.define_side(full_chess_corner, extended_chess_corner, gray_edit.copy(), gray.copy())
            
        
            # Displaying the image 
            image = gray_edit_color.copy()
            for i in extended_chess_corner  :
                image = cv2.circle(image, (int(i[0]),int(i[1])), 3, (0,0,255), -1)
            cv2.imshow('Detect Point', image)

            if compare == False:
                warped1 = warped.copy()
                # Split image and add border
                split_cell = utlis.split_cell(warped1, 'cropped_1')
                cv2.imshow('split_cell_1', split_cell)
                compare = True
            elif compare == True:
                # Split image and add border
                split_cell_1 = utlis.split_cell(warped1, 'cropped_1')
                cv2.imshow('split_cell_1', split_cell_1)

                warped2 = warped.copy()
                # Split image and add border
                split_cell_2 = utlis.split_cell(warped2, 'cropped_2')
                cv2.imshow('split_cell_2', split_cell_2)

                compare_cell = utlis.changes()
                # cv2.imshow('compare_cell', compare_cell)
                indicated_cell = utlis.change_indicator()
                cv2.imshow('indicated_cell', indicated_cell)

                warped1 = warped2.copy()

            cv2.waitKey()    

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 