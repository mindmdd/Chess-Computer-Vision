import sys, time, math, yaml
import statistics
import os 
import cv2
import glob
import sys
import argparse
import yaml
import numpy as np


def distort(fname):
    #--- Get the camera calibration path
    with open(r'calibration_matrix.yaml') as file:
        documents = yaml.full_load(file)
        for item, doc in documents.items():
            #print(item, ":", doc)
            if item == 'camera_matrix':
                camera_matrix = np.array(doc)
            elif item == 'camera_distortion':
                camera_distortion = np.array(doc)

    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, (w,h), 1, (w,h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, camera_distortion, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    dst     = image_resize(dst, height = 500)
    cv2.imshow('dst',dst)
    cv2.waitKey()
    return dst

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def houghline(gray):
        dst = cv2.Canny(gray, 100, 250, None, 3)
    
        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        
        
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        
        return cdst

def sort_order(org_list):
        #print(approx)
    x_coor = []
    y_coor = []
    for i in org_list:
        #print(i)
        x_coor.append(i[0])
        y_coor.append(i[1])

    organized_list = []
    tempx = []
    tempy = []

    x = 0
    y = 0


    for n in range(len(x_coor)):
        min_diff = 100000
        
        for i in range(len(x_coor)):

            #print(x_coor[i], y_coor[i])
            if i != n:
                diff = abs(x_coor[i] - x) + abs(y_coor[i] - y) 
                if diff < min_diff:
                    nearest_x = x_coor[i]
                    nearest_y = y_coor[i]
                    nearest_coor = i
                    min_diff = diff
        #used.append(nearest_coor)
        del x_coor[nearest_coor]
        del y_coor[nearest_coor]
        if n!= 0:
            if min_diff <= 3:
                x = nearest_x
                y = nearest_y
                organized_list.append([x, y])
        else:
            x = nearest_x
            y = nearest_y
            organized_list.append([x, y])
    return organized_list

def detect_chessboard(img, gray):
    nRows = 6
    nCols = 6
    dimension = 20

    chess_list = []

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nRows*nCols,3), np.float32)
    objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # print("Pattern found! Press ESC to skip or ENTER to accept")
        #--- Sometimes, Harris cornes fails with crappy pictures, so
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nCols,nRows), corners2,ret)
        #print(corners2)
        # cv2.imshow('img',img)
        # cv2.waitKey()
        # print("Image accepted")
        objpoints.append(objp)
        imgpoints.append(corners2)

        # cv2.waitKey(0)
    else:
        print("Image declined")

    for i in range(nRows*nCols):
        chess_list.append([imgpoints[0][i][0][0], imgpoints[0][i][0][1]])
    #print(chess_list)

    return chess_list

def track_feature(gray):
    # Find the chess board corners
    corners = cv2.goodFeaturesToTrack(gray,81,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        chess_corner.append((x,y))
    #print(chess_corner)
    return chess_corner

def field_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #gray = cv2.equalizeHist(gray)
    #gray = cv2.medianBlur(gray,9)  

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,5))
    gray = clahe.apply(gray) 
    gray_cp = gray.copy() 
    gray_blur = cv2.medianBlur(gray_cp,7)
    gray = cv2.medianBlur(gray_cp,3)
    cv2.imwrite('data/gray.jpg', gray_blur)
    # cv2.imshow('gray',gray)
    # cv2.waitKey()

def get_extend_point(x1,y1,x2,y2, dist):
    

    if abs(x1-x2) != 0 and abs(x1-x2) > abs(y1-y2):
        m = (y1 - y2) / (x1-x2)
        c = y1 - (x1 * m)
        if x1>x2: 
            step = -1
        elif x1<x2: 
            step = 1
        if step == 1:
            y = int((m*(x2+dist)) + c)
            x = x2+dist
        else:                    
            y = int((m*(x2-dist)) + c)
            x = x2-dist

    elif abs(x1-x2) != 0 and abs(x1-x2) <  abs(y1-y2):
        m = (y1 - y2) / (x1-x2)
        c = y1 - (x1 * m)
        if y1>y2: 
            step = -1
        elif y1<y2: 
            step = 1
        if step == 1:
            x = int((y2+dist-c)/m)
            y = y2+dist
        else:    
            x = int((y2-dist-c)/m)
            y = y2-dist
    elif abs(x1-x2) != 0 and abs(x1-x2) ==  abs(y1-y2):
        m = (y1 - y2) / (x1-x2)
        c = y1 - (x1 * m)
        if y1>=y2: 
            step = -1
        elif y1<y2: 
            step = 1
        if step == 1:    
            x = int((y2+dist-c)/m)
            y = y2+dist
        else:    
            x = int((y2-dist-c)/m)
            y = y2-dist                            
    else:
        x = x2
        if y1>y2: 
            step = -1
        elif y1<y2: 
            step = 1
        if step == 1:
            y = y2+dist
        else:
            y = y2-dist
    return (x,y)

def find_edge(chess_corner,side, dist):
    temp_new_corner = []
    temp2_new_corner = []
    new_corner = []    
    for i in range(side**2):    
        if i%side == 0 and i == 0:
            x1 = chess_corner[i+1][0]
            y1 = chess_corner[i+1][1]
            x2 = chess_corner[i][0]
            y2 = chess_corner[i][1]
            point = get_extend_point(x1,y1,x2,y2,dist)
            temp_new_corner.append([point[0],point[1]])
            temp_new_corner.append([x2,y2])
        elif i%side == 0 and i != 0:   
            x1 = chess_corner[i-2][0]
            y1 = chess_corner[i-2][1]
            x2 = chess_corner[i-1][0]
            y2 = chess_corner[i-1][1]
            point = get_extend_point(x1,y1,x2,y2,dist)
            temp_new_corner.append([point[0],point[1]])

            x1 = chess_corner[i+1][0]
            y1 = chess_corner[i+1][1]
            x2 = chess_corner[i][0]
            y2 = chess_corner[i][1]
            
            point = get_extend_point(x1,y1,x2,y2,dist)
            temp_new_corner.append([point[0],point[1]])
            temp_new_corner.append([x2,y2])
        elif i == (side**2)-1:
            x1 = chess_corner[i-1][0]
            y1 = chess_corner[i-1][1]
            x2 = chess_corner[i][0]
            y2 = chess_corner[i][1]
            temp_new_corner.append([x2,y2])
            point = get_extend_point(x1,y1,x2,y2,dist)
            temp_new_corner.append([point[0],point[1]])
        else:
            temp_new_corner.append([chess_corner[i][0], chess_corner[i][1]])

    for i in range(len(temp_new_corner)):
        if i>=0 and i<=side+1:
            x1 = temp_new_corner[i+side+2][0]
            y1 = temp_new_corner[i+side+2][1]
            x2 = temp_new_corner[i][0]
            y2 = temp_new_corner[i][1]
            point = get_extend_point(x1,y1,x2,y2,dist)
            temp2_new_corner.append([point[0],point[1]])
        elif i == side+2:
            for i in temp2_new_corner:
                new_corner.append(i)
            for i in temp_new_corner:
                new_corner.append(i)
            temp2_new_corner = []
        elif i>=(side+2)*(side-1) and i<= (side+2)*(side-1)+(side+1):
            x1 = temp_new_corner[i-(side+2)][0]
            y1 = temp_new_corner[i-(side+2)][1]
            x2 = temp_new_corner[i][0]
            y2 = temp_new_corner[i][1]
            point = get_extend_point(x1,y1,x2,y2,dist)
            temp2_new_corner.append([point[0],point[1]])
            if i == (side+2)*(side-1)+(side+1):
                for i in temp2_new_corner:
                    new_corner.append(i)
    return new_corner
        
def define_side(full_chess_corner, chess_corner,img):
    sides = [[],[],[],[]]
    test = []
    color = [[],[],[],[]]

    for i in range(len(chess_corner)):
        if i%10 == 0:
            if i != 0 and i != 90 and i != 80:
                mid = ((chess_corner[i][0]+chess_corner[i+10][0])//2 , (chess_corner[i][1]+chess_corner[i+10][1])//2)
                sides[0].append(mid)
                
            if i != 90 and i != 0 and i != 10:
                mid = ((chess_corner[i-1][0]+chess_corner[i+9][0])/2 , (chess_corner[i-1][1]+chess_corner[i+9][1])//2)
                sides[2].append(mid)
               
        if i>= 1 and i<=7:
            mid = ((chess_corner[i][0]+chess_corner[i+1][0])//2 , (chess_corner[i][1]+chess_corner[i+1][1])//2)
            sides[1].append(mid)
            
        if i>= 91 and i<=97:
            mid = ((chess_corner[i][0]+chess_corner[i+1][0])//2 , (chess_corner[i][1]+chess_corner[i+1][1])//2)
            sides[3].append(mid)
            
    for coor in sides[0]:
        color[0].append(img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    for coor in sides[1]:
        color[1].append(img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    for coor in sides[2]:
        color[2].append(img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    for coor in sides[3]:
        color[3].append(img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    result = []
    check_w = 0
    check_b = 0
    for m in range(4):
        white = 0
        black = 0
        error = 0
        temp = []
        
        print(color[m])
        if m == 0:
            c1 = full_chess_corner[0]
            c2 = full_chess_corner[90]
        elif m == 1:
            c1 = full_chess_corner[9]
            c2 = full_chess_corner[0]
        elif m == 2:
            c1 = full_chess_corner[99]
            c2 = full_chess_corner[9]
        elif m == 3:
            c1 = full_chess_corner[90]
            c2 = full_chess_corner[99]

        #(full_chess_corner[0], full_chess_corner[90])
        for n in color[m]:
            if n > 190:
                temp.append(0)
                white += 1
            elif n < 50:
                temp.append(1)
                black += 1
            else:
                temp.append(-1)
                error -= 1
        if error < -2:
            result.append([0,0,'error'])
            print(m , 'error')
        else:            
            if black > 3 and white < 1:
                check_b += 1
                result.append([c1, c2, 'black', check_b])
                print(m , 'black')
            elif black < 1 and white > 3:
                check_w += 1
                result.append([c1, c2, 'white', check_w])
                print(m , 'white')
            else:
                result.append([c1, c2, 'checker'])
                print(m , 'checker')

    if check_b == 1 and check_w == 1:
        for i in range(len(result)):
            if result[i][2] == 'white':
                x1 = result[i][0][0]
                y1 = result[i][0][1]
                x2 = result[i][1][0]
                y2 = result[i][1][1]
            elif result[i][2] == 'black':
                x3 = result[i][0][0]
                y3 = result[i][0][1]
                x4 = result[i][1][0]
                y4 = result[i][1][1]
    elif check_b == 0 and check_w == 1:
        for i in range(len(result)):
            if result[i][2] == 'white':
                x1 = result[i][0][0]
                y1 = result[i][0][1]
                x2 = result[i][1][0]
                y2 = result[i][1][1]
                if i == 0:                    
                    x3 = result[2][0][0]
                    y3 = result[2][0][1]
                    x4 = result[2][1][0]
                    y4 = result[2][1][1]
                elif i == 1:                    
                    x3 = result[3][0][0]
                    y3 = result[3][0][1]
                    x4 = result[3][1][0]
                    y4 = result[3][1][1]
                elif i == 2:                    
                    x3 = result[0][0][0]
                    y3 = result[0][0][1]
                    x4 = result[0][1][0]
                    y4 = result[0][1][1]
                elif i == 3:                    
                    x3 = result[1][0][0]
                    y3 = result[1][0][1]
                    x4 = result[1][1][0]
                    y4 = result[1][1][1]
    elif check_b == 1 and check_w == 0:
        for i in range(len(result)):
            if result[i][2] == 'black':
                x3 = result[i][0][0]
                y3 = result[i][0][1]
                x4 = result[i][1][0]
                y4 = result[i][1][1]
                if i == 0:                    
                    x1 = result[2][0][0]
                    y1 = result[2][0][1]
                    x2 = result[2][1][0]
                    y2 = result[2][1][1]
                elif i == 1:                    
                    x1 = result[3][0][0]
                    y1 = result[3][0][1]
                    x2 = result[3][1][0]
                    y2 = result[3][1][1]
                elif i == 2:                    
                    x1 = result[0][0][0]
                    y1 = result[0][0][1]
                    x2 = result[0][1][0]
                    y2 = result[0][1][1]
                elif i == 3:                    
                    x1 = result[1][0][0]
                    y1 = result[1][0][1]
                    x2 = result[1][1][0]
                    y2 = result[1][1][1]
        
    w = 500
    src_pts = np.array([[x1, y1],
                        [x4, y4],
                        [x3, y3],
                        [x2, y2]], dtype="float32")
    dst_pts = np.array([[0, w],
                        [0, 0],
                        [w, 0],
                        [w, w]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, w))

    return warped

            


    









