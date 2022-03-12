import math, yaml
import cv2
import glob
import yaml
import numpy as np
import imutils


def read_chessboard():
    if len(images) < 1:
        print("No image found")
        sys.exit()

    else:    
        for index in range(len(images)):
            horz1 = [[],[],[]]
            horz2 = [[],[],[]]

            fname = images[index]
            # Read the file and convert file
            img     = cv2.imread(fname)
            img     = ImageOperation.image_resize(img, height = 500)
            gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ImageOperation.field_contour(img.copy(), './Image/gray.jpg')
            gray_edit_color = cv2.imread('./Image/gray.jpg')
            gray_edit    = cv2.cvtColor(gray_edit_color,cv2.COLOR_BGR2GRAY)

            # Detect Chessboard corner
            print("Reading image ", fname)
            chess_corner = Chessboard.detect(gray_edit_color.copy(), gray.copy())
            new_chess_corner = Chessboard.find_edge(chess_corner,6,1)
            full_chess_corner = Chessboard.find_edge(new_chess_corner,8,1)
            extended_chess_corner = Chessboard.find_edge(new_chess_corner,8,1/8)

            # Define side and warp chessboard             
            warped_chessboard, test, warp_coor = Chessboard.define_side(full_chess_corner, extended_chess_corner, gray_edit.copy(), gray.copy())
            
            # Displaying the image 
            detected_chessboard = gray_edit_color.copy()
            for i in extended_chess_corner  :
                detected_chessboard = cv2.circle(detected_chessboard, (int(i[0]),int(i[1])), 3, (0,0,255), -1)
            # cv2.imshow('Detect Point', detected_chessboard)

            # Get the first frame
            if compare == False:
                prev_warped_chessboard = warped_chessboard.copy()
                split_cell = ChessboardCell.split(prev_warped_chessboard, 'cropped_1')
                # cv2.imshow('prev_split_cell', split_cell)
                compare = True

            # Compare current frame with previous frame
            elif compare == True:
                prev_split_cell = ChessboardCell.split(prev_warped_chessboard, 'cropped_1')
                # cv2.imshow('prev_split_cell', prev_split_cell)

                current_warped_chessboard = warped_chessboard.copy()
                current_split_cell = ChessboardCell.split(current_warped_chessboard, 'cropped_2')
                # cv2.imshow('current_split_cell', current_split_cell)

                ChessboardCell.compare('./Image/cropped_1', './Image/cropped_2')
                changed_cell_img, list_changed_cell = ChessboardCell.change_indicator()
                # cv2.imshow('changed_cell_img', changed_cell_img)
                
                prev_warped_chessboard = current_warped_chessboard.copy()
                all_cell_img, list_all_cell = ChessboardCell.detect_peice()

                horz1[0] = ImageOperation.image_resize(img, height = 250)
                horz2[0] = ImageOperation.image_resize(detected_chessboard, height = 250)
                

                horz1[1] = ImageOperation.image_resize(cv2.cvtColor(prev_split_cell, cv2.COLOR_GRAY2RGB), height = 250)
                horz2[1] = ImageOperation.image_resize(cv2.cvtColor(current_split_cell, cv2.COLOR_GRAY2RGB), height = 250)

                horz1[2] = ImageOperation.image_resize(cv2.cvtColor(changed_cell_img, cv2.COLOR_GRAY2RGB), height = 250)
                horz2[2] = ImageOperation.image_resize(cv2.cvtColor(all_cell_img, cv2.COLOR_GRAY2RGB), height = 250)
                
                row1 = np.concatenate(horz1, axis=1)
                row2 = np.concatenate(horz2, axis=1)

                final_img = np.concatenate([row1, row2], axis=0)
                cv2.imshow('final_img', final_img)

                to_cell, from_cell, added_cell, removed_cell = None, None, None, None

                if len(list_changed_cell) >= 2:
                    for changed_index in range(len(list_changed_cell)):
                        for all_index in range(len(list_all_cell)):
                            if list_changed_cell[changed_index] == list_all_cell[all_index]:
                                to_cell = list_changed_cell[changed_index]
                            else:
                                from_cell = list_changed_cell[changed_index]
                
                if len(list_changed_cell) == 1:
                    for all_index in range(len(list_all_cell)):
                        if list_changed_cell[0] == list_all_cell[all_index]:
                            added_cell = list_changed_cell[0]
                    if added_cell == None:
                        removed_cell = list_changed_cell[0]
                
                print("------------------------------------------------")
                print('changed cells:', list_changed_cell)
                print('all cells:', list_all_cell)
                print("------------------------------------------------")
                print('FROM:', from_cell)
                print('TO:', to_cell)
                print('ADDED:', added_cell)
                print('ROMOVED:', removed_cell)
                print("------------------------------------------------")
                            
            cv2.waitKey()    
            
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
    dst     = image_resize(dst, height = 640)
    cv2.imshow('dst',dst)
    cv2.waitKey()
    return dst   


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

def track_feature(gray):
    # Find the chess board corners
    corners = cv2.goodFeaturesToTrack(gray,81,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        chess_corner.append((x,y))
    #print(chess_corner)
    return chess_corner

def getHist(folname):
    workingFolder   = folname
    imageType       = 'jpg'
    filename    = workingFolder + "/*." + imageType
    images      = glob.glob(filename)

    histogram = []
    name = []
    # channels = [0, 1]
    # h_bins = 50
    # s_bins = 60
    # histSize = [h_bins, s_bins]
    # # hue varies from 0 to 179, saturation from 0 to 255
    # h_ranges = [0, 180]
    # s_ranges = [0, 256]
    # ranges = h_ranges + s_ranges # concat lists

    for fname in images:
        img     = cv2.imread(fname)
        gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # h = cv2.calcHist([gray], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        name.append(fname[len(fname)-6:len(fname)-4])
        histogram.append(h)
        
    return (histogram,name)
    
def compareHist(hist1, hist2, name):
    change = 0
    for i in range(len(hist1)):        
        plt.subplot(8, 8, 8*(8-(i//8)-1)+(i%8)+1)
        plt.plot(list(range(len(hist1)*4)), hist1[i],color='green')
        plt.ylabel(name[i])

        plt.twinx()
        plt.plot(list(range(len(hist2)*4)), hist2[i], color='blue')
        plt.ylabel(name[i])
    plt.show()


    for i in range(len(hist1)): 
        base_test = cv2.compareHist(hist1[i], hist2[i], 1)
        print(name[i], ' : ', base_test)
        if base_test >500:
            change += 1
    if change == 2:
        return False
    else:
        return True

def draw_line(img):
    d = 0
    for x in range(8):        
        c1 = ((x*80)+d,0)
        c2 = ((x*80)+d,640)
        cv2.line(img,c1,c2,(255,255,255),7)
    for y in range(8):        
        c1 = (0,(y*80)+d)
        c2 = (640,(y*80)+d)
        cv2.line(img,c1,c2,(255,255,255),7)
    return img

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
    
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    return cdst