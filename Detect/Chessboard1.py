import math
import cv2
import numpy as np
import Calculation

def detect_center_chessboard(real_gray, color_gray):
    gray = cv2.equalizeHist(real_gray.copy())  
    nRows = 6
    nCols = 6
    dimension = 25

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
        # cv2.drawChessboardCorners(img, (nCols,nRows), corners2,ret)

        # cv2.imshow('Detect chessboard',img)
        # cv2.waitKey()
        objpoints.append(objp)
        imgpoints.append(corners2)

        for i in range(nRows*nCols):
            chess_list.append([imgpoints[0][i][0][0], imgpoints[0][i][0][1]])

    else:
        print("Image declined")
        # new = np.float32(gray.copy())
        # dst = cv2.cornerHarris(new,2,3,0.04)
        # #result is dilated for marking the corners, not important
        # dst = cv2.dilate(dst,None)
        # # Threshold for an optimal value, it may vary depending on the image.
        # gray_edit_color = cv2.imread('./Image/gray.jpg')
        # gray_edit_color[dst>0.01*dst.max()]=[0,0,255]
        # cv2.imshow('harris', gray_edit_color)

        

    return chess_list

def ccget_extend_point(x1,y1,x2,y2, xs, ys, ratio):
    m, c = Calculation.best_fit_slope_and_intercept(np.array(xs),np.array(ys))
    # print('xx', m,c)
    dist_list = []
    for i in range(len(xs)-1):
        dist_list.append(math.sqrt(math.pow((xs[i]-xs[i+1]), 2) + math.pow((ys[i]-ys[i+1]), 2)))
    avg_dist = np.average(np.array(dist_list))*ratio

    if abs(x1-x2) != 0 and abs(x1-x2) > abs(y1-y2): 
        if x1>x2: 
            step = -1
        elif x1<x2: 
            step = 1

        start = [x2,(m*x2) + c,0]

        new_x = x2+(step*avg_dist)
        end = [new_x,(m*new_x) + c,0]

        x, y ,z = Calculation.point_on_line_from_distance(start, end, avg_dist)
    
    elif abs(x1-x2) != 0 and abs(x1-x2) <= abs(y1-y2): 
        if y1>y2: 
            step = -1
        elif y1<y2: 
            step = 1

        start = [(y2-c)/m,y2 ,0]

        new_y = y2+(step*avg_dist)
        end = [(new_y-c)/m, new_y, 0]

        x, y ,z = Calculation.point_on_line_from_distance(start, end, avg_dist)
                                 
    else:
        x = x2
        if y1>y2: 
            step = -1
        elif y1<y2: 
            step = 1
        if step == 1:
            y = y2+avg_dist
        else:
            y = y2-avg_dist
    return (x,y)

def xxget_extend_point(x1,y1,x2,y2, xs, ys, ratio):
    m, c = Calculation.best_fit_slope_and_intercept(np.array(xs),np.array(ys))
    # print('xx', m,c)
    dist_list = []
    for i in range(len(xs)-1):
        dist_list.append(math.sqrt(math.pow((xs[i]-xs[i+1]), 2) + math.pow((ys[i]-ys[i+1]), 2)))
    avg_dist = np.average(np.array(dist_list))*ratio

    x_list = []
    y_list = []
    for i in range(1, len(xs)):
        new_dist =  avg_dist*(len(ys)-i-1) + avg_dist*ratio

        if abs(xs[i-1]-xs[i]) > 1 and abs(xs[i-1]-xs[i]) >= abs(ys[i-1]-ys[i]): 
            if xs[i-1]>xs[i]: 
                step = -1
            elif xs[i-1]<xs[i]: 
                step = 1

            start = [xs[i], (m*xs[i])+c, 0]
            new_x = xs[i]+(step*60)
            end = [new_x, (m*new_x)+c, 0]

            x, y ,z = Calculation.point_on_line_from_distance(start, end, new_dist)
            # print('aa', x, y)
        
        elif abs(xs[i-1]-xs[i]) > 1 and abs(xs[i-1]-xs[i]) < abs(ys[i-1]-ys[i]): 
            
            if ys[i-1]>ys[i]: 
                step = -1
            elif ys[i-1]<=ys[i]: 
                step = 1

            start = [(ys[i]-c)/m, ys[i], 0]
            new_y = ys[i]+(step*60)
            end = [(new_y-c)/m, new_y, 0]

            x, y ,z = Calculation.point_on_line_from_distance(start, end, new_dist)
            # print('bb', x, y)

        else:
            x = xs[i]
            if ys[i-1]>ys[i]: 
                step = -1
            elif ys[i-1]<ys[i]: 
                step = 1

            y = ys[i]+(step*new_dist)
            # print('cc', x, y)
            
        x_list.append(x)
        y_list.append(y)      

    avg_x = np.median(np.array(x_list))
    avg_y = np.median(np.array(y_list))
             
    
    return ( avg_x, avg_y )
    
def get_extend_point(x1,y1,x2,y2,xs, ys, ratio):
    lm = []
    lc = []
    for i in range(len(xs)):
        for j in range(len(xs)):
            if j > i:
                if xs[i] - xs[j] != 0:
                    lx = [xs[i], xs[j]]
                    ly = [ys[i], ys[j]]

                    if xs[i]-xs[j] != 0:    
                        m, c = Calculation.best_fit_slope_and_intercept(np.array(lx),np.array(ly))
                        
                        lm.append(m)
                        lc.append(c)

    # print(lm, '-----', lc)
    m = np.average(np.array(lm))
    c = np.average(np.array(lc))
    # print(m,c)

    dist_list = []
    for i in range(len(xs)-1):
        dist_list.append(math.sqrt(math.pow((xs[i]-xs[i+1]), 2) + math.pow((ys[i]-ys[i+1]), 2)))
        
    avg_dist = np.average(np.array(dist_list))

    x_list = []
    y_list = []
    for i in range(1, len(xs)):
        new_dist =  avg_dist*(len(ys)-i-1) + avg_dist*ratio

        if abs(xs[i-1]-xs[i]) > 1 and abs(xs[i-1]-xs[i]) >= abs(ys[i-1]-ys[i]): 
            if xs[i-1]>xs[i]: 
                step = -1
            elif xs[i-1]<xs[i]: 
                step = 1

            start = [xs[i], (m*xs[i])+c, 0]
            new_x = xs[i]+(step*60)
            end = [new_x, (m*new_x)+c, 0]

            x, y ,z = Calculation.point_on_line_from_distance(start, end, new_dist)
            # print('aa', x, y)
        
        elif abs(xs[i-1]-xs[i]) > 1 and abs(xs[i-1]-xs[i]) < abs(ys[i-1]-ys[i]): 
            
            if ys[i-1]>ys[i]: 
                step = -1
            elif ys[i-1]<=ys[i]: 
                step = 1

            start = [(ys[i]-c)/m, ys[i], 0]
            new_y = ys[i]+(step*60)
            end = [(new_y-c)/m, new_y, 0]

            x, y ,z = Calculation.point_on_line_from_distance(start, end, new_dist)
            # print('bb', x, y)

        else:
            x = xs[i]
            if ys[i-1]>ys[i]: 
                step = -1
            elif ys[i-1]<ys[i]: 
                step = 1

            y = ys[i]+(step*new_dist)
            # print('cc', x, y)
            
        x_list.append(x)
        y_list.append(y)      

    avg_x = np.median(np.array(x_list))
    avg_y = np.median(np.array(y_list))

    # print(m,c)
    # print(x_list)
    # print(y_list)
    # print(ans_x, avg_x, ans_x - avg_x)
    # print(ans_y, avg_y, ans_y - avg_y)

    # avg_x = min(x_list)
    # avg_y = min(y_list)
                  
    
    return ( avg_x, avg_y )

def find_edge(chess_corner,side, ratio):
    temp_new_corner = []
    temp2_new_corner = []
    new_corner = []    
    for i in range(side**2):   
        if i%side == 0 and i == 0: #END of row
            # print("IN1")
            x1 = chess_corner[i+1][0]
            y1 = chess_corner[i+1][1]
            x2 = chess_corner[i][0]
            y2 = chess_corner[i][1]
            xs = []
            ys = []
            for n in range(i,i+side):
                xs.append(chess_corner[n][0])
                ys.append(chess_corner[n][1])
            point = get_extend_point(x1,y1,x2,y2,xs[::-1],ys[::-1],ratio)
            temp_new_corner.append([point[0],point[1]])
            temp_new_corner.append([x2,y2])
        
        elif i%side == 0 and i != 0:  # BEGINNING of row
            # print("IN2", i)
            x1 = chess_corner[i-2][0]
            y1 = chess_corner[i-2][1]
            x2 = chess_corner[i-1][0]
            y2 = chess_corner[i-1][1]
            xs = []
            ys = []
            for n in range(i-side, i):
                # print(n)
                xs.append(chess_corner[n][0])
                ys.append(chess_corner[n][1])
            point = get_extend_point(x1,y1,x2,y2,xs,ys,ratio)
            temp_new_corner.append([point[0],point[1]])

            # print("IN3")
            x1 = chess_corner[i+1][0]
            y1 = chess_corner[i+1][1]
            x2 = chess_corner[i][0]
            y2 = chess_corner[i][1]
            xs = []
            ys = []
            for n in range(i, i+side):
                xs.append(chess_corner[n][0])
                ys.append(chess_corner[n][1])

            point = get_extend_point(x1,y1,x2,y2,xs[::-1],ys[::-1],ratio)
            temp_new_corner.append([point[0],point[1]])
            temp_new_corner.append([x2,y2])

        elif i == (side**2)-1: #LAST
            # print("IN4", i)
            x1 = chess_corner[i-1][0]
            y1 = chess_corner[i-1][1]
            x2 = chess_corner[i][0]
            y2 = chess_corner[i][1]
            xs = []
            ys = []
            for n in range(i-side+1, i+1):
                # print(n)
                xs.append(chess_corner[n][0])
                ys.append(chess_corner[n][1])
            temp_new_corner.append([x2,y2])
            point = get_extend_point(x1,y1,x2,y2,xs,ys,ratio)
            temp_new_corner.append([point[0],point[1]])
        else:
            temp_new_corner.append([chess_corner[i][0], chess_corner[i][1]])

    for i in range(len(temp_new_corner)):
        if i>=0 and i<=side+1:
            # print("IN5", len(temp_new_corner))
            x1 = temp_new_corner[i+side+2][0]
            y1 = temp_new_corner[i+side+2][1]
            x2 = temp_new_corner[i][0]
            y2 = temp_new_corner[i][1]
            xs = []
            ys = []
            for n in range(i, len(temp_new_corner), side+2):
                # print(n)
                xs.append(temp_new_corner[n][0])
                ys.append(temp_new_corner[n][1])
            point = get_extend_point(x1,y1,x2,y2,xs[::-1],ys[::-1],ratio)
            temp2_new_corner.append([point[0],point[1]])
        elif i == side+2:
            for i in temp2_new_corner:
                new_corner.append(i)
            for i in temp_new_corner:
                new_corner.append(i)
            temp2_new_corner = []
        elif i>=(side+2)*(side-1) and i<= (side+2)*(side-1)+(side+1):
            # print("IN6")
            x1 = temp_new_corner[i-(side+2)][0]
            y1 = temp_new_corner[i-(side+2)][1]
            x2 = temp_new_corner[i][0]
            y2 = temp_new_corner[i][1]
            xs = []
            ys = []
            for n in range(i,-1,-(side+2)):
                xs.append(temp_new_corner[n][0])
                ys.append(temp_new_corner[n][1])
            point = get_extend_point(x1,y1,x2,y2,xs[::-1],ys[::-1],ratio)
            temp2_new_corner.append([point[0],point[1]])
            if i == (side+2)*(side-1)+(side+1):
                for i in temp2_new_corner:
                    new_corner.append(i)
    return new_corner
        
def define_side(full_chess_corner, chess_corner,edited_img, img):
    
    final_img = edited_img.copy()
    final_img = cv2.equalizeHist(final_img)
    sides = [[],[],[],[]]
    test = []
    color = [[],[],[],[]]
    gray = [[],[],[],[]]

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
        gray[0].append(final_img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    for coor in sides[1]:
        color[1].append(img[int(coor[1]), int(coor[0])])
        gray[1].append(final_img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    for coor in sides[2]:
        color[2].append(img[int(coor[1]), int(coor[0])])
        gray[2].append(final_img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    for coor in sides[3]:
        color[3].append(img[int(coor[1]), int(coor[0])])
        gray[3].append(final_img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])

    result = []
    check_w = 0
    check_b = 0
    check_e = 0
    for m in range(4):
        white = 0
        black = 0
        error = 0
        temp = []
        
        # print(gray[m], color[m])s

        #(full_chess_corner[0], full_chess_corner[90])
        for n in range(len(gray[m])):
            if gray[m][n] > 200:
                temp.append(0)
                white += 1
            elif gray[m][n] < 80:
                temp.append(1)
                black += 1
            else:
                temp.append(-1)
                error -= 1
        
        if error < -2:
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
            check_e += 1
            result.append([c1, c2,'error', m, check_e])
            print(m , 'error')
        else:
            if black > 3 and white <= 1:
                if m == 0:
                    c1 = full_chess_corner[1]
                    c2 = full_chess_corner[91]
                elif m == 1:
                    c1 = full_chess_corner[19]
                    c2 = full_chess_corner[10]
                elif m == 2:
                    c1 = full_chess_corner[98]
                    c2 = full_chess_corner[8]
                elif m == 3:
                    c1 = full_chess_corner[80]
                    c2 = full_chess_corner[89]  

                check_b += 1  
                result.append([c1, c2, 'black', m, check_b])
                print(m , 'black')
            elif black <= 1 and white > 3:
                if m == 0:
                    c1 = full_chess_corner[1]
                    c2 = full_chess_corner[91]
                elif m == 1:
                    c1 = full_chess_corner[19]
                    c2 = full_chess_corner[10]
                elif m == 2:
                    c1 = full_chess_corner[98]
                    c2 = full_chess_corner[8]
                elif m == 3:
                    c1 = full_chess_corner[80]
                    c2 = full_chess_corner[89]  
                    
                check_w += 1
                result.append([c1, c2, 'white', m, check_w])
                print(m , 'white')
            else:
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
                result.append([c1, c2, 'checker', m])
                print(m , 'checker')

    if check_b == 1 and check_w == 1:
        for i in range(len(result)):
            if result[i][2] == 'white':
                get_side1 = result[i][3]
                x1 = result[i][0][0]
                y1 = result[i][0][1]
                x2 = result[i][1][0]
                y2 = result[i][1][1]
            elif result[i][2] == 'black':
                get_side2 = result[i][3]
                x3 = result[i][0][0]
                y3 = result[i][0][1]
                x4 = result[i][1][0]
                y4 = result[i][1][1]
    elif check_b != 1 and check_w == 1:
        for i in range(len(result)):
            if result[i][2] == 'white':
                get_side1 = result[i][3]
                x1 = result[i][0][0]
                y1 = result[i][0][1]
                x2 = result[i][1][0]
                y2 = result[i][1][1]
                if i == 0:
                    get_side2 = result[2][3]                    
                    x3 = result[2][0][0]
                    y3 = result[2][0][1]
                    x4 = result[2][1][0]
                    y4 = result[2][1][1]
                elif i == 1:   
                    get_side2 = result[3][3]                  
                    x3 = result[3][0][0]
                    y3 = result[3][0][1]
                    x4 = result[3][1][0]
                    y4 = result[3][1][1]
                elif i == 2: 
                    get_side2 = result[0][3]                    
                    x3 = result[0][0][0]
                    y3 = result[0][0][1]
                    x4 = result[0][1][0]
                    y4 = result[0][1][1]
                elif i == 3:   
                    get_side2 = result[1][3]                  
                    x3 = result[1][0][0]
                    y3 = result[1][0][1]
                    x4 = result[1][1][0]
                    y4 = result[1][1][1]
    elif check_b == 1 and check_w != 1:
        for i in range(len(result)):
            if result[i][2] == 'black':
                get_side2 = result[i][3] 
                x3 = result[i][0][0]
                y3 = result[i][0][1]
                x4 = result[i][1][0]
                y4 = result[i][1][1]
                if i == 0:     
                    get_side1 = result[2][3]                
                    x1 = result[2][0][0]
                    y1 = result[2][0][1]
                    x2 = result[2][1][0]
                    y2 = result[2][1][1]
                elif i == 1: 
                    get_side1 = result[3][3]                    
                    x1 = result[3][0][0]
                    y1 = result[3][0][1]
                    x2 = result[3][1][0]
                    y2 = result[3][1][1]
                elif i == 2:   
                    get_side1 = result[0][3]                  
                    x1 = result[0][0][0]
                    y1 = result[0][0][1]
                    x2 = result[0][1][0]
                    y2 = result[0][1][1]
                elif i == 3:   
                    get_side1 = result[1][3]                  
                    x1 = result[1][0][0]
                    y1 = result[1][0][1]
                    x2 = result[1][1][0]
                    y2 = result[1][1][1]
    
    final_coor = [[x1, y1], [x4, y4], [x3, y3], [x2, y2]]

    if check_e == 1:
        for i in range(len(result)):
            if result[i][2] == 'error':
                error_side = result[i][3]
                error_index = i
            if result[i][2] == 'black' or result[i][2] == 'white':
                main_side = result[i][3]

        if main_side == 0:
            main_coor = [full_chess_corner[1], full_chess_corner[91]]
            if error_index == 3: 
                new_coor = [full_chess_corner[1], full_chess_corner[81]]
            if error_index == 1:
                new_coor = [full_chess_corner[11], full_chess_corner[91]]
        elif main_side == 1:
            main_coor = [full_chess_corner[19], full_chess_corner[10]]
            if error_index == 0:
                new_coor = [full_chess_corner[19], full_chess_corner[11]]
            if error_index == 2:
                new_coor = [full_chess_corner[18], full_chess_corner[10]]
        elif main_side == 2:
            main_coor = [full_chess_corner[98], full_chess_corner[8]]
            if error_index == 1:
                new_coor = [full_chess_corner[98], full_chess_corner[18]]
            if error_index == 3:
                new_coor = [full_chess_corner[88], full_chess_corner[8]]
        elif main_side == 3:
            main_coor = [full_chess_corner[80], full_chess_corner[89]]
            if error_index == 0:
                new_coor = [full_chess_corner[79], full_chess_corner[89]]
            if error_index == 2:
                new_coor = [full_chess_corner[80], full_chess_corner[88]]
        
        for i in range(len(final_coor)):
            if final_coor[i] == main_coor[0]:
                final_coor[i] = new_coor[0]
            elif final_coor[i] == main_coor[1]:
                final_coor[i] = new_coor[1]

            if final_coor[i] == result[error_index][0]:
                if result[error_index][3] == 0:
                    final_coor[i] = full_chess_corner[1]
                elif result[error_index][3] == 1:
                    final_coor[i] = full_chess_corner[19]
                elif result[error_index][3] == 2:
                    final_coor[i] = full_chess_corner[98]
                elif result[error_index][3] == 3:
                    final_coor[i] = full_chess_corner[80]
            
            if final_coor[i] == result[error_index][1]:
                if result[error_index][3] == 0:
                    final_coor[i] = full_chess_corner[91]
                elif result[error_index][3] == 1:
                    final_coor[i] = full_chess_corner[10]
                elif result[error_index][3] == 2:
                    final_coor[i] = full_chess_corner[8]
                elif result[error_index][3] == 3:
                    final_coor[i] = full_chess_corner[89]

    w = 640
    src_pts = np.array(final_coor, dtype="float32")
    dst_pts = np.array([[0, w],
                        [0, 0],
                        [w, 0],
                        [w, w]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(final_img, M, (w, w))
    return warped, test, final_coor

