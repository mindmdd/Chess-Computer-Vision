import math
import cv2
import numpy as np
import Calculation, ImageProcessing

def corner_warp(chess_corner,edited_img):
    
    final_img = edited_img.copy()
    # final_img = cv2.equalizeHist(final_img)
    cv2.imwrite('./Image/Detect.jpg', final_img)
    
    w = 640
    src_pts = np.array(chess_corner, dtype="float32")

    dst_pts = np.array([[0, 0],
                        [0, w],
                        [w, w],
                        [w, 0]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(final_img, M, (w, w))

    return warped

def define_side(full_chess_corner, extended_chess_corner, for_readding_img):
    
    # final_img = cv2.equalizeHist(final_img)
    sides = [[],[],[],[]]
    test = []
    gray = [[],[],[],[]]
    extra_case = False

    for i in range(len(extended_chess_corner)):
        if i%11 == 0:
            if i != 0 and i != 99 and i != 110:
                mid = ((extended_chess_corner[i][0]+extended_chess_corner[i+11][0])//2 , (extended_chess_corner[i][1]+extended_chess_corner[i+11][1])//2)
                sides[0].append(mid)
                
            if i != 0 and i != 11 and i != 110:
                mid = ((extended_chess_corner[i-1][0]+extended_chess_corner[i+10][0])/2 , (extended_chess_corner[i-1][1]+extended_chess_corner[i+10][1])//2)
                sides[2].append(mid)
               
        if i>= 1 and i<=8:
            mid = ((extended_chess_corner[i][0]+extended_chess_corner[i+1][0])//2 , (extended_chess_corner[i][1]+extended_chess_corner[i+1][1])//2)
            sides[1].append(mid)
            
        if i>= 111 and i<=118:
            mid = ((extended_chess_corner[i][0]+extended_chess_corner[i+1][0])//2 , (extended_chess_corner[i][1]+extended_chess_corner[i+1][1])//2)
            sides[3].append(mid)
            
    for coor in sides[0]:
        gray[0].append(for_readding_img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    for coor in sides[1]:
        gray[1].append(for_readding_img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    for coor in sides[2]:
        gray[2].append(for_readding_img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    for coor in sides[3]:
        gray[3].append(for_readding_img[int(coor[1]), int(coor[0])])
        test.append([int(coor[0]),int(coor[1])])
    
    result = []
    avg_color = []
    check_w = 0
    check_b = 0
    check_e = 0
    for m in range(4):
        avg_color.append(np.average(gray[m]))
    white_index = np.where(avg_color == np.amax(avg_color))

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
                c2 = full_chess_corner[72]
            elif m == 1:
                c1 = full_chess_corner[8]
                c2 = full_chess_corner[0]
            elif m == 2:
                c1 = full_chess_corner[80]
                c2 = full_chess_corner[8]
            elif m == 3:
                c1 = full_chess_corner[72]
                c2 = full_chess_corner[80]         
            check_e += 1
            result.append([c1, c2,'error', m, check_e])
            print(m , 'error', gray[m])
        else:
            if black > 3 and white <= 1:
                if m == 0:
                    c1 = full_chess_corner[0]
                    c2 = full_chess_corner[72]
                elif m == 1:
                    c1 = full_chess_corner[8]
                    c2 = full_chess_corner[0]
                elif m == 2:
                    c1 = full_chess_corner[80]
                    c2 = full_chess_corner[8]
                elif m == 3:
                    c1 = full_chess_corner[72]
                    c2 = full_chess_corner[80]  

                check_b += 1  
                result.append([c1, c2, 'black', m, check_b])
                print(m , 'black', gray[m])
            elif black <= 1 and white > 3:
                if m == 0:
                    c1 = full_chess_corner[0]
                    c2 = full_chess_corner[72]
                elif m == 1:
                    c1 = full_chess_corner[8]
                    c2 = full_chess_corner[0]
                elif m == 2:
                    c1 = full_chess_corner[80]
                    c2 = full_chess_corner[8]
                elif m == 3:
                    c1 = full_chess_corner[72]
                    c2 = full_chess_corner[80]  
                    
                check_w += 1
                result.append([c1, c2, 'white', m, check_w])
                print(m , 'white', gray[m])
            else:
                if m == 0:
                    c1 = full_chess_corner[0]
                    c2 = full_chess_corner[72]
                elif m == 1:
                    c1 = full_chess_corner[8]
                    c2 = full_chess_corner[0]
                elif m == 2:
                    c1 = full_chess_corner[80]
                    c2 = full_chess_corner[8]
                elif m == 3:
                    c1 = full_chess_corner[72]
                    c2 = full_chess_corner[80] 
                result.append([c1, c2, 'checker', m])
                print(m , 'checker', gray[m])

    if check_b == 1 and check_w == 1:
        for i in range(len(result)):
            if result[i][2] == 'white':
                w = i
            elif result[i][2] == 'black':
                b = i
        if np.abs(w-b) == 2:
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
        else:
            extra_case = True
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
    # elif check_b == 1 and check_w != 1:
    #     for i in range(len(result)):
    #         if result[i][2] == 'black':
    #             get_side2 = result[i][3] 
    #             x3 = result[i][0][0]
    #             y3 = result[i][0][1]
    #             x4 = result[i][1][0]
    #             y4 = result[i][1][1]
    #             if i == 0:     
    #                 get_side1 = result[2][3]                
    #                 x1 = result[2][0][0]
    #                 y1 = result[2][0][1]
    #                 x2 = result[2][1][0]
    #                 y2 = result[2][1][1]
    #             elif i == 1: 
    #                 get_side1 = result[3][3]                    
    #                 x1 = result[3][0][0]
    #                 y1 = result[3][0][1]
    #                 x2 = result[3][1][0]
    #                 y2 = result[3][1][1]
    #             elif i == 2:   
    #                 get_side1 = result[0][3]                  
    #                 x1 = result[0][0][0]
    #                 y1 = result[0][0][1]
    #                 x2 = result[0][1][0]
    #                 y2 = result[0][1][1]
    #             elif i == 3:   
    #                 get_side1 = result[1][3]                  
    #                 x1 = result[1][0][0]
    #                 y1 = result[1][0][1]
    #                 x2 = result[1][1][0]
    #                 y2 = result[1][1][1]
    else:
        extra_case = True
    
    if extra_case == True:
        print('white_index: ', white_index[0][0])
        i = white_index[0][0]
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

    print('warp side: ', get_side1, get_side2)
    warp_corners = [[x2, y2], [x3, y3], [x4, y4], [x1, y1]]

    return warp_corners

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
            point = get_extend_point(xs[::-1],ys[::-1],ratio)
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
            point = get_extend_point(xs,ys,ratio)
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

            point = get_extend_point(xs[::-1],ys[::-1],ratio)
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
            point = get_extend_point(xs,ys,ratio)
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
            point = get_extend_point(xs[::-1],ys[::-1],ratio)
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
            point = get_extend_point(xs[::-1],ys[::-1],ratio)
            temp2_new_corner.append([point[0],point[1]])
            if i == (side+2)*(side-1)+(side+1):
                for i in temp2_new_corner:
                    new_corner.append(i)
    
    return new_corner
  
def get_extend_point(xs, ys, ratio):
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

    m = np.average(np.array(lm))
    c = np.average(np.array(lc))
    
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
        
        elif abs(xs[i-1]-xs[i]) > 1 and abs(xs[i-1]-xs[i]) < abs(ys[i-1]-ys[i]): 
            
            if ys[i-1]>ys[i]: 
                step = -1
            elif ys[i-1]<=ys[i]: 
                step = 1

            start = [(ys[i]-c)/m, ys[i], 0]
            new_y = ys[i]+(step*60)
            end = [(new_y-c)/m, new_y, 0]

            x, y ,z = Calculation.point_on_line_from_distance(start, end, new_dist)

        else:
            x = xs[i]
            if ys[i-1]>ys[i]: 
                step = -1
            elif ys[i-1]<ys[i]: 
                step = 1

            y = ys[i]+(step*new_dist)
            
        x_list.append(x)
        y_list.append(y)      

    avg_x = np.average(np.array(x_list))
    avg_y = np.average(np.array(y_list))
    
    return (avg_x, avg_y )




