import math
import cv2
import numpy as np
import utlis.Calculation as Calculation
import glob
import imutils
from scipy.spatial import distance as dist


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

class ChessboardFeature():
    def split_cell(img, folname):
        horz = []
        for x in range(8): 
            vert = []
            for y in range(8):
                c1 = (x*80, (y+1)*80)
                c2 = (x*80, y*80)
                c3 = ((x+1)*80, y*80)
                c4 = ((x+1)*80, (y+1)*80)
                
                w = 80
                coor = [c1, c2, c3, c4]
                src_pts = np.array(coor, dtype="float32")
                dst_pts = np.array([[0, w],
                                    [0, 0],
                                    [w, 0],
                                    [w, w]], dtype="float32")
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                result = cv2.warpPerspective(img, M, (w, w))

                if x%2 == 0: 
                    if y%2 ==0:
                        val = 255
                    elif y%2 == 1:
                        val = 0
                elif x%2 == 1:
                    if y%2 ==0:
                        val = 0
                    elif y%2 == 1:
                        val = 255

                letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
                num = ['8', '7', '6', '5', '4', '3', '2', '1']
                fname = './Image/' + folname + '/'+ letter[x] + num[y] + '.jpg'

                vert.append(result)
                #print(fname)
                cv2.imwrite(fname, result)
            V = np.concatenate(vert, axis=0)
            horz.append(V)
        H = np.concatenate(horz, axis=1)
        return H

    def get_warp_corner(full_chess_corner, extended_chess_corner, for_reading_img):
        
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
            gray[0].append(for_reading_img[int(coor[1]), int(coor[0])])
            test.append([int(coor[0]),int(coor[1])])
        for coor in sides[1]:
            gray[1].append(for_reading_img[int(coor[1]), int(coor[0])])
            test.append([int(coor[0]),int(coor[1])])
        for coor in sides[2]:
            gray[2].append(for_reading_img[int(coor[1]), int(coor[0])])
            test.append([int(coor[0]),int(coor[1])])
        for coor in sides[3]:
            gray[3].append(for_reading_img[int(coor[1]), int(coor[0])])
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

    def draw_chessboard(treshSavingFolder, finalSavingFolder, colors):
        imageType       = 'jpg'
        filename    = treshSavingFolder + "/*." + imageType
        images      = glob.glob(filename)

        colorFiles      = glob.glob('./Image/color_img' + "/*." + imageType)


        horz_final = []
        vert_final = [[],[],[],[],[],[],[],[]]

        result = []

        row = 7

        for i in range(len(images)-1, -1, -1):
            treshSavingName = treshSavingFolder + '/' + images[i][len(images[i])-6::]

            color_img = cv2.cvtColor(cv2.imread(colorFiles[i]).copy(), cv2.COLOR_BGR2HSV)
            img     = cv2.imread(images[i])
            img    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            edge = cv2.Canny(img.copy(), 175, 175)
            cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts) != 0:
                for c in cnts:
                    mask = np.zeros(color_img.shape[:2], dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.erode(mask, None, iterations=2)
                    hsv_color = cv2.mean(color_img, mask=mask)[:3]
                    color = [hsv_color[0], 0, 0]

                minDist = (np.inf, None)
                for (color_idx, exist_color) in enumerate(colors):
                    
                    d = dist.euclidean(exist_color, color)
                    if d < minDist[0]: 
                        minDist = (d, color_idx)

            cv2.imwrite(treshSavingName, img)
            
            finalSavingName = finalSavingFolder + '/' + images[i][len(images[i])-6::]

            if images[i][len(images[i])-6] in ['a','c','e','g'] and images[i][len(images[i])-5]in ['1', '3', '5', '7']:
                board_color = 'black.jpg'
            elif images[i][len(images[i])-6] in ['b', 'd', 'f', 'h'] and images[i][len(images[i])-5]in ['2', '4', '6', '8']:
                board_color = 'black.jpg'
            else:
                board_color = 'white.jpg'

            final_img = cv2.imread('./Image/board_template/'+board_color)
            
            if len(cnts) >= 1:
                if minDist[1] == 0:
                    result.append([images[i][len(images[i])-6] + images[i][len(images[i])-5], 0])
                    cv2.circle(final_img,(40,40), 15, (0,0,255), -1)
                elif minDist[1] == 1:
                    result.append([images[i][len(images[i])-6] + images[i][len(images[i])-5], 1])
                    cv2.circle(final_img,(40,40), 15, (0,255,0), -1)
            
            cv2.imwrite(finalSavingName, final_img)
            horz_final.append(final_img)

            if i != 64 and (i)%8 == 0:
                H1 = np.concatenate(horz_final, axis=0)
                vert_final[row] = H1
                horz_final = []
                row -= 1

        final_img = np.concatenate(vert_final, axis=1)
        cv2.imwrite(finalSavingFolder + '.jpg', final_img)

        return final_img, result

    def get_extend_edge(chess_corner,side, ratio):
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
    
