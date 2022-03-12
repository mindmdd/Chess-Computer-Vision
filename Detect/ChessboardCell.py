import cv2
import glob
import numpy as np
import imutils
import ImageProcessing
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance as dist


def CannyThreshold(img):
    low_threshold = 15
    ratio = 3
    kernel_size = 3
    img_blur = cv2.blur(img, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    # mask = detected_edges != 0
    # dst = img * (mask[:,:,None].astype(img.dtype))
    return detected_edges

def split(img, folname):
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

def detect_peice(workingFolder, treshSavingFolder):
    imageType       = 'jpg'
    filename    = workingFolder + "/*." + imageType
    images      = glob.glob(filename)

    horz_tresh = []
    vert_tresh = [[],[],[],[],[],[],[],[]]

    result = []
    row = 7

    for i in range(len(images)-1, -1, -1):
        treshSavingName = treshSavingFolder + '/' + images[i][len(images[i])-6::]

        img     = cv2.imread(images[i])
        img    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = CannyThreshold(img)

        # Fill the edge
        path_edge = 8
        img[0:path_edge, 0:img.shape[1]]= 0
        img[img.shape[0] - path_edge:img.shape[0], 0:img.shape[1]] = 0
        img[0:img.shape[0], 0:path_edge]  = 0
        img[0:img.shape[0], img.shape[1] - path_edge:img.shape[1]] = 0
        

        kernel = np.ones((15, 15), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        
        img = ImageProcessing.floodfill(img)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)

        edge = cv2.Canny(img.copy(), 175, 175)
        cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) != 0:
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cnt_area = w * h
                if cnt_area < 0.05*(img.shape[0]*img.shape[1]):
                    img[y:y + h, x:x + w] = 0
        
        kernel = np.ones((11, 11), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = ImageProcessing.floodfill(img)

        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        
        checkName = './Image/check' + '/' + images[i][len(images[i])-6::]
        cv2.imwrite(checkName, img)

        edge = cv2.Canny(img.copy(), 175, 175)
        cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) >= 1:
            result.append(images[i][len(images[i])-6] + images[i][len(images[i])-5])

        cv2.imwrite(treshSavingName, img)
        horz_tresh.append(img)

        if i != 64 and (i)%8 == 0:

            H2 = np.concatenate(horz_tresh, axis=0)
            vert_tresh[row] = H2
            horz_tresh = []

            row -= 1
        
    final_tresh = np.concatenate(vert_tresh, axis=1)
    cv2.imwrite(treshSavingFolder+ '.jpg', final_tresh)

    return final_tresh, result

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

        color_img = cv2.imread(colorFiles[i])
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
                color = cv2.mean(color_img, mask=mask)[:3]

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

def compare(workingFolder1, workingFolder2):

    imageType       = 'jpg'
    filename1    = workingFolder1 + "/*." + imageType
    filename2    = workingFolder2 + "/*." + imageType
    images1      = glob.glob(filename1)
    images2      = glob.glob(filename2)

    horz = []
    vert = [[],[],[],[],[],[],[],[]]

    row = 7

    for i in range(len(images1)-1, -1, -1):
        img1     = cv2.imread(images1[i])
        img2     = cv2.imread(images2[i])

        
        diff = cv2.absdiff(img1, img2)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        th = 35
        imask =  mask>th

        canvas = np.zeros_like(img2, np.uint8)
        canvas[imask] = 255


        path_edge = 8
        canvas[0:path_edge, 0:canvas.shape[1]]= 0
        canvas[canvas.shape[0] - path_edge:canvas.shape[0], 0:canvas.shape[1]] = 0
        canvas[0:canvas.shape[0], 0:path_edge]  = 0
        canvas[0:canvas.shape[0], canvas.shape[1] - path_edge:canvas.shape[1]] = 0


        name = "./Image/mask/" + images1[i][len(images1[i])-6::]

        # print(images1[i][len(images1[i])-6::])

        cv2.imwrite(name, canvas)

        horz.append(canvas)

        if i != 64 and (i)%8 == 0:
            H = np.concatenate(horz, axis=0)
            vert[row] = H
            row -= 1
            horz = []
    fin = np.concatenate(vert, axis=1)
    return fin

def change_indicator(prev_list, current_list):
    from_cell, to_cell, added_cell, removed_cell = [], [], [], []
    
    if len(prev_list) == len(current_list):
        for cell in current_list:
            if cell not in prev_list:
                from_cell.append(cell)

        for cell in prev_list:
            if cell not in current_list:
                to_cell.append(cell)
    else:
        for cell in current_list:
            if cell not in prev_list:
                removed_cell.append(cell)

        for cell in prev_list:
            if cell not in current_list:
                added_cell.append(cell)

    return from_cell, to_cell, added_cell, removed_cell
   
def detect_color(colorFlolder, treshFolder):
    imageType       = 'jpg'
    treshFiles      = glob.glob(treshFolder + "/*." + imageType)
    colorFiles      = glob.glob(colorFlolder + "/*." + imageType)

    all_color_list = []
    b =[]
    g =[]
    r =[]
    for i in range(len(treshFiles)-1, -1, -1):
        tresh_img     = cv2.imread(treshFiles[i])
        color_img      = cv2.imread(colorFiles[i])

        edge = cv2.Canny(tresh_img.copy(), 175, 175)
        cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) != 0:
            for c in cnts:
                for c in cnts:
                    mask = np.zeros(color_img.shape[:2], dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.erode(mask, None, iterations=2)
                    color = cv2.mean(color_img, mask=mask)[:3]
                    all_color_list.append([color[0], color[1], color[2]])

    if len(all_color_list) != 0:   
        X = np.array(all_color_list)
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        set1 = []
        for idx in indices[0]:
            set1.append(all_color_list[idx])
        set1 = np.array(set1)
        color1 = [np.mean(set1[:,0]), np.mean(set1[:,1]), np.mean(set1[:,2])]

        set2 = []
        for idx in indices[1]:
            set2.append(all_color_list[idx])
        set2 = np.array(set2)
        color2 = [np.mean(set2[:,0]), np.mean(set2[:,1]), np.mean(set2[:,2])]

    print('color: ', color1, color2)
    return(color1, color2)

        

                    
