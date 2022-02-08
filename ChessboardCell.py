import cv2
import glob
import numpy as np
import imutils
import ImageOperation

def split(img, folname):
    horz = []
    for x in range(8): 
        vert = []
        for y in range(8):
            c1 = (x*80,(y+1)*80)
            c2 = (x*80,y*80)
            c3 = ((x+1)*80,y*80)
            c4 = ((x+1)*80,(y+1)*80)
            
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
            fname = './data/' + folname + '/'+ letter[x] + num[y] + '.jpg'

            vert.append(result)
            #print(fname)
            cv2.imwrite(fname, result)
        V = np.concatenate(vert, axis=0)
        horz.append(V)
    H = np.concatenate(horz, axis=1)
    return H

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


        name = "./data/mask/" + images1[i][len(images1[i])-6::]

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

def change_indicator():
    workingFolder   = './data/mask'
    imageType       = 'jpg'
    filename    = workingFolder + "/*." + imageType
    images      = glob.glob(filename)

    horz = []
    vert = [[],[],[],[],[],[],[],[]]

    result = []

    row = 7

    for i in range(len(images)-1, -1, -1):
        img     = cv2.imread(images[i])
        img    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        kernel = np.ones((13, 13), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        contour_img = cv2.threshold(opening,127,255,cv2.THRESH_BINARY)[1]
        check_contour_img = contour_img.copy()

        edge = cv2.Canny(check_contour_img, 175, 175)
        cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        rect_areas = []
        sum_area = 0

        if len(cnts) != 0:
            # Remove small blobs
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                sum_area += w*h
                # print(sum_area)
                rect_areas.append(w * h)
            avg_area = sum_area/len(rect_areas)

            # 2450

            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cnt_area = w * h
                if cnt_area < 0.4*(64*64):
                    check_contour_img[y:y + h, x:x + w] = 0
            
            #Find max area 
            for contour in cnts:
                area = cv2.contourArea(contour)
                max_contour = max(cnts, key = cv2.contourArea)
                x_max,y_max,w_max,h_max = cv2.boundingRect(max_contour)
                max_contour_area = w_max*h_max
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                width = int(rect[1][0])
                height = int(rect[1][1])

            # Remove large blobs
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cnt_area = w * h
                if cnt_area < max_contour_area or cnt_area > 0.9*(64*64):
                    # bolb_rect = cv2.minAreaRect(c)
                    # bolb_box = cv2.boxPoints(bolb_rect)
                    # bolb_box = np.int0(bolb_box)
                    # cv2.fillConvexPoly(contour_img, bolb_box, 0)
                    check_contour_img[y:y + h, x:x + w] = 0
            
            # for c in cnts:
            #     (x, y, w, h) = cv2.boundingRect(c)
            #     cnt_area = w * h
            #     if cnt_area < 0.3*(64*64):
            #         check_contour_img[y:y + h, x:x + w] = 0
            #     circle, corner = ImageOperation.check_circle(c)
            #     # print('object corner:', corner)
            #     if circle == False:
            #         check_contour_img[y:y + h, x:x + w] = 0
        
        edge = cv2.Canny(check_contour_img, 175, 175)
        cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) >= 1:
            result.append(images[i][len(images[i])-6] + images[i][len(images[i])-5])

        horz.append(check_contour_img)

        if i != 64 and (i)%8 == 0:
            H = np.concatenate(horz, axis=0)
            vert[row] = H
            row -= 1
            horz = []
    fin = np.concatenate(vert, axis=1)

    return fin, result
   
def detect_peice():
    workingFolder   = './data/current_img'
    imageType       = 'jpg'
    filename    = workingFolder + "/*." + imageType
    images      = glob.glob(filename)

    horz = []
    vert = [[],[],[],[],[],[],[],[]]

    result = []

    row = 7

    for i in range(len(images)-1, -1, -1):
        img     = cv2.imread(images[i])
        img    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img,(7,7),0)
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

        # Fill the edge
        path_edge = 8
        img[0:path_edge, 0:img.shape[1]]= 0
        img[img.shape[0] - path_edge:img.shape[0], 0:img.shape[1]] = 0
        img[0:img.shape[0], 0:path_edge]  = 0
        img[0:img.shape[0], img.shape[1] - path_edge:img.shape[1]] = 0

        # Fill hole
        im_floodfill = img.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        img = img | im_floodfill_inv

        
        kernel = np.ones((13, 13), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        check_contour_img = img.copy()

        edge = cv2.Canny(check_contour_img, 175, 175)
        cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) != 0:
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cnt_area = w * h
                if cnt_area < 0.3*(64*64):
                    check_contour_img[y:y + h, x:x + w] = 0
                circle, corner = ImageOperation.check_circle(c)
                print('all object corner:', corner)
                if circle == False:
                    check_contour_img[y:y + h, x:x + w] = 0
        
        edge = cv2.Canny(check_contour_img, 175, 175)
        cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) >= 1:
            result.append(images[i][len(images[i])-6] + images[i][len(images[i])-5])


        # name = "./data/cropped_3/" + images[i][len(images[i])-6::]
        # cv2.imwrite(name, img)

        horz.append(img)

        if i != 64 and (i)%8 == 0:
            H = np.concatenate(horz, axis=0)
            vert[row] = H
            row -= 1
            horz = []
    fin = np.concatenate(vert, axis=1)
    return fin, result

