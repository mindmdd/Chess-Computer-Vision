import cv2
import glob
import numpy as np
import imutils
import ImageProcessing
from sklearn.neighbors import NearestNeighbors

class CellFeature():
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
            img     = ImageProcessing.canny_threshold(img)

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

    def detect_color(colorFlolder):
        imageType       = 'jpg'
        colorFiles      = glob.glob(colorFlolder + "/*." + imageType)

        all_color_list = []

        for i in range(len(colorFiles)-1, -1, -1):
            color_img      = cv2.cvtColor(cv2.imread(colorFiles[i]).copy(), cv2.COLOR_BGR2HSV)

            average_color_row = np.average(color_img, axis=0)
            average_color = np.average(average_color_row, axis=0)

            all_color_list.append([average_color[0], 0, 0])
        
        if len(all_color_list) != 0:   

            X = np.array(all_color_list)
            nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)

            test = []
            all_grouping = []
            for id in indices:
                test = []
                # print(np.sort(np.array(id)))
                for idx in id:
                    test.append(all_color_list[idx])
                test = np.array(test)
                # print( 'test', [np.average(test[:,0]), np.average(test[:,1]), np.average(test[:,2])])
                all_grouping.append(np.average(test[:,0]))
            color1 = [max(all_grouping), 0, 0]
            color2 = [min(all_grouping), 0, 0]

            # set1 = []
            # for idx in indices[0]:
            #     set1.append(all_color_list[idx])
            # set1 = np.array(set1)
            # color1 = [np.average(set1[:,0]), np.average(set1[:,1]), np.average(set1[:,2])]

            # set2 = []
            # for idx in indices[1]:
            #     set2.append(all_color_list[idx])
            # set2 = np.array(set2)
            # color2 = [np.average(set2[:,0]), np.average(set2[:,1]), np.average(set2[:,2])]

        print('color: ', color1, color2)
        return(color1, color2)

            

                    
