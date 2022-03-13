import cv2
import numpy as np
import ImageProcessing, SetVariable
from Detect import Chessboard1, Chessboard2, ChessboardCell


class Compare():
    def __init__(self):
        self.start = False
        self.prev_warped_chessboard = 0
        self.color1 = [0,0,0]
        self.color2 = [0,0,0]
        
    def get_move(self, img):
        horz1 = [[], [], [], []]
        horz2 = [[], [], [], []]

        img     = ImageProcessing.image_resize(img, height = 500)
        color_contrast_img = ImageProcessing.add_contrast(img.copy(), -20, 64)
        color_img = img.copy()
        contrast_img = ImageProcessing.add_contrast(img.copy(), 50, 64)
        contrast_gray    = cv2.cvtColor(contrast_img.copy(),cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./Image/contrast_gray.jpg', contrast_gray)
        
        normal_gray    = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./Image/norml_gray.jpg', normal_gray)


        ImageProcessing.field_contour(img.copy(), './Image/clahe_gray.jpg')
        clahe_gray = cv2.imread('./Image/clahe_gray.jpg')

        cv2.imwrite("./MatlabLib/data/img.jpg", contrast_gray)
        corners,chessboards = SetVariable.Matlab.engine.demo("MatlabLib/data/img.jpg", nargout=2)

        chess_corner = []
        for set in range(len(chessboards[0])-1,-1,-1):
            for index in chessboards[0][set]:
                coord = corners['p'][int(index)-1]
                chess_corner.append([coord[0]-1, coord[1]-1])

        detected_chessboard = cv2.imread('./Image/clahe_gray.jpg')
        for i in chess_corner  :
            detected_chessboard = cv2.circle(detected_chessboard, (int(i[0]),int(i[1])), 3, (0,0,255), -1)
        cv2.imwrite('./Image/Detect_ORG.jpg', detected_chessboard)

        print('chesscorner: ', np.array(chess_corner).shape)
        if np.array(chess_corner).shape[0] / 7 != 7:
            return 'error'

        full_chess_corner = Chessboard2.find_edge(chess_corner,7,1)
        extended_chess_corner = Chessboard2.find_edge(full_chess_corner,9,1/3)
        print('full chesscorner: ', np.array(full_chess_corner).shape)

        # Displaying the image 
        detected_chessboard = cv2.imread('./Image/clahe_gray.jpg')
        for i in full_chess_corner  :
            detected_chessboard = cv2.circle(detected_chessboard, (int(i[0]),int(i[1])), 3, (0,0,255), -1)
        cv2.imwrite('./Image/Detect.jpg', detected_chessboard)
        warp_corners = Chessboard2.define_side(full_chess_corner, extended_chess_corner, cv2.cvtColor(clahe_gray.copy(),cv2.COLOR_BGR2GRAY))

        new_contrast_img = ImageProcessing.add_contrast(img.copy(), -20, 60)
        warped_chessboard = ImageProcessing.warp_corner(warp_corners, cv2.cvtColor(new_contrast_img.copy(),cv2.COLOR_BGR2GRAY), 640, 640)
        
        warped_color_img = ImageProcessing.warp_corner(warp_corners, color_img, 640, 640)
        warped_color_contrast_img = ImageProcessing.warp_corner(warp_corners, color_contrast_img, 640, 640)
        ChessboardCell.split(warped_color_contrast_img, 'color_img')

        if self.start == False:
            self.prev_detected_chessboard = detected_chessboard.copy()
            self.prev_warped_chessboard = warped_chessboard.copy()
            self.prev_color_warped_chessboard = warped_color_img.copy()
            cv2.imwrite('./Image/prev_original_img.jpg', warped_chessboard)
            ChessboardCell.split(self.prev_warped_chessboard, 'prev_img')
            self.all_prev_tresh_cell_img, self.list_all_prev_cell = ChessboardCell.detect_peice('./Image/prev_img', './Image/prev_tresh_img')
            self.color1, self.color2 = ChessboardCell.detect_color('./Image/color_img', './Image/prev_tresh_img')
            self.all_prev_cell_img, self.list_color_all_prev_cell = ChessboardCell.draw_chessboard('./Image/prev_tresh_img', './Image/prev_final_img', [self.color1, self.color2])
            self.start = True

        # Compare current frame with previous frame
        elif self.start == True:

            ChessboardCell.split(self.all_prev_tresh_cell_img, 'prev_tresh_img')
            cv2.imwrite('./Image/prev_tresh_img.jpg', self.all_prev_tresh_cell_img)
            cv2.imwrite('./Image/prev_final_img.jpg', self.all_prev_cell_img)
            cv2.imwrite('./Image/prev_original_img.jpg', self.prev_warped_chessboard)
            
            prev_split_cell = ChessboardCell.split(self.prev_warped_chessboard, 'prev_img')
            prev_detected_chessboard = self.prev_detected_chessboard.copy()

            current_warped_chessboard = warped_chessboard.copy()
            current_color_warped_chessboard = warped_color_img.copy()
            cv2.imwrite('./Image/current_original_img.jpg', warped_chessboard)
            current_split_cell = ChessboardCell.split(current_warped_chessboard, 'current_img')
            
            all_current_tresh_cell_img, list_all_current_cell = ChessboardCell.detect_peice('./Image/current_img', './Image/current_tresh_img')
            # self.color1, self.color2 = ChessboardCell.detect_color('./Image/color_img', './Image/current_tresh_img')
            all_current_cell_img, list_color_all_current_cell = ChessboardCell.draw_chessboard('./Image/current_tresh_img', './Image/current_final_img', [self.color1, self.color2])

        
            # ChessboardCell.compare('./Image/prev_final_img', './Image/current_final_img')
            from_cell, to_cell, added_cell, removed_cell = ChessboardCell.change_indicator(self.list_all_prev_cell, list_all_current_cell)
            # cv2.imshow('changed_cell_img', changed_cell_img)
            
            horz1[0] = ImageProcessing.image_resize(prev_detected_chessboard, height = 250)
            horz2[0] = ImageProcessing.image_resize(detected_chessboard, height = 250)

            horz1[1] = ImageProcessing.image_resize(cv2.cvtColor(prev_split_cell, cv2.COLOR_GRAY2RGB), height = 250)
            horz2[1] = ImageProcessing.image_resize(cv2.cvtColor(current_split_cell, cv2.COLOR_GRAY2RGB), height = 250)

            horz1[2] = ImageProcessing.image_resize(self.prev_color_warped_chessboard, height = 250)
            horz2[2] = ImageProcessing.image_resize(current_color_warped_chessboard, height = 250)

            horz1[3] = ImageProcessing.image_resize(self.all_prev_cell_img, height = 250)
            horz2[3] = ImageProcessing.image_resize(all_current_cell_img, height = 250)

            
            
            row1 = np.concatenate(horz1, axis=1)
            row2 = np.concatenate(horz2, axis=1)

            final_img = np.concatenate([row1, row2], axis=0)
            cv2.imshow('final_img', final_img)

            print("------------------------------------------------")
            print('all previous cells:', self.list_all_prev_cell)
            print('all current cells:', list_all_current_cell)
            print("------------------------------------------------")
            print('FROM:', from_cell)
            print('TO:', to_cell)
            print('ADDED:', added_cell)
            print('ROMOVED:', removed_cell)
            print("------------------------------------------------")

            self.all_prev_tresh_cell_img = all_current_tresh_cell_img.copy()
            self.all_prev_cell_img = all_current_cell_img.copy()
            self.prev_detected_chessboard = detected_chessboard.copy()
            self.prev_warped_chessboard = current_warped_chessboard.copy()
            self.list_all_prev_cell = list_all_current_cell.copy()
            self.prev_color_warped_chessboard = current_color_warped_chessboard.copy()
        return 'done'