import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque
import argparse
import matlab.engine

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device1", type=int, default=0)
    parser.add_argument("--device2", type=int, default=4)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--max_num_hands",
                        help='max_num_hands',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.9)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.9)

    args = parser.parse_args()

    return args

class Matlab:
    engine = matlab.engine.start_matlab()
    engine.addpath("./MatlabLib", nargout = 0)
    engine.addpath("./MatlabLib/matching", nargout = 0)

class Camera:
    rt_landmark_array = [np.zeros((21, 3)),np.zeros((21, 3))]
    bounding = [0,0]
    landmark_array = [np.zeros((21, 3)),np.zeros((21, 3))]
    landmark_history_cam1 = deque(maxlen=50)
    landmark_history_cam2 = deque(maxlen=50)
    bounding_history_cam1 = deque(maxlen=50)
    bounding_history_cam2 = deque(maxlen=50)
    
    # Argument parsing ------------------------------------------------------
    args = get_args()
    cap_device1 = args.device1
    cap_device2 = args.device2
    cap_width = args.width
    cap_height = args.height

    # Camera preparation ----------------------------------------------------
    # cap1 = cv.VideoCapture(cap_device1, cv.CAP_DSHOW)
    cap1 = cv.VideoCapture('test.mp4')

    # cap1.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap1.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    cap2 = cv.VideoCapture(cap_device2, cv.CAP_DSHOW)
    cap2.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap2.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    save_image1 = []
    save_image2 = []

class HandModel:
    # Argument parsing --------------------------------------------------------------------  
    args = get_args()

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = 0.6
    min_tracking_confidence = args.min_tracking_confidence
    max_num_hands = 2

    mp_drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )