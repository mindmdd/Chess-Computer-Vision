import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque
from utlis.SetVariable import Camera, HandModel


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
 
    x, y, w, h = cv.boundingRect(landmark_array) 
    return w*h

def create_landmark_arr(landmarks):
    landmark_array = np.empty((0, 3), int)
 
    for _, landmark in enumerate(landmarks.landmark):
 
        landmark_point = [np.array((landmark.x, landmark.y, landmark.z))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    return landmark_array

def clearData():
    Camera.landmark_array = [np.zeros((21, 3)),np.zeros((21, 3))]
    Camera.bounding = [0,0]


def annotated_image(image, id):
    landmark_array = np.zeros((21, 3))
    try:
        results = HandModel.hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        # print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            if id == 1:
                Camera.bounding[0] = 0
                Camera.landmark_array[0] = 0
            elif id == 2:
                Camera.bounding[1] = 0
                Camera.landmark_array[1] = 0
            return image
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            size = calc_bounding_rect(image, hand_landmarks)
            landmark_array = create_landmark_arr(hand_landmarks)
            if id == 1:
                Camera.bounding[0] = size
                Camera.landmark_array[0] = landmark_array
            elif id == 2:
                Camera.bounding[1] = size
                Camera.landmark_array[1] = landmark_array
            HandModel.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                HandModel.mp_hands.HAND_CONNECTIONS,
                HandModel.drawing_styles.get_default_hand_landmarks_style(),
                HandModel.drawing_styles.get_default_hand_connections_style())
            return annotated_image
    except cv.error as e:
        print("Camera is None OR OpenCV error")

def handLandmarkProcess(index):
    if Camera.bounding[index] != 0:
        status = "detected"
    else:
        status = "undetected"
    
    return Camera.landmark_array[index], status