import cv2
import glob
import HandTrack, SetVariable, ImageProcessing
from Detect.Changes import Compare

# Set parameter for images files
workingFolder   = "./Image/history"
imageType       = 'jpg'
filename    = workingFolder + "/*." + imageType
images = glob.glob(filename)
#------------------------------------------

saving_sequence = 1
countdown_time = 1

 
def main():
    Chess = Compare()
    before_move_count = 0
    after_move_count = 0
    rotate_deg = 0
    fname = 0
    state = 1
    num = 0
    crop_area = 130
    while True:
        images = glob.glob(filename)
        status_detect1, status_detect2 = 'undetected', 'undetected'
        # if SetVariable.Camera.cap1.isOpened():
        _, image_cam1 = SetVariable.Camera.cap1.read() 
        image_cam1 = image_cam1[0 : image_cam1.shape[0], crop_area:image_cam1.shape[1]-crop_area]
        annotated_image1 = HandTrack.annotated_image(image_cam1, 1)
        cv2.imshow("annotated_image1", annotated_image1)
        hand_landmark1, status_detect1 = HandTrack.handLandmarkProcess(0)
        HandTrack.clearData()
        # if (status_detect1 == "detected"):
        #     print("DETECTED_CAM1")

        if SetVariable.Camera.cap2.isOpened():
            _, image_cam2 = SetVariable.Camera.cap2.read()
            annotated_image2 = HandTrack.annotated_image(image_cam2, 2)
            cv2.imshow("annotated_image2", annotated_image2)
            hand_landmark2, status_detect2 = HandTrack.handLandmarkProcess(1)
            HandTrack.clearData()
            # if (status_detect2 == "detected"):
            #     print("DETECTED_CAM2")
        
        if status_detect1 == 'undetected' and status_detect2 == 'undetected':
            if state == 1:
                if before_move_count >= saving_sequence*10 or before_move_count == 0:
                    before_move_count = 0
                    if fname < 10 :
                        full_fname = '0' + str(fname)
                    else:
                        full_fname = str(fname)
                    name = "./Image/history/" + full_fname + '.jpg'
                    print("SAVED", fname)
                    cv2.imwrite(name, image_cam1)
                    fname += 1  
                    if fname >= 100:
                        fname = 0
                before_move_count += 1
            if state == 2:
                after_move_count += 1
                if after_move_count %10 == 0:
                    print("WILL COMPARE IN....", (countdown_time*10-after_move_count)//10)
                if after_move_count >= countdown_time*10:
                    state = 3
        else:
            state = 2
            before_move_count = 0
            after_move_count = 0

        if state == 3:    
            # _, img = SetVariable.Camera.cap1.read()
            if fname < 2 :
                if len(images)-1 < 2:
                    new_fname = 0
                else:
                    new_fname = 99 - (2 - fname)
            else:
                new_fname = fname  - 2

            if Chess.start == False:
                print(len(images)-1, fname, new_fname)
                print(images[new_fname])
                img = cv2.imread(images[new_fname])
                state = Chess.get_move(img)
                if state == 'error':
                    img = cv2.imread(images[new_fname+1])
                    state = Chess.get_move(img)


            # _, img = SetVariable.Camera.cap1.read()
            # img = cv2.imread(image_cam1)
            # img = cv2.resize(image_cam1, dim, interpolation = cv2.INTER_AREA)
            state = Chess.get_move(image_cam1)
            while state == 'error':
                _, image_cam1 = SetVariable.Camera.cap1.read()
                image_cam1 = image_cam1[0 : image_cam1.shape[0], crop_area:image_cam1.shape[1]-crop_area]
                state = Chess.get_move(image_cam1)
            # Chess.start = False
            state = 1
            after_move_count = 0
            num += 1

        
        HandTrack.clearData()
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
            
        # if num == len(images)-1:
        #     break
        
    cv2.destroyAllWindows()
    SetVariable.Matlab.engine.quit()

if __name__ == "__main__":
    main() 