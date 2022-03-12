import cv2
import glob
import HandTrack, SetVariable, ImageProcessing
from Detect.Changes import Compare

# Set parameter for images files
workingFolder   = "./Image/template"
imageType       = 'jpg'
filename    = workingFolder + "/*." + imageType
images      = glob.glob(filename)
#------------------------------------------

saving_sequence = 3
countdown_time = 3

 
def main():
    Chess = Compare()
    before_move_count = 0
    after_move_count = 0
    rotate_deg = 0
    fname = 0
    state = 1
    num = 0
    
    while True:
        # print(before_move_count, after_move_count, state)
        status_detect1, status_detect2 = 'undetected', 'undetected'
        if SetVariable.Camera.cap1.isOpened():
            _, image_cam1 = SetVariable.Camera.cap1.read()
            annotated_image1 = HandTrack.annotated_image(image_cam1, 1)
            cv2.imshow("annotated_image1", annotated_image1)
            hand_landmark1, status_detect1 = HandTrack.handLandmarkProcess(0)
            HandTrack.clearData()
            if (status_detect1 == "detected"):
                print("DETECTED_CAM1")

        if SetVariable.Camera.cap2.isOpened():
            _, image_cam2 = SetVariable.Camera.cap2.read()
            annotated_image2 = HandTrack.annotated_image(image_cam2, 2)
            cv2.imshow("annotated_image2", annotated_image2)
            hand_landmark2, status_detect2 = HandTrack.handLandmarkProcess(1)
            HandTrack.clearData()
            if (status_detect2 == "detected"):
                print("DETECTED_CAM2")
        
        if status_detect1 == 'undetected' and status_detect2 == 'undetected':
            if state == 1:
                if before_move_count >= saving_sequence*10 or before_move_count == 0:
                    before_move_count = 0
                    name = "./Image/history/" + str(fname) + '.jpg'
                    print("SAVED", fname)
                    cv2.imwrite(name, image_cam1)
                    fname += 1  
                    if fname > 100:
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
            
            img = cv2.imread(images[num])
            print(img.shape) # 1900, 3400
            dim = (3400, 1900)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            Chess.get_move(img)

            # _, img = SetVariable.Camera.cap1.read()
            img = cv2.imread(images[num+1])
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            Chess.get_move(img)
            Chess.start = False
            state = 1
            after_move_count = 0
            num += 1
        
        HandTrack.clearData()
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
            
        if num == len(images)-1:
            break
        
    cv2.destroyAllWindows()
    SetVariable.Matlab.engine.quit()

if __name__ == "__main__":
    main() 