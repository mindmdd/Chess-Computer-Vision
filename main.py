import cv2
import glob, os
import HandTrack, SetVariable
from Chessboard import Chessboard

# Aprrove param: 
# TODO this is just a temporary set up, it has to be determined by feedback function instead
approved = True

# Set parameter for images files
workingFolder   = "./Image/history"
imageType       = 'jpg'
filename    = workingFolder + "/*." + imageType
images = glob.glob(filename)
#------------------------------------------

def main():

    # Declare class
    Chess = Chessboard()

    # Saving interval time (second)
    saving_interval = 1

    # Count down after hand detected --> act as a buffer and to make sure that hand is out of frame
    countdown_time = 1

    # Current clock
    before_move_count = 0
    after_move_count = 0

    # Filename for saving the sequence
    fname = 0

    # Determine the stae of processing:
    # 1 = ready
    # 2 = hand detected, wait a little bit
    # 3 = process picture after the hand is not detected any more, and get the result
    processing_state = 1

    # Determine manipulator status:
    # 1 = ready
    # 2 = working
    # 3 = done working
    robot_working = 1

    # crop image edge to reduce unnecessary elements that may appear in the picture
    crop_area = 80

    while True:
        # Get saved image
        images = glob.glob(filename)

        # Hand detecting status
        status_detect = 'undetected'

        # Read either camera or saved video, in case camera is not available
        _, image_cam = SetVariable.Camera.cap.read()
        # Crop picture edge
        image_cam = image_cam[0 : image_cam.shape[0], crop_area:image_cam.shape[1]-crop_area]
        # Get real time hand tracked image
        annotated_image = HandTrack.annotated_image(image_cam, 1)
        cv2.imshow("annotated_image", annotated_image)
        # Get detecting status
        hand_landmark, status_detect = HandTrack.handLandmarkProcess(0)
        HandTrack.clearData()

        # TODO received robot_status    
        
        # If no hand detected and the robot is not working
        if status_detect == 'undetected' and robot_working != 2:
            if processing_state == 1:
                # save the image at the start and continus as the interval time setted
                if before_move_count >= saving_interval*10 or before_move_count == 0:
                    # Reset the time
                    before_move_count = 0

                    # Save captured image:
                    # to make all single digit number save as 01, 02, ... not 1, 2 --> so program will be able to sort the file correctly
                    if fname < 10 :
                        full_fname = '0' + str(fname)
                    else:
                        full_fname = str(fname)
                    name = "./Image/history/" + full_fname + '.jpg'
                    print("SAVED", fname)
                    cv2.imwrite(name, image_cam)
                    fname += 1

                    # Reset file name number if it reaches 100  
                    if fname >= 100:
                        fname = 0
                before_move_count += 1
            
            # if hand is not detected after is has been detect
            if processing_state == 2:
                # Will process the picture only after hand is out as time setted
                after_move_count += 1
                if after_move_count %10 == 0:
                    print("WILL COMPARE IN....", (countdown_time*10-after_move_count)//10)
                if after_move_count >= countdown_time*10:
                    processing_state = 3

        # if hand detected or the robot just finish working
        elif status_detect == 'detected' or robot_working == 3:
            processing_state = 2
            before_move_count = 0
            after_move_count = 0
            robot_working = 1

        # Process the picture
        if processing_state == 3: 
            # get previous picture frame to compare   
            if fname < 2 :
                # use the first frame if there are not enough picture
                if len(images)-1 < 2:
                    prev_fname = 0
                #if there are enough frame but current firle name is at the beginning (since we only record 99 frame)
                else:
                    prev_fname = 99 - (2 - fname)
            else:
                prev_fname = fname  - 2

            # If this is the first frame
            if Chess.start == False:
                
                # read the previous image frame recorded
                img = cv2.imread(images[prev_fname])

                # process the image
                chessboard_process_state = Chess.detect_chess(img)

                # redo the process again until it is not error
                while chessboard_process_state == 'error':
                    img = cv2.imread(images[prev_fname+1])
                    chessboard_process_state = Chess.detect_chess(img)
            
            # Process the current frame
            chessboard_process_state = Chess.detect_chess(image_cam)
            from_cell, to_cell, added_cell, removed_cell = Chess.get_move()

            # redo the process again until it is not error --> unable to detect chessboard or there are extra cell
            while chessboard_process_state == 'error' or added_cell != [] or len(from_cell) != 1 or len(to_cell) != 1:
                # Get new picture from camera
                _, image_cam = SetVariable.Camera.cap.read()
                image_cam = image_cam[0 : image_cam.shape[0], crop_area:image_cam.shape[1]-crop_area]
                # Reprocess
                chessboard_process_state = Chess.detect_chess(image_cam)
                from_cell, to_cell, added_cell, removed_cell = Chess.get_move()

            # TODO send the from_cell and to_cell to other part
            # TODO the approved feedback have to be determined

            # Save the result only if it is approved
            if approved == True:
                Chess.reset()

            # Reset the processing_state
            processing_state = 1
            # Reset recorded coutdown time
            after_move_count = 0

        # Clear handtrack data
        HandTrack.clearData()

        # Press ESC to exit loop
        key = cv2.waitKey(1)
        if key == 27:
            break

    # Destroy all window        
    cv2.destroyAllWindows()

    # Quit MATLAB engine
    SetVariable.Matlab.engine.quit()

    # Delete all picture if exit the screen
    image_files = glob.glob(filename)
    for f in image_files:
        os.remove(f)

if __name__ == "__main__":
    main() 