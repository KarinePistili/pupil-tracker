"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
import cv2
from gaze_tracking import GazeTracking
import numpy as np
import csv

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

gaze_points = []
circle_position = (0,10)
i = 0

def add_gaze_point(left_pupil, img, gaze_point):
    if(left_pupil is not None):
        left_pupil = {
            "pupil_x" : left_pupil[0],
            "pupil_y" : left_pupil[1],
            "img": img,
            "gaze_x": gaze_point[0],
            "gaze_y": gaze_point[1],
        }

        gaze_points.append(left_pupil)

def gen_train_dataset():
    csv_file = './training/train_data.csv'
    csv_columns = ['pupil_x','pupil_y','img', 'gaze_x', 'gaze_y']
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in gaze_points:
                writer.writerow(data)
    except IOError:
        print("I/O error")

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()    
    cv2.namedWindow("calib", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("calib",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()

    left_pupil = gaze.pupil_left_coords()
    # right_pupil = gaze.pupil_right_coords()
    # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    
    width  = webcam.get(3)  # float `width`
    height = webcam.get(4)  # float `height`

    # Move circle
    if (circle_position[1] < height):
        if (circle_position[0] <= width):
            circle_position = (circle_position[0] + 8, circle_position[1])
        else:
            # jump a line
            circle_position = (1, circle_position[1] + 100)
    else:
        print('calib ended')
        break

    calib_frame = np.zeros((500, 500, 3), np.uint8)
    calib_frame[:] = (0, 0, 0)

    cv2.circle(calib_frame, circle_position, 5, (0, 0, 255), 5)

    cv2.imshow("calib", calib_frame)
    # cv2.imshow("cam", frame)

    # save calib points
    left_eye = gaze.eye_left
    cv2.imwrite('./training/eyes/eye'+str(i)+'.jpg',left_eye.frame)
    add_gaze_point(left_pupil, 'eye'+str(i)+'.jpg', circle_position)
    i+=1

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()

# create csv with the relation between the images, position of circle on screen and pupil
gen_train_dataset()

cv2.destroyAllWindows()
