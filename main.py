"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
from screeninfo import get_monitors
import pyautogui
import keyboard
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height

gaze_points = []


def add_gaze_point(left_pupil):
    if(left_pupil is not None):
        left_pupil = {
            "pupil_x" : left_pupil[0],
            "pupil_y" : left_pupil[1],
            "mouse_x" : pyautogui.position().x,
            "mouse_y" : pyautogui.position().y
        }

        gaze_points.append(left_pupil)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()    

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Create array of (x,y) pupil location VS (x,y) coordinate on screen
    if keyboard.is_pressed('s'):
        add_gaze_point(left_pupil)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()

print(gaze_points)

cv2.destroyAllWindows()
