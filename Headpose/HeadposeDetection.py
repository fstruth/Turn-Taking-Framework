# Author Jan-Niklas Oberließ
# Author Michael Schiffmann

import cv2
from multiprocessing import Process, Event
from loguru import logger
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from Headpose.get_math_functions import *
from Headpose.get_mp_landmarks import *
import time


class HeadposeDetection:

    def __init__(self, event):
        self._running = True
        self.event = event

    def terminate(self):
        self.event.set()

    # Head movement definitions
    def head_movement(self, x, y):

        if y < -0.5:
            text = "Looking Left"
            turn = 0

        elif y > 12:
            text = "Looking Right"
            turn = 0

        elif -160 < x < -130:
            text = "Looking Down"
            turn = 0

        elif 120 < x < 160:
            text = "Looking Up"
            turn = 0

        else:
            text = "Head Facing"
            turn = 1

        return turn, text

    def run(self, OutputQueue, InputQueue):
        # Intializing the mediapipe specs
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = solutions.holistic

        vid = cv2.VideoCapture(0)

        # Posture definitions
        points_3d_FACE = [[0, -1.126865, 7.475604],
                    [-4.445859, 2.663991, 3.173422],
                    [4.445859, 2.663991, 3.173422],
                    [-2.456206, -4.342621, 4.283884],
                    [2.456206, -4.342621, 4.283884],
                    [0, -9.403378, 4.264492]]
        turn = 1
        
        while not self.event.is_set():
            with mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as holistic:
                # image = InputQueue.get()
                ret, image = vid.read()

                # get image resolution
                height, width, _ = image.shape

                

                try:
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)

                    #############################################################################################
                    # Get Landmarks of facemesh and pose
                    #############################################################################################
                    [face_fm, nose, L_eye, R_eye, L_ear, R_ear, L_mouth, R_mouth, L_shoulder, R_shoulder, L_hip, R_hip, L_elbow,
                    L_wrist,
                    R_elbow, R_wrist, L_knee, L_ankle, R_knee, R_ankle, L_heel, L_foot_tip, R_heel,
                    R_foot_tip] = get_landmarks(results, mp_holistic)

                    ####################################################################################################################################################
                    ## BAP Postures
                    ####################################################################################################################################################
                    ## Head
                    ###################################################################################################################################################
                    # calculate rotations, translations
                    # nose_2d,nose_3d_projection, x, y, z = pnp_calculator(face_fm[0], face_fm, width, height)
                    nose_2d, nose_3d_projection, x, y, z = new_pnp_calculator(face_fm[0], face_fm, points_3d_FACE, width,
                                                                            height)
                    # Display the nose direction
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                    ## See where the user's head turning
                    turn, direction = self.head_movement(x, y)
                    # logger.debug("X: {}; Y: {}", x, y)
                    logger.debug("Direction: {}", direction)
                    
                #################################################################################################################
                ## End of main loop
                except AttributeError as a:
                    print("Entire Body landmarks not detectable: " + str(a))
                    print("warning will be ignored, program going to next new frame...")
                except Exception as e:
                    # print("Error in Entire_Body: "+str(e))
                    print("Error in Entire_Body: " + str(e))
                    pass
                #######################################################################################################
                if type(turn) == "Nonetype":
                    turn = 1
                else:
                    pass

                OutputQueue.put(turn)
                time.sleep(0.4)

        vid.release()

event = Event()
c = HeadposeDetection(event)
t = None


def StopHeadposeDetection():
    global c, t
    # Signal termination
    c.terminate()
    logger.debug("Process terminated")


def StartHeadposeDetection(InputQueue=None, OutputQueue=None):
    global t, event, c
    # c = VideoRecorder_WebCam()
    if t is None:
        event = Event()
        c = HeadposeDetection(event)
        t = Process(target=c.run, args=(OutputQueue, InputQueue))
        t.start()
        logger.debug("Process started")
