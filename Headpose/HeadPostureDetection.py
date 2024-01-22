## Code written in Python 3.9
# Author Jan-Niklas Oberließ
# Author Michael Schiffmann


import cv2
import mediapipe as mp
from get_math_functions import *
from get_mp_landmarks import *
############################################################################################
# Intializing the mediapipe specs


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic



##################################################################################
## Local Functions
##################################################################################


# Head movement definitions
def head_movement(x,y):
    # print("X: " + str(x) + "    Y: " + str(y))
    # Winkel korrigiert, da so nicht realistisch
    # TODO Winkel anpassen !
    if y < -0.5:
        text = "Looking Left"
        #averted
    elif y > 12:
        text = "Looking Right"
        # averted
    elif x > -176 & x < -150:
        text = "Looking Down"
        #averted
    elif x > 150:
        text = "Looking Up"
        #averted
    else:
        text = "Head Facing"
    return text


# draw landmarks on the live image
def draw_lm(image):
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

#Posture definitions
points_3d_FACE = [[0, -1.126865, 7.475604],
              [-4.445859, 2.663991, 3.173422],
              [4.445859, 2.663991, 3.173422],
              [-2.456206, -4.342621, 4.283884],
              [2.456206, -4.342621, 4.283884],
              [0, -9.403378, 4.264492]]
# Get Camera Connection
cap = cv2.VideoCapture(0) #1 for external webcam / 0 for internal cam

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
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

            # Additional Landmarks (Trunk)
            mid_L = midpoint(L_shoulder, L_hip)
            mid_R = midpoint(R_shoulder, R_hip)
            center_trunk = midpoint(mid_L, mid_R)
            center_trunk = midpoint(mid_L, mid_R)
            center_trunk = [center_trunk[0], center_trunk[1], center_trunk[2]]  # 1
            mid_shoulder = midpoint(L_shoulder, R_shoulder)
            mid_shoulder = [mid_shoulder[0], mid_shoulder[1], mid_shoulder[2]]
            # save trunk in list
            trunk_list = [L_shoulder, R_shoulder, L_hip, R_hip, center_trunk, mid_shoulder]

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
            turning = head_movement(x, y)
            # print(turning)
            print("X: {}, Y: {}", x, y)
            print("Turning:" + turning)
            cv2.putText(image,str(turning),(20,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            ## See where the user's head tilting (L/R)
            tilt = False
            TiltValue = str(0)
            if R_shoulder[3] > 0.9 and L_shoulder[3] > 0.9:
                # angle_L_Headside = angle(face_fm[6], mid_shoulder, L_shoulder)
                lm_facemesh = results.face_landmarks.landmark

                HeadAngle = angleHead([lm_facemesh[199].x, lm_facemesh[199].y, lm_facemesh[199].z],
                                      [lm_facemesh[9].x, lm_facemesh[9].y, lm_facemesh[9].z], L_shoulder, R_shoulder)

                angle_L_Headside = HeadAngle
                if angle_L_Headside < 85:
                    tilting = "Head Tilting Right"
                    tilt = True
                elif angle_L_Headside > 95:
                    tilting = "Head Tilting Left"
                    tilt = True
                else:
                    tilting = " "
                    tilt = False
                TiltValue = str(angle_L_Headside)
            else:
                tilting = " "
                tilt = False
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
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw_lm(image)
        # showing live image
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27:
            print("Escape pressed...")
            print("Programm will be terminated...")
            break
# terminate camera connection
cap.release()
