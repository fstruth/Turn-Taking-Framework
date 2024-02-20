from math import sqrt, pi, atan2, asin

from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
from loguru import logger

def get_landmarks(results, mp_holistic):
    # CHECK WITH visibility Value if the points are really there !


    ## Face landmarks (Face Mesh)
        lm_facemesh=results.face_landmarks.landmark
        # head face mesh
        nose_fm=[lm_facemesh[1].x,lm_facemesh[1].y,lm_facemesh[1].z]
        R_eye_fm = [lm_facemesh[33].x, lm_facemesh[33].y, lm_facemesh[33].z]
        L_eye_fm=[lm_facemesh[263].x,lm_facemesh[263].y,lm_facemesh[263].z]
        R_mouth_fm = [lm_facemesh[61].x, lm_facemesh[61].y, lm_facemesh[61].z]
        L_mouth_fm=[lm_facemesh[291].x,lm_facemesh[291].y,lm_facemesh[291].z]
        chin_fm=[lm_facemesh[199].x,lm_facemesh[199].y,lm_facemesh[199].z]
        # chin_fm = [lm_facemesh[9].x, lm_facemesh[9].y, lm_facemesh[9].z]
        L_headside_fm=[lm_facemesh[454].x,lm_facemesh[454].y,lm_facemesh[454].z]
        R_headside_fm=[lm_facemesh[234].x,lm_facemesh[234].y,lm_facemesh[234].z]
        topofface_fm=[lm_facemesh[10].x,lm_facemesh[10].y,lm_facemesh[10].z]
        # Save in list
        face_fm=[nose_fm,R_eye_fm,L_eye_fm,R_mouth_fm,L_mouth_fm,chin_fm,L_headside_fm,R_headside_fm,topofface_fm]
        #############################################################################################
    ## Pose landmarks (entire body)
        landmarks=results.pose_landmarks.landmark
        #Head
        nose= [landmarks[mp_holistic.PoseLandmark.NOSE.value].x,landmarks[mp_holistic.PoseLandmark.NOSE.value].y, landmarks[mp_holistic.PoseLandmark.NOSE.value].z,landmarks[mp_holistic.PoseLandmark.NOSE.value].visibility ]
        L_eye = [landmarks[mp_holistic.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_EYE.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_EYE.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_EYE.value].visibility]
        R_eye = [landmarks[mp_holistic.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_EYE.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_EYE.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_EYE.value].visibility]
        L_ear = [landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].visibility]
        R_ear = [landmarks[mp_holistic.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_EAR.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_EAR.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_EAR.value].visibility]
        L_mouth = [landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value].x,landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value].y, landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value].z,landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value].visibility]
        R_mouth = [landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value].x,landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value].y, landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value].z,landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value].visibility]
        # Trunk
        L_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].visibility]
        R_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].visibility]
        L_hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].visibility]
        R_hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].visibility]
        # Left Arm
        L_elbow= [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].visibility]
        L_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].visibility]
        # Right Arm
        R_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].visibility]
        R_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].visibility]
        # Left Leg and Foot
        L_knee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].visibility]
        L_ankle = [landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].visibility]
        L_heel=[landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value].visibility]
        L_foot_tip=[landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].z,landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].visibility]        
        # Right Leg and Foot
        R_knee = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].visibility]
        R_ankle = [landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].visibility]
        R_heel=[landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value].visibility]
        R_foot_tip=[landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].z,landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility]
        ## 
    
        return face_fm, nose, L_eye, R_eye,L_ear,R_ear,L_mouth,R_mouth, L_shoulder,R_shoulder,L_hip,R_hip,L_elbow, L_wrist,R_elbow,R_wrist,L_knee,L_ankle,R_knee,R_ankle, L_heel,L_foot_tip,R_heel,R_foot_tip

def get_Lhands(results): #get_hands(results):
    ## Hand landmarks
    #Left Hand
    LA_lm=results.left_hand_landmarks.landmark   
    LA_wrist=[LA_lm[0].x,LA_lm[0].y,LA_lm[0].z,LA_lm[0].visibility] 
    LA_thumb_cmc=[LA_lm[1].x,LA_lm[1].y,LA_lm[1].z,LA_lm[1].visibility]
    LA_thumb_mcp=[LA_lm[2].x,LA_lm[2].y,LA_lm[2].z,LA_lm[2].visibility]
    LA_thumb_ip=[LA_lm[3].x,LA_lm[3].y,LA_lm[3].z,LA_lm[3].visibility]
    LA_thumb_tip=[LA_lm[4].x,LA_lm[4].y,LA_lm[4].z,LA_lm[4].visibility]
    LA_index_mcp=[LA_lm[5].x,LA_lm[5].y,LA_lm[5].z,LA_lm[5].visibility]
    LA_index_pip=[LA_lm[6].x,LA_lm[6].y,LA_lm[6].z,LA_lm[6].visibility]
    LA_index_dip=[LA_lm[7].x,LA_lm[7].y,LA_lm[7].z,LA_lm[7].visibility]
    LA_index_tip=[LA_lm[8].x,LA_lm[8].y,LA_lm[8].z,LA_lm[8].visibility]
    LA_middle_tip=[LA_lm[9].x,LA_lm[9].y,LA_lm[9].z,LA_lm[9].visibility]
    LA_middle_mcp=[LA_lm[10].x,LA_lm[10].y,LA_lm[10].z,LA_lm[10].visibility]
    LA_middle_pip=[LA_lm[11].x,LA_lm[11].y,LA_lm[11].z,LA_lm[11].visibility]
    LA_middle_dip=[LA_lm[12].x,LA_lm[12].y,LA_lm[12].z,LA_lm[12].visibility]
    LA_ring_mcp=[LA_lm[13].x,LA_lm[13].y,LA_lm[13].z,LA_lm[13].visibility]
    LA_ring_pip=[LA_lm[14].x,LA_lm[14].y,LA_lm[14].z,LA_lm[14].visibility]
    LA_ring_dip=[LA_lm[15].x,LA_lm[15].y,LA_lm[15].z,LA_lm[15].visibility]
    LA_ring_tip=[LA_lm[16].x,LA_lm[16].y,LA_lm[16].z,LA_lm[16].visibility]
    LA_pinky_mcp=[LA_lm[17].x,LA_lm[17].y,LA_lm[17].z,LA_lm[17].visibility]
    LA_pinky_pip=[LA_lm[18].x,LA_lm[18].y,LA_lm[18].z,LA_lm[18].visibility]
    LA_pinky_dip=[LA_lm[19].x,LA_lm[19].y,LA_lm[19].z,LA_lm[19].visibility]
    LA_pinky_tip=[LA_lm[20].x,LA_lm[20].y,LA_lm[20].z,LA_lm[20].visibility]
    
    LHand=[LA_wrist,LA_thumb_cmc,LA_thumb_mcp,LA_thumb_ip,LA_thumb_tip,LA_index_mcp,LA_index_pip,LA_index_dip,LA_index_tip,LA_middle_tip,LA_middle_mcp,
            LA_middle_pip,LA_middle_dip,LA_ring_mcp,LA_ring_pip,LA_ring_dip,LA_ring_tip,LA_pinky_mcp,LA_pinky_pip,LA_pinky_dip,LA_pinky_tip]

    return LHand
##
def get_Rhands(results):
    # Right Hand
    RA_lm=results.right_hand_landmarks.landmark 
    RA_wrist=[RA_lm[0].x,RA_lm[0].y,RA_lm[0].z,RA_lm[0].visibility]
    RA_thumb_cmc=[RA_lm[1].x,RA_lm[1].y,RA_lm[1].z,RA_lm[1].visibility]
    RA_thumb_mcp=[RA_lm[2].x,RA_lm[2].y,RA_lm[2].z,RA_lm[2].visibility]
    RA_thumb_ip=[RA_lm[3].x,RA_lm[3].y,RA_lm[3].z,RA_lm[3].visibility]
    RA_thumb_tip=[RA_lm[4].x,RA_lm[4].y,RA_lm[4].z,RA_lm[4].visibility]
    RA_index_mcp=[RA_lm[5].x,RA_lm[5].y,RA_lm[5].z,RA_lm[5].visibility]
    RA_index_pip=[RA_lm[6].x,RA_lm[6].y,RA_lm[6].z,RA_lm[6].visibility]
    RA_index_dip=[RA_lm[7].x,RA_lm[7].y,RA_lm[7].z,RA_lm[7].visibility]
    RA_index_tip=[RA_lm[8].x,RA_lm[8].y,RA_lm[8].z,RA_lm[8].visibility]
    RA_middle_tip=[RA_lm[9].x,RA_lm[9].y,RA_lm[9].z,RA_lm[9].visibility]
    RA_middle_mcp=[RA_lm[10].x,RA_lm[10].y,RA_lm[10].z,RA_lm[10].visibility]
    RA_middle_pip=[RA_lm[11].x,RA_lm[11].y,RA_lm[11].z,RA_lm[11].visibility]
    RA_middle_dip=[RA_lm[12].x,RA_lm[12].y,RA_lm[12].z,RA_lm[12].visibility]
    RA_ring_mcp=[RA_lm[13].x,RA_lm[13].y,RA_lm[13].z,RA_lm[13].visibility]
    RA_ring_pip=[RA_lm[14].x,RA_lm[14].y,RA_lm[14].z,RA_lm[14].visibility]
    RA_ring_dip=[RA_lm[15].x,RA_lm[15].y,RA_lm[15].z,RA_lm[15].visibility]
    RA_ring_tip=[RA_lm[16].x,RA_lm[16].y,RA_lm[16].z,RA_lm[16].visibility]
    RA_pinky_mcp=[RA_lm[17].x,RA_lm[17].y,RA_lm[17].z,RA_lm[17].visibility]
    RA_pinky_pip=[RA_lm[18].x,RA_lm[18].y,RA_lm[18].z,RA_lm[18].visibility]
    RA_pinky_dip=[RA_lm[19].x,RA_lm[19].y,RA_lm[19].z,RA_lm[19].visibility]
    RA_pinky_tip=[RA_lm[20].x,RA_lm[20].y,RA_lm[20].z,RA_lm[20].visibility]

    RHand=[RA_wrist,RA_thumb_cmc,RA_thumb_mcp,RA_thumb_ip,RA_thumb_tip,RA_index_mcp,RA_index_pip,RA_index_dip,RA_index_tip,RA_middle_tip,RA_middle_mcp,
                RA_middle_pip,RA_middle_dip,RA_ring_mcp,RA_ring_pip,RA_ring_dip,RA_ring_tip,RA_pinky_mcp,RA_pinky_pip,RA_pinky_dip,RA_pinky_tip]

    return RHand  

# Zhane_Method
def _pnp_calculator(center_point,lm_list, width, height):
        points_3d = []
        points_2d = []
        center_2d = (int(center_point[0] * width), int(center_point[1] * height))
        center_3d = (int(center_point[0] * width), int(center_point[1] * height), int(center_point[2] * 3000))

        points_3d_ = [[0, -1.126865,	7.475604],
                     [-4.445859,	2.663991,	3.173422],
                     [ 4.445859,	2.663991,	3.173422],
                     [-2.456206,	-4.342621,	4.283884],
                     [2.456206,	-4.342621,	4.283884],
                     [0,	-9.403378,	4.264492]]
        #points_3d_ = [[0, 344, -0.028813883662223816],
         #            [706, 302, 0.016568755730986595],
           #          [605, 304, 0.010903461836278439],
          #           [688, 382, -0.0027632093988358974],
          #           [633, 385, -0.00583950849249959],
          #           [656, 274, -0.0033414678182452917]]
        for i in range(len(lm_list)):
            x, y = int(lm_list[i][0] * width), int(lm_list[i][1] * height)
            # Just six values needed:
            if i <= 5:
                # Get the 2D Coordinates
                points_2d.append([x, y])
                # Get the 3D Coordinates
                points_3d.append([x, y, lm_list[i][2]])
                #points_3d.append([lm_list[i][0], lm_list[i][1], lm_list[i][2]])
        # Convert it to the NumPy array
        face_2d = np.array(points_2d, dtype=np.float64)
        face_3d = np.array(points_3d_, dtype=np.float64)
        # The camera matrix
        focal_length = 1.28 * width
        cam_matrix = np.array([[focal_length, 0, height / 2],
                                [0, focal_length, width / 2],
                                [0, 0, 1]])
        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        # Get the y rotation degree
        x =  round(angles[0], 2) # pitch/nicken
        y =  round(angles[1], 2) # roll/neigen
        z =  round(angles[2], 2) # yaw/drehen

        # xx =  round( 180*(angles[0]/pi), 2) # pitch/nicken
        # yy =  round( 180*(angles[1]/pi), 2) # roll/neigen
        # zz =  round(180*(angles[2]/pi), 2) # yaw/drehen
        print("----------------------------------------------------------------------")
        print("roll/neigen: " + str(y) + " //  pitch/nicken " + str(x) + " // yaw/drehen " + str(z) + " // ")
        # print("roll/neigen: " + str(yy) + " //  pitch/nicken " + str(xx) + " // yaw/drehen " + str(zz) + " // ")

        # visualize center line of the calculator
        # 012
        # 345
        # 678
        sy = sqrt(rmat[0][0] * rmat[0][0] + rmat[1][1] * rmat[1][1])
        R = rmat
        pitch  = round((180 * atan2(-R[2][1], R[2][2]) / pi), 2)
        roll = round((180 * atan2(-R[2][0], sy) / pi), 2)
        yaw   = round((180 * atan2(-R[0][2], R[0][0]) / pi), 2)

        # print("roll: " + str(roll) + " pitch:  " + str(pitch) + " yaw: " + str(yaw) )
        print("roll/neigen: " + str(roll) + " //  pitch/nicken " + str(pitch) + " // yaw/drehen " + str(yaw) + " // ")

        # rot_params = [roll, pitch, yaw]
        # print(rot_params)
        center_3d_projection, jacobian = cv2.projectPoints(center_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        
        return center_2d, center_3d_projection, x, y, z


def pnp_calculator(center_point, lm_list, width, height):
    points_3d = []
    points_2d = []
    center_2d = (int(center_point[0] * width), int(center_point[1] * height))
    center_3d = (int(center_point[0] * width), int(center_point[1] * height), int(center_point[2] * 100))

    points_3d_ = [[0, -1.126865, 7.475604],
                  [-4.445859, 2.663991, 3.173422],
                  [4.445859, 2.663991, 3.173422],
                  [-2.456206, -4.342621, 4.283884],
                  [2.456206, -4.342621, 4.283884],
                  [0, -9.403378, 4.264492]]

    for i in range(len(lm_list)):
        x, y = int(lm_list[i][0] * width), int(lm_list[i][1] * height)
        # Just six values needed:
        if i <= 5:
            # Get the 2D Coordinates
            points_2d.append([x, y])
            # Get the 3D Coordinates
            points_3d.append([x, y, lm_list[i][2]])
            # points_3d.append([lm_list[i][0], lm_list[i][1], lm_list[i][2]])
    # Convert it to the NumPy array
    face_2d = np.array(points_2d, dtype=np.float64)
    face_3d = np.array(points_3d_, dtype=np.float64)
    face_3d_mes = np.array(points_3d, dtype=np.float64)
    # The camera matrix
    focal_length = 1.28 * width
    cam_matrix = np.array([[focal_length, 0, height / 2],
                           [0, focal_length, width / 2],
                           [0, 0, 1]])
    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    # Calculate euler angle
    pose_mat = cv2.hconcat((rmat, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    print(euler_angles)
    x = euler_angles[0]  #  Tilt
    y = euler_angles[1]  # Gieren YAW
    z = euler_angles[2]  # '' Pitch

    # for drawing
    # success, rot_vec, trans_vec = cv2.solvePnP(face_3d_mes, face_2d, cam_matrix, dist_matrix)
    center_3d_projection, jacobian = cv2.projectPoints(center_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

    return center_2d, center_3d_projection, x, y, z


def ORIGpnp_calculator(center_point, lm_list, width, height):
    points_3d = []
    points_2d = []
    center_2d = (int(center_point[0] * width), int(center_point[1] * height))
    center_3d = (int(center_point[0] * width), int(center_point[1] * height), int(center_point[2] * 100))

    for i in range(len(lm_list)):
        x, y = int(lm_list[i][0] * width), int(lm_list[i][1] * height)
        # Just six values needed:
        if i <= 5:
            # Get the 2D Coordinates
            points_2d.append([x, y])
            # Get the 3D Coordinates
            points_3d.append([x, y, lm_list[i][2]])
    # Convert it to the NumPy array
    face_2d = np.array(points_2d, dtype=np.float64)
    face_3d = np.array(points_3d, dtype=np.float64)
    # The camera matrix
    focal_length = 1 * width
    cam_matrix = np.array([[focal_length, 0, height / 2],
                           [0, focal_length, width / 2],
                           [0, 0, 1]])
    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # r_s = R.from_matrix(rmat)
    # r_e = r_s.as_euler('xyz')
    # print(r_e)
    #Test
    #xx = rmat[0][0] * 360
    #yy = rmat[0][1] * 360
    #zz = rmat[0][2] * 360
    #print([xx, yy, zz])


    #
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    # visualize center line of the calculator
    center_3d_projection, jacobian = cv2.projectPoints(center_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

    return center_2d, center_3d_projection, x, y, z


def new_pnp_calculator(center_point, lm_list, norm_lm_List , width, height):
    points_3d = []
    points_2d = []
    center_2d = (int(center_point[0] * width), int(center_point[1] * height))
    center_3d = (int(center_point[0] * width), int(center_point[1] * height), int(center_point[2] * 100))

    points_3d_ = [[0, -1.126865, 7.475604],
                  [-4.445859, 2.663991, 3.173422],
                  [4.445859, 2.663991, 3.173422],
                  [-2.456206, -4.342621, 4.283884],
                  [2.456206, -4.342621, 4.283884],
                  [0, -9.403378, 4.264492]]

    for i in range(len(lm_list)):
        x, y = int(lm_list[i][0] * width), int(lm_list[i][1] * height)
        # Just six values needed:
        if i <= 5:
            # Get the 2D Coordinates
            points_2d.append([x, y])
            # Get the 3D Coordinates
            points_3d.append([x, y, lm_list[i][2]])
            # points_3d.append([lm_list[i][0], lm_list[i][1], lm_list[i][2]])
    # Convert it to the NumPy array
    face_2d = np.array(points_2d, dtype=np.float64)
    face_3d = np.array(norm_lm_List, dtype=np.float64)
    face_3d_ = np.array(points_3d_, dtype=np.float64)
    face_3d_mes = np.array(points_3d, dtype=np.float64)
    # The camera matrix
    focal_length = 1.28 * width
    cam_matrix = np.array([[focal_length, 0, height / 2],
                           [0, focal_length, width / 2],
                           [0, 0, 1]])
    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    # Calculate euler angle
    pose_mat = cv2.hconcat((rmat, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    # print(euler_angles)
    x = euler_angles[0]  #  Tilt
    y = euler_angles[1]  # Gieren YAW
    z = euler_angles[2]  # '' Pitch

    # for drawing
    # success, rot_vec, trans_vec = cv2.solvePnP(face_3d_mes, face_2d, cam_matrix, dist_matrix)
    center_3d_projection, jacobian = cv2.projectPoints(center_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

    return center_2d, center_3d_projection, x, y, z