import math
import numpy as np

def euclidean_distance(point1 , point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
# midpoint of two image points 
def midpoint(point1 ,point2):
    return (point1[0] + point2[0])/2,(point1[1] + point2[1])/2, (point1[2] + point2[2])/2

# calculate angle of Mid point 
# Example: Angle of Elbow -> You need points from shoulder (First), Elbow (Mid) and Wrist(End)
# angle(face_fm[6], mid_shoulder, L_shoulder)
def angle(point1,point2,point3):
    a = np.array(point1) # First (point before the one you want)
    b = np.array(point2) # Mid (point which angle you want)
    c = np.array(point3) # End (point after the one you want)
    
    radians1 = np.arctan2(c[1]-b[1], c[0]-b[0])
    radians2 = np.arctan2(a[1]-b[1], a[0]-b[0])
    rad = radians1 - radians2
    # angle1 = np.abs(radians1*180.0/np.pi)
    # angle2 = np.abs(radians2 * 180.0 / np.pi)
    angle3 = np.abs(rad*180.0/np.pi)

    if angle3>180.0:
        angle3 = 360-angle3
        
    return round(angle3, 1)


def angleHead(Chin, forehead, L_Shoulder, R_Shoulder):
    a = np.array(Chin)
    b = np.array(forehead)
    c = np.array(L_Shoulder)
    d = np.array(R_Shoulder)

    radians1 = np.arctan2(a[1]-b[1] , a[0]-b[0] )
    radians2 = np.arctan2(c[1] - d[1], c[0] - d[0])

    rad = radians1-radians2

    angle1 = np.abs(radians1 * 180.0 / np.pi)
    #print("Angle1: " + str(round(angle1, 2)))
    angle2 = np.abs(radians2 * 180.0 / np.pi)
    #print("Angle2: " + str(round(angle2, 2)))
    angleFin = np.abs(rad * 180.0 / np.pi)
    #print("Combined: " + str(round(angleFin, 2)))#
    if angleFin > 180.0:
        angleFin = 360 - angleFin#
    return round(angleFin, 1)
