import numpy as np
import cv2
import math

def CreateDetector() -> tuple:
    # set up Aruco Detector and which marker library see https://chev.me/arucogen/
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # we will use markers 0 and 1 in this dict
    arucoParams = cv2.aruco.DetectorParameters() # use default detect params
    detector = cv2.aruco.ArucoDetector(arucoDict,arucoParams) #define detector object
    return(arucoDict,arucoParams,detector)

def GetRelativeYaw(corners: np.ndarray) -> float:
    x1= int(corners[0][0])
    y1= int(corners[0][1])
    x2= int(corners[2][0])
    y2= int(corners[2][1])
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle
