import mujoco
import random
import numpy as np
import math
import cv2
from typing import Tuple
from PIL import Image
import sys
import time

def PIL_show(img, colorspace = 'RGB'):
    if (colorspace != 'RGB'):
        color_trans = getattr(cv2, f'COLOR_{colorspace}2RGB')
        img = cv2.cvtColor(img, color_trans)
    pil_image = Image.fromarray(img)
    pil_image.show()

def find_colored_pixels(img, color = 'red'):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower2 = upper2 = None

    if color == 'red':
        lower1 = np.array([0, 50, 50])    
        upper1 = np.array([10, 255, 255]) 
        lower2 = np.array([170, 50, 50])  
        upper2 = np.array([180, 255, 255])

    if color == 'grey':
        lower1 = np.array([ 0, 0, 66])
        upper1 = np.array([ 10, 50, 166])

    if color == 'green':
        lower1 = np.array([35, 50, 50])
        upper1 = np.array([85, 255, 255])

    if color == 'blue':
        lower1 = np.array([110, 138, 65])
        upper1 = np.array([130, 255, 255])

    mask = cv2.inRange(hsv_img, lower1, upper1)

    if lower2 is not None:
        mask2 = cv2.inRange(hsv_img, lower2, upper2)
        mask = cv2.bitwise_or(mask, mask2)

    return mask

def cut_vertical_strip(img, width, cut_bottom = None):
    _, W, _ = img.shape
    cut_from, cut_to = (W - width) // 2, (W + width) // 2
    if cut_bottom is None:
        return img[:, cut_from : cut_to, :]
    else:
        return img[: -cut_bottom, cut_from : cut_to, :]

def cut_strip_and_count(img, strip_width, color = 'red', cut_bottom = None, show = False):
    img = cut_vertical_strip(img, strip_width, cut_bottom)
    red_mask = find_colored_pixels(img, color = color) 
    pixels_detected = np.sum(red_mask > 0)

    if(show and pixels_detected > 0):
        PIL_show(img)
        PIL_show(red_mask)
        time.sleep(2) 

    return pixels_detected

def shape_detection(img, gray, verbose = False):
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
  
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    if verbose:
        print(f'found {len(contours)} contours')

    for contour in contours: 
   
        approx = cv2.approxPolyDP( 
            contour, 0.01 * cv2.arcLength(contour, True), True) 

        if verbose:  
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5) 

        if verbose:
            print(len(approx))

        if len(approx) < 10: 
            return 'box' 
        else: 
            return 'sphere'

class Detector:
    def __init__(self) -> None:
        self.saw_shape = False
        self.prediction = None
        self.img = None
        self.shape = "" 

    def detect(self, img) -> None:
        verbose = False 

        if not self.saw_shape:

            pixels_detected = cut_strip_and_count(img, strip_width = 50)
            red_mask = find_colored_pixels(img, color = 'red')

            if verbose:
                print(pixels_detected, end = ' ')

            if(pixels_detected > 100):
                self.saw_shape = True

                if verbose:
                    PIL_show(red_mask)

                self.shape = shape_detection(img, red_mask)

                if verbose:
                    print(self.shape)
                    PIL_show(img)

    def result(self) -> str:
        return self.shape 

def detect_corners(gray, verbose = False):
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
  
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    if verbose:
        print(f'found {len(contours)} contours')

    contours = np.array([list(c) for c in contours])

    return contours 

def get_dash_camera_intrinsics():
    '''
    Returns the intrinsic matrix and distortion coefficients of the camera.
    '''
    h = 480
    w = 640
    o_x = w / 2
    o_y = h / 2
    f = 200 
    intrinsic_matrix = np.array([[f, 0, o_x], [0, f, o_y], [0, 0, 1]])
    distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # no distortion

    return intrinsic_matrix, distortion_coefficients

def get_objpoints(edge = 0.1):

    objpoints = np.array([
        [0, 10, 0],
        [0, 12, 0],
        [2, 12, 0],
        [2, 10, 0],

        [0, 0, 0],
        [0, 2, 0],
        [2, 2, 0],
        [2, 0, 0],
    ])

    return objpoints * edge
        
def pnp_project(red_mask):
    corners = detect_corners(red_mask, verbose = True)
    print(corners)
    corners = np.array(corners)
    corners = corners.astype(np.float32).reshape(-1, 1, 2)

    print(corners.shape)

    cMat, dCoeff = get_dash_camera_intrinsics() 
    objpoints = get_objpoints()

    ret, rvec, tvec, *_ = cv2.solvePnP(
        objectPoints = objpoints,
        imagePoints = corners,
        cameraMatrix = cMat,
        distCoeffs = dCoeff,
    )

    if not ret:
        return None, None

    image_points, _ = cv2.projectPoints(
        objectPoints = objpoints,
        rvec = rvec,
        tvec = tvec,
        cameraMatrix = cMat,
        distCoeffs = dCoeff
    )

    error = np.linalg.norm(corners - image_points.squeeze(), axis=1)
    mean_error = np.mean(error)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    tvec_world_frame = -np.dot(rotation_matrix.T, tvec)

    return tvec_world_frame.flatten(), mean_error

class DetectorPos:
    def __init__(self) -> None:
        pass

    def detect(self, img) -> Tuple[float, float]:
        pos00 = np.array([0.22684832,  0.5507112, -0.87000681])

        red_mask = find_colored_pixels(img, color = 'red')

        tvec, error = pnp_project(red_mask)

        print(tvec - pos00)

        # tvec - pos00 is the value we're looking for
        # but the code is not working
        # i gave it a try
        # the function suppoused to find contours doesn't work
        # and i don't have the time to debug

        return 0, 0
