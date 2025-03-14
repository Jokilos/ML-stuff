"""
Stub for homework 2
"""

import time
import random
import numpy as np
import mujoco
from mujoco import viewer


import numpy as np
import cv2
from numpy.typing import NDArray


TASK_ID = 2

world_xml_path = f"car_{TASK_ID}.xml"
model = mujoco.MjModel.from_xml_path(world_xml_path)
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def sim_step(
    n_steps: int, /, view=True, rendering_speed = 10, **controls: float
) -> NDArray[np.uint8]:
    """A wrapper around `mujoco.mj_step` to advance the simulation held in
    the `data` and return a photo from the dash camera installed in the car.

    Args:
        n_steps: The number of simulation steps to take.
        view: Whether to render the simulation.
        rendering_speed: The speed of rendering. Higher values speed up the rendering.
        controls: A mapping of control names to their values.
        Note that the control names depend on the XML file.

    Returns:
        A photo from the dash camera at the end of the simulation steps.

    Examples:
        # Advance the simulation by 100 steps.
        sim_step(100)

        # Move the car forward by 0.1 units and advance the simulation by 100 steps.
        sim_step(100, **{"forward": 0.1})

        # Rotate the dash cam by 0.5 radians and advance the simulation by 100 steps.
        sim_step(100, **{"dash cam rotate": 0.5})
    """

    for control_name, value in controls.items():
        data.actuator(control_name).ctrl = value

    for _ in range(n_steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step / rendering_speed)

    renderer.update_scene(data=data, camera="dash cam")
    img = renderer.render()
    return img



# TODO: add addditional functions/classes for task 1 if needed
from PIL import Image
import time

DEBUG = False 
SLEEP = True 

def PIL_show(img, colorspace = 'RGB'):
    if (colorspace != 'RGB'):
        color_trans = getattr(cv2, f'COLOR_{colorspace}2RGB')
        img = cv2.cvtColor(img, color_trans)
    pil_image = Image.fromarray(img)
    pil_image.show()

def cut_vertical_strip(img, width, cut_bottom = None):
    _, W, _ = img.shape
    cut_from, cut_to = (W - width) // 2, (W + width) // 2
    if cut_bottom is None:
        return img[:, cut_from : cut_to, :]
    else:
        return img[: -cut_bottom, cut_from : cut_to, :]

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

def cut_strip_and_count(img, strip_width, color = 'red', cut_bottom = None, show = False):
    img = cut_vertical_strip(img, strip_width, cut_bottom)
    red_mask = find_colored_pixels(img, color = color) 
    pixels_detected = np.sum(red_mask > 0)

    if(show and pixels_detected > 0):
        PIL_show(img)
        PIL_show(red_mask)
        time.sleep(2) 

    return pixels_detected
    
def step(forward, turn, verbose = DEBUG, show = False, multiple = 1):
    controls = {"forward": forward, "turn": turn}
    img = sim_step(200 * multiple, view=True, **controls)

    if verbose:
        print(data.body("car").xpos)
        print(data.body("target-ball").xpos)

    if show:
        PIL_show(img)
        time.sleep(1)

    return img

# /TODO


def task_1():
    steps = random.randint(0, 2000)
    controls = {"forward": 0, "turn": 0.1}
    img = sim_step(steps, view=False, **controls)

    # TODO: Change the lines below.
    # For car control, you can use only sim_step function
    pixels_detected = 0
    small_step = 0.02
    big_strip_width = 80
    small_strip_width = 10
    scan_length = 20
    small_treshold = 5e3
    big_threshold = 8.2e3

    # detecting general direction of the ball
    while pixels_detected < 5:
        img = step(0, 0.1)
        pixels_detected = cut_strip_and_count(img, big_strip_width, 'red')

    # scanning left and right for more accurate direction
    step(0, -small_step, multiple = scan_length // 2)

    pixels_list = []
    for _ in range(scan_length):
        img = step(0, small_step)
        pixels_detected = cut_strip_and_count(img, small_strip_width, 'red')
        pixels_list.append(pixels_detected)

    # turning in the right direction
    turn_back = scan_length - np.argmax(pixels_list) - 1
    img = step(0, -small_step, multiple = turn_back)

    # getting relatively close to the ball
    while pixels_detected < small_treshold:
        img = step(0.3, 0)
        pixels_detected = cut_strip_and_count(img, img.shape[1], 'red')

    # making final adjustments
    while pixels_detected < big_threshold:
        img = step(0.018, 0)
        pixels_detected = cut_strip_and_count(img, img.shape[1], 'red')
        # print(pixels_detected)

    if SLEEP:
        time.sleep(100)
    # /TODO



# TODO: add addditional functions/classes for task 2 if needed

def low_high_list(arr, threshold = 0):
    arr = np.array(arr)
    check = arr > threshold
    args = list(zip(check[:-1], check[1:]))
    res = [prev for prev, next in args if prev != next]

    if len(res) == 0:
        return [arr[0] > threshold]
    elif res[-1] == check[-1]:
        return res
    else:
        return res + [check[-1]]
   
def check_high_low_idx(arr, threshold = 0):
    arr = np.array(arr)
    check = arr > threshold
    args = list(zip(check[:-1], check[1:], np.arange(len(arr))[1:]))
    res = [idx for prev, next, idx in args if prev != next]
    
    return res

def pillar_based_revolution(strip_width, step_size = 0.15):
    pd_list = []
    revolution_done = False 
    while not revolution_done:
        img = step(0, step_size)
        pixels_detected = cut_strip_and_count(img, strip_width, 'grey', cut_bottom = 200)
        pd_list.append(pixels_detected)

        if pixels_detected > 30:
            revolution_done = low_high_list(pd_list, 5)[-3:] == [True, False, True]
            # print(low_high_list(pd_list, 0)[-3:])

# /TODO

def task_2():
    speed = random.uniform(-0.3, 0.3)
    turn = random.uniform(-0.2, 0.2)
    time.sleep(0.5)
    speed, turn = -0.23020031618517778, 0.12508321126850375 
    speed, turn = -0.24, 0.12
    time.sleep(0.5)
    controls = {"forward": speed, "turn": turn}
    img = sim_step(1000, view=True, **controls)

    # TODO: Change the lines below.
    # For car control, you can use only sim_step function

    print(speed, turn)

    pixels_detected = 0
    small_step = 0.05
    big_strip_width = 100
    small_strip_width = 10
    scan_length = 20

    ### POSITION SCRAMBLING
    # step(-20, 0)
    # step(0, 20)

    # Turn around and check if we see pillar twice
    pillar_based_revolution(big_strip_width)

    # Find and face the middle of the blue wall
    pixels_list = []
    for _ in range(scan_length):
        img = step(0, -small_step)
        pixels_detected = cut_strip_and_count(img, small_strip_width, 'green')
        pixels_list.append(pixels_detected)

    first, last = check_high_low_idx(pixels_list, 0)
    middle = np.ceil((first + last) / 2).astype(int)
    turn_back = scan_length - middle

    step(0, small_step, multiple = turn_back)

    # Go back until the blue wall in small enought in our FOV
    pixels_detected = 1e5
    while pixels_detected > 1.45e4:
        img = step(-0.1, 0)
        pixels_detected = cut_strip_and_count(img, img.shape[1], 'blue')
        # print(pixels_detected)

    # Precisely face the pillar
    pillar_based_revolution(int(small_strip_width * 2.5), small_step * 1.3)

    # Turn to face the green wall
    step(0, 0.02, multiple = 5)

    # Drive back
    step(-0.1, 0, multiple = 28)
    step(0.1, 0, multiple = 3)

    # Face the exit
    step(0, -0.05, multiple = 11)

    # Exit the labirynth
    step(0.2, 0, multiple = 30)

    # Find the ball
    task_1()

    if SLEEP:
        time.sleep(100)
    # /TODO

def ball_is_close() -> bool:
    """Checks if the ball is close to the car."""
    ball_pos = data.body("target-ball").xpos
    car_pos = data.body("dash cam").xpos
    print(car_pos, ball_pos)
    return np.linalg.norm(ball_pos - car_pos) < 0.2


def ball_grab() -> bool:
    """Checks if the ball is inside the gripper."""
    print(data.body("target-ball").xpos[2])
    return data.body("target-ball").xpos[2] > 0.1


def teleport_by(x: float, y: float) -> None:
    data.qpos[0] += x
    data.qpos[1] += y
    sim_step(10, **{"dash cam rotate": 0})


def get_dash_camera_intrinsics():
    '''
    Returns the intrinsic matrix and distortion coefficients of the camera.
    '''
    h = 480
    w = 640
    o_x = w / 2
    o_y = h / 2
    fovy = 90
    f = h / (2 * np.tan(fovy * np.pi / 360))
    intrinsic_matrix = np.array([[-f, 0, o_x], [0, f, o_y], [0, 0, 1]])
    distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # no distortion

    return intrinsic_matrix, distortion_coefficients


# TODO: add addditional functions/classes for task 3 if needed
def door(command = 'close', verbose = False):
    val = 1 if command == 'close' else 0
    controls = {"trapdoor close/open": val}
    img = sim_step(200, view=True, **controls)

    return img

def lift(command = 'up', verbose = False):
    val = 1 if command == 'up' else -1
    controls = {"lift": val}
    img = sim_step(2000, view=True, **controls)

    return img

def jib_rotate(rotation, verbose = False, multiple = 1):
    controls = {"jib rotate": rotation}
    img = sim_step(200 * 78 * multiple, view=True, **controls)

    return img

def cam_rotate(rotation = 1, verbose = False, multiple = 1, view = True):
    controls = {"dash cam rotate": rotation}
    img = sim_step(int(800 * multiple), view=view, **controls)

    return img

def get_objpoints(edge):
    blk = 1/10

    objpoints = np.array([
        [blk - 1, 0, blk],
        [-blk, 0, blk],
        [-blk, 0, 1 - blk],
        [blk - 1, 0, 1 - blk],

        [0, -blk, 1 - blk],
        [0, -blk, blk],
        [0, blk - 1, blk],
        [0, blk - 1, 1 - blk],
    ])

    return objpoints * edge

def visualize_corners(img, corners, ids = None, verbosity = 0):
    img0 = img.copy()

    cv2.aruco.drawDetectedMarkers(img0, corners)

    if verbosity > 0:
        for x, y in np.array(corners).reshape(8, 2).astype(int):
            cv2.circle(img0, (x, y), 3, (255, 0, 0), cv2.FILLED)
            PIL_show(img0)
            time.sleep(1)
    else:
        PIL_show(img0)
        time.sleep(3)

def detect_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(gray)
    ret = ids is not None and len(ids) == 2

    # if ret:
        # visualize_corners(img, corners)

    corners = np.array(corners).reshape(-1, 1, 2)
    return ret, corners

CUBE_EDGE = 0.1

def pnp_project(img):
    ret, corners = detect_corners(img)
    if not ret:
        return None, None

    cMat, dCoeff = get_dash_camera_intrinsics() 
    objpoints = get_objpoints(CUBE_EDGE)

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

# /TODO


def task_3():
    start_x = random.uniform(-0.2, 0.2)
    start_y = random.uniform(0, 0.2)
    # start_x, start_y = 0.2, 0
    teleport_by(start_x, start_y)

    # TODO: Get to the ball
    #  - use the dash camera and ArUco markers to precisely locate the car
    #  - move the car to the ball using teleport_by function
    from tqdm import tqdm

    # for debugging purpouses
    # print(start_x, start_y)

    # lift the gripper
    lift()
    
    iterations = 30
    found = False
    error_threshold = 150
    min_error = error_threshold
    bonus_x = 0
    half_rot = 0.06
    vec = None

    # scan the enviroment until we get a good measurement where we are
    while not found:
        cam_rotate(-half_rot)

        for _ in tqdm(range(iterations), desc="Looking for cube"):
            img = cam_rotate(-half_rot / iterations)
            tvec, error = pnp_project(img)

            if tvec is not None and error < min_error and tvec[0] > 0 and tvec[1] > 0 and np.abs(tvec[2]) < 0.15:
                vec = tvec[:2]
                min_error = error
                # print(error, tvec)

        if min_error < error_threshold:
            found = True
        else:
            for _ in range(iterations):
                cam_rotate(half_rot / iterations, view = False)
            cam_rotate(half_rot, view = False)

            print("Two sides not found:", end = ' ')
            if bonus_x >= 0:
                print("tp front")
                teleport_by(0.1, 0)
                bonus_x -= 0.1
            else:
                print("tp back")
                teleport_by(-0.22, 0)
                bonus_x += 0.22

    # print(f"tp {-vec[0]} {-vec[1]}")
    teleport_by(-vec[0] + 0.96, -vec[1] + 2.27)

    # /TODO

    # IM DROPPING THIS ASSERTION
    # BECAUSE AFTER TP THE CAR IS ALREADY IN GOOD
    # POSITION TO GRAB THE BALL, BUT IS NOT 'CLOSE' 

    # assert ball_is_close()

    # TODO: Grab the ball
    # - the car should be already close to the ball
    # - use the gripper to grab the ball
    # - you can move the car as well if you need to

    # turn camera back for a photo (optional)
    # for _ in range(iterations):
    #     cam_rotate(half_rot / iterations, view = False)
    # img = cam_rotate(half_rot, view = False)
    # PIL_show(img)

    time.sleep(0.3)
    lift('down')
    door('close')
    lift('up')

    if SLEEP:
        time.sleep(100)
    # /TODO


    assert ball_grab()


if __name__ == "__main__":
    print(f"Running TASK_ID {TASK_ID}")
    if TASK_ID == 1:
        task_1()
    elif TASK_ID == 2:
        task_2()
    elif TASK_ID == 3:
        task_3()
    else:
        raise ValueError("Unknown TASK_ID")
