#!/usr/bin/env python
# coding: utf-8

# # Enabling easy image printing in ipynb notebook

import cv2
import numpy as np

print(f"OpenCV version is: {cv2.__version__}")

if True:  # change to True if you want to use the notebook locally
    # and use cv2_imshow from matplotlib (eg. Vscode)
    import matplotlib.pyplot as plt

    def cv2_imshow(img):
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

# # Task 1
# # Code i used to confirm what are the ids of aruco tags 

GREEN = (0, 255, 0)
RED = (0, 0, 255)

def draw_quadrilateral(img, p_list, color, girth):
    shifted = np.concat([p_list[-1:],p_list[:-1]]).astype(int)
    zip_list = list(zip(p_list, shifted))

    for (p1, p2) in zip_list:
        cv2.line(img, p1, p2, color, girth)

def draw_square(img, p1, p3, color, girth):
    p_matrix = np.vstack([p1, p3])
    new = (p_matrix * np.eye(2), p_matrix * np.array([0,1,1,0]).reshape(2,2))

    p2, p4 = list(map(lambda x: x[np.nonzero(x)], new))
    p4 = p4[::-1]

    p_list = np.array([p1, p2, p3, p4]).astype(int)
    draw_quadrilateral(img, p_list, color, girth)

def drawMarkers(img, corners, ids):
    if ids is None:
        return

    square_size = 7
    for i in range(ids.shape[0]):
        c = corners[i][0]
        id = ids[i][0]
        draw_quadrilateral(img, c.astype(int), GREEN, 1)
        draw_square(img, c[0] - (square_size // 2), c[0] + (square_size // 2), RED, 1)
        cv2.putText(img, f"{id}", c[0].astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

def input_parse(img = None, filename = None):
    if filename is not None:
        current = cv2.imread(filename)
    elif img is not None:
        current = img
    else:
        assert False

    return current

def inputs_parse(imgs = None, filenames = None):
    if filenames is not None:
        return [cv2.imread(f) for f in filenames]
    elif imgs is not None:
        return imgs
    else:
        assert False

def get_objpoints(shape, width, uniform_grid = True):
    x, y = shape

    if uniform_grid:
        arr1 = np.linspace(0, width * x, x + 1), 
        arr2 = np.linspace(0, width * y, y + 1)
    else:
        arr1, arr2 = [0], [0]
        for i in range(x):
            arr1.append(arr1[i] + width[i % len(width)])
        for i in range(y):
            arr2.append(arr2[i] + width[i % len(width)])

    xx, yy = np.meshgrid(arr1, arr2)
    xx, yy = xx.reshape(-1), yy.reshape(-1)
    objarr = np.vstack([xx, yy, np.zeros(len(xx))]).T

    return objarr.astype(np.float32)

def get_marker_objp(width):
    objp = get_objpoints((1,1), width)
    objp[[2,3]] = objp[[3,2]]
    return objp

# # Functions i use to retrieve properly permuted grid image point list

# i noticed it's sorted so i didn't have to use it
IDS_ORDERING = [29, 28, 24, 23, 19, 18]

def permute_ids_corners(ids, corners):
    ids = ids.reshape(-1)

    ids_order = np.argsort(ids)
    ids = ids[ids_order][::-1]
    corners = corners[ids_order][::-1] 

    return ids.reshape(-1, 1), corners

def find_aruco_info(img = None, filename = None, draw = False):
    img = input_parse(img, filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = np.array(list(corners)).reshape(-1,1,2)
    corners_sub = cv2.cornerSubPix(gray, corners_sub, (5, 5), (-1, -1), criteria)
    corners_sub = corners_sub.reshape(-1,4,1,2)

    if draw:
        corners_sub = tuple(corners_sub.reshape(-1,1,4,2))
        img_ = img.copy()
        drawMarkers(img_, corners, ids)
        cv2_imshow(img_)

    if np.any(IDS_ORDERING != ids.reshape(-1)):
        ids, corners_sub = permute_ids_corners(ids, corners_sub)

    return (corners_sub, ids)

def permute_corners(corners):
    corners = corners.reshape(-1,1,2)
    perm = np.array([0,1,4,5,3,2,7,6])
    coords = np.arange(len(perm))

    for i in range(3):
        corners[coords + (i * len(perm))] = corners[perm + (i * len(perm))]

    return corners

TAG_SIDE=168
SPACING=70
TAG_GRID=(3,5)

def calibrate(img = None, filename = None, use_grid = False):
    current = input_parse(img, filename)

    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    corners = find_aruco_info(img = current)[0]
    if use_grid:
        corners = [permute_corners(corners)]
        objpoints = [get_objpoints(TAG_GRID, [TAG_SIDE, SPACING], False)]
    else:
        objpoints = [get_marker_objp(TAG_SIDE)] * 6

    calibration_res = cv2.calibrateCamera(
        objectPoints = objpoints,
        imagePoints = corners,
        imageSize = gray.shape[::-1],
        cameraMatrix = None,
        distCoeffs = None,
    )

    return (
        calibration_res,
        objpoints,
        corners,
    )

def get_projection_error(cMat, dCoeff, rvecs, tvecs, objpoints, imgpoints):
    total_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            cMat,
            dCoeff,
        )

        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(objpoints[i]) 

    mean_error = np.sqrt(total_error / total_points)
    return mean_error

def get_undistort_rectify(img = None, filename = None, use_grid = False, alpha = 0):
    current = input_parse(img, filename)

    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]

    cal = calibrate(img = current, use_grid = use_grid)
    ((_, cMat, dCoeff, rvecs, tvecs), oP, iP) = cal

    projection_error = get_projection_error(cMat, dCoeff, rvecs, tvecs, oP, iP)

    newCMat, _ = cv2.getOptimalNewCameraMatrix(
        cameraMatrix = cMat,
        distCoeffs = dCoeff,
        imageSize = size,
        alpha = alpha,
        newImgSize = size,
    )

    undistort, rectify = cv2.initUndistortRectifyMap(
        cameraMatrix = cMat,
        distCoeffs = dCoeff,
        R = np.eye(3),
        newCameraMatrix = newCMat,
        size = size,
        m1type = cv2.CV_32FC1,
    )

    return (undistort, rectify, projection_error)

def undistort(img = None, filename = None, use_grid = False, alpha = 0):
    current = input_parse(img, filename)

    (undistort, rectify, err) = get_undistort_rectify(
        img = current,
        use_grid = use_grid,
        alpha = alpha,
    )

    print(f"Mean reprojection error : {err}")

    undistorted = cv2.remap(
        src = current,
        map1 = undistort,
        map2 = rectify,
        interpolation = cv2.INTER_LINEAR,
    )

    return undistorted

# # Comparing calibration on a single image
def task1_snippet():
    used_img = 'data1/img2.png'

    print("Original image:")
    cv2_imshow(cv2.imread(used_img))
    print("With grid usage:")
    img = undistort(filename = used_img, use_grid = True)
    cv2.imwrite('task_solutions/task1_solution_grid.png', img)
    print("Without grid usage:")
    img = undistort(filename = used_img, use_grid = False)
    cv2.imwrite('task_solutions/task1_solution_no_grid.png', img)


# # Comment on the results
# 
# I checked if the lines are straight on images produced by both methods and on the original image.
# In my opinion, the image produced with all available information is clearly superior.
# The edges of the board are really straight, and when you flip between undistorded and original image, you can actually see the distosion.
# 
# However, that method also results with higher reprojection error.
# This might be because i made a mistake, but on the other hand it also makes sense.
# It is easier to make 6 different transformations to fit 6 sets of points, than making one that should fit all the points combined.

# # Task 2

# # Calibrate camera on all images using all available data (grid method)

def calibrate_all():
    TAG_SIDE=168
    SPACING=70
    TAG_GRID=(3,5)

    all_files = [f'data1/img{i}.png' for i in range(1,29)]
    imgs = [cv2.imread(f) for f in all_files]

    corners = [find_aruco_info(img = i)[0] for i in imgs]
    permuted = [permute_corners(cor) for cor in corners]

    objpoints = [get_objpoints(TAG_GRID, [TAG_SIDE, SPACING], False)]
    objpoints = objpoints * len(imgs)

    inliners = []
    for o, p in list(zip(objpoints, permuted)):
        calibration_res = cv2.calibrateCamera(
            objectPoints = [o],
            imagePoints = [p],
            imageSize = imgs[0].shape[:2][::-1],
            cameraMatrix = None,
            distCoeffs = None,
        )

        if(calibration_res[0] < 0.8):
            inliners.append((o, p))

    obj, perm = zip(*inliners)    
    calibration_res = cv2.calibrateCamera(
        objectPoints = obj,
        imagePoints = perm,
        imageSize = imgs[0].shape[:2][::-1],
        cameraMatrix = None,
        distCoeffs = None,
    )
    print(calibration_res[0])
    return calibration_res[1:3]

def undistort_all():
    cMat, dCoeff = calibrate_all()

    all_files = [f'data2/img{i}.png' for i in range(1,10)]
    imgs = [cv2.imread(f) for f in all_files]
    size = imgs[0].shape[:2][::-1]

    newCMat, _ = cv2.getOptimalNewCameraMatrix(
        cameraMatrix = cMat,
        distCoeffs = dCoeff,
        imageSize = size,
        alpha = 1,
        newImgSize = size,
    )

    undistort, rectify = cv2.initUndistortRectifyMap(
        cameraMatrix = cMat,
        distCoeffs = dCoeff,
        R = np.eye(3),
        newCameraMatrix = newCMat,
        size = size,
        m1type = cv2.CV_32FC1,
    )

    for current, name in list(zip(imgs, all_files)):
        new_name = 'data3' + name[5:]

        undistorted = cv2.remap(
            src = current,
            map1 = undistort,
            map2 = rectify,
            interpolation = cv2.INTER_LINEAR,
        )

        cv2.imwrite(new_name, undistorted) 

def find_span(coords):
    col = coords[:, 0]
    xspan = [np.min(col), np.max(col)]
    col = coords[:, 1]
    yspan = [np.min(col), np.max(col)]
    return np.vstack([xspan, yspan])

def multiply_normalize(mA, mB):
    res = mA @ mB 
    last_row = res[-1, :] 
    res_norm = res / last_row

    return res_norm

def multiply_coords_matrix(range_x, range_y, matrix):
    xx, yy = np.meshgrid(np.arange(*range_x), np.arange(*range_y))
    length = (range_x[1] - range_x[0]) * (range_y[1] - range_y[0])
    stack_list = [xx.flatten(), yy.flatten(), np.ones(length)]
    xy1_vectors = np.stack(stack_list, axis=1).astype(int)

    new_coords = multiply_normalize(matrix, xy1_vectors.T)
    new_coords = np.round(new_coords).astype(int)
    return new_coords

def transform(tx, ty, matrix):
    transform_matrix = np.array([1,0,tx,0,1,ty,0,0,1]).reshape(3,3)
    new_coords = transform_matrix @ matrix
    last_row = new_coords[-1, :] 
    new_coords = np.round(new_coords / last_row).astype(int)
    return new_coords

def pixel_list_to_img(w, h, array):
    array = array.reshape(w, h, 3)
    return array.astype(np.uint8)

def cut_to_dimensions(width, heigth, matrix):
    xs, ys = matrix[:, 0], matrix[:, 1]
    xl, yl = xs < 0, ys < 0
    xu, yu = xs >= width, ys >= heigth
    cond = np.any(np.vstack([xl, yl, xu, yu]), axis = 0)
    matrix[cond] = [0, heigth]

    return matrix.astype(int)

def find_colors(point_coords, img):
    _, w, _ = img.shape
    black_row = np.zeros((1, w, 3), dtype=np.uint8)
    img = np.vstack([img, black_row])
    x_coords, y_coords = point_coords[:, 0], point_coords[:, 1]

    return img[y_coords, x_coords]

def apply_homography(homography, filename = None, img = None):
    current = input_parse(img, filename)

    h, w, _ = current.shape

    new_coords = multiply_coords_matrix([0,w], [0,h], homography).T 

    span = find_span(new_coords)
    new_h, new_w = list(span[:, 1] - span[:, 0])

    inv_homography = np.linalg.inv(homography)
    original_cords = multiply_coords_matrix(span[0, :], span[1, :], inv_homography)
    original_cords = original_cords.T
    original_cords = cut_to_dimensions(w, h, original_cords[:, [0,1]])
    
    new_colors = find_colors(original_cords, current)
    new_image = pixel_list_to_img(new_w, new_h, new_colors)

    return (span, new_image)

def img_identity(img):
    current = img
    h, w, _ = current.shape

    new_coords = multiply_coords_matrix(w, h, np.eye(3)).T
    new_colors = find_colors(new_coords, current)
    new_image = pixel_list_to_img(w, h, new_colors)

    cv2_imshow(new_image)

# # Task 3

def intertwine(even, odd):
    if isinstance(even, np.ndarray):
        length = 2 * np.array(even).shape[0] 
    elif isinstance(odd, np.ndarray):
        length = 2 * np.array(odd).shape[0] 
    else:
        return None

    matrix = np.zeros(length)
    matrix[0::2] = even
    matrix[1::2] = odd
    return matrix.reshape(-1,1)

def find_A_matrix(source_points, dest_points):
    xs, ys = source_points[:, 0], source_points[:, 1]
    xd, yd = dest_points[:, 0], dest_points[:, 1]
    n, _ = source_points.shape 
    A_rows = [
        intertwine(xs, 0),
        intertwine(ys, 0),
        intertwine(1, np.zeros(n)),
        intertwine(0, xs),
        intertwine(0, ys),
        intertwine(np.zeros(n), 1),
        intertwine(-1 * xd * xs, -1 * yd * xs),
        intertwine(-1 * xd * ys, -1 * yd * ys),
        intertwine(-1 * xd, -1 * yd)
    ]

    A_matrix = np.hstack(A_rows)
    return A_matrix

def find_homography(source_points, dest_points):
    A_matrix = find_A_matrix(source_points, dest_points)

    _, _, V = np.linalg.svd(A_matrix)
    smallest_eingenvector = V[-1, :]
    homography = smallest_eingenvector.reshape(3, 3)

    # normalize
    return homography / homography[-1, -1]

def translation_M(tx, ty):
    return np.array([1,0,tx,0,1,ty,0,0,1]).reshape(3,3)

def rot_M(axis, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    
    if axis == 0:
        a = [1, 0, 0, 0, c, -s, 0, s, c]
    elif axis == 1:
        a = [c, 0, -s, 0, 1, 0, s, 0, c]
    elif axis == 2:
        a = [c, -s, 0, s, c, 0, 0, 0, 1]
    else:
        return None
    return np.array(a).reshape(3,3)

def scale_M(sx, sy):
    return np.array([sx,0,0,0,sy,0,0,0,1]).reshape(3,3)

def random_homography():
    srange = [0.3, 1.7]
    sx, sy = np.random.uniform(*srange), np.random.uniform(*srange)
    
    trange = [-200, 200]
    tx, ty = np.random.uniform(*trange), np.random.uniform(*trange)

    hrange = [0.5e-3, 1e-4]    
    h31, h32 = np.random.uniform(*hrange), np.random.uniform(*hrange)
    h31, h32 = h31 * np.random.choice([-1,1]), h32 * np.random.choice([-1,1])

    axis = np.random.choice([0,1,2])
    phi = np.random.uniform(0, 2 * np.pi)

    ops = [
        scale_M(sx, sy),
        translation_M(tx, ty),
        rot_M(axis, phi)
    ]

    homography = np.eye(3)
    for i in np.random.permutation(3):
        homography @= ops[i]

    homography[2, :] = [h31, h32, 1]

    lower, upper = 0.1, 10
    det = np.abs(np.linalg.det(homography))
    if upper > det > lower:
        return homography
    else:
        return random_homography()

# # Task 2 snippet
def task2_snippet():
    imgpath = 'data2/img1.png'
    random = random_homography()
    print(random)
    cv2_imshow(cv2.imread(imgpath))
    _, img = apply_homography(filename=imgpath, homography = random)
    cv2.imwrite('task_solutions/task2_solution_img.png', img)

def test_find_homography(tests = 10, points = 4):
    for _ in range(tests):
        prange = [0, 100]
        x = np.random.uniform(*prange, points).astype(int)
        y = np.random.uniform(*prange, points).astype(int)
        source = np.vstack([x, y])

        RH = random_homography()
        dest = multiply_normalize(RH, np.vstack([source, np.ones(points)]))

        FH = find_homography(source.T, dest.T)

        error = cv2.norm(RH, FH, cv2.NORM_L2)
        print(error)
        assert error < 1e-5

def task3_snippet():
    test_find_homography()

def fun_to_inspect_pixels():
    path = 'manual_stitching/img1.png'
    img = cv2.imread(path)
    print(img.shape)
    cv2.imshow(path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # Task 4

def task4_snippet():
    points_right = [801, 576, 834, 589, 921, 360, 1093, 184, 1105, 309, 456, 420, 514, 372]
    points_right = np.array(points_right).reshape(-1,2)
    points_left = [909, 588, 945, 602, 1037, 366, 1228, 179, 1243, 312, 565, 425, 621, 378]
    points_left = np.array(points_left).reshape(-1,2)
    print(find_homography(points_right, points_left))

    return points_right, points_left

def find_optimal_seam(L, R, L_margin, R_margin):
    abs = np.abs(L - R)

    # approximately: 0.3 Red + 0.59 Green + 0.11 * Blue
    grey = 0.11 * abs[:, :, 0] + 0.59 * abs[:, :, 1] + 0.3 * abs[:, :, 2]
    h, w = grey.shape
    grey[:, np.arange(L_margin + 20)] = 255
    grey[:, np.arange(R_margin - 20, w)] = 255

    # cv2_imshow(grey.astype(np.uint8))
    grey **= 2
    grey = grey.astype(int)

    for y in range(1, h):
        grey[y, [0, w - 1]] += grey[y - 1, [0, w - 1]]

    for y in range(1, h):
        for x in range(1, w - 1):
            grey[y][x] += np.min(grey[y - 1, [x - 1, x, x + 1]])

    xmin = np.zeros(h, dtype=int)
    xmin[-1] = np.argmin(grey[h - 1])

    for y in range(h - 1)[::-1]:
        prevmin = xmin[y + 1]
        findmin = np.argmin(grey[y][prevmin - 1:prevmin + 2])
        xmin[y] = prevmin + findmin - 1

    return xmin

def merge_spans(L, R):
    first_col = np.min(np.vstack([L[:, 0], R[:, 0]]), axis = 0)
    second_col = np.max(np.vstack([L[:, 1], R[:, 1]]), axis = 0)

    return np.vstack([first_col, second_col]).T

def subtract_spans(L, R, size):
    res = L - R
    res[:, 1] += size[::-1]
    return res

def put_on_image(img, data, span):
    x1, x2 = span[0, :]
    y1, y2 = span[1, :]
    img[y1:y2, x1:x2] = data

# # Task 5

def stitch_images(points_left, points_right, imgs = None, filenames = None):
    left, right = inputs_parse(imgs, filenames)

    H = find_homography(points_right, points_left)
    span_left = np.vstack([np.zeros(2), list(left.shape)[:2][::-1]]).T
    span_right, warped_right = apply_homography(H, img = right)
    span_left, span_right = span_left.astype(int), span_right.astype(int)
    span = merge_spans(span_left, span_right).astype(int)
    new_size = tuple((span[:, 1] - span[:, 0])[::-1]) + tuple([3])

    new_left = np.zeros(new_size).astype(np.uint8)
    new_right = new_left.copy()
    
    put_on_image(new_left, left, subtract_spans(span_left, span, list(new_size[:2])))
    put_on_image(new_right, warped_right, subtract_spans(span_right, span, list(new_size[:2])))

    margin_left, margin_right = span_right[0][0], span_left[0][1]
    xmin = find_optimal_seam(new_left, new_right, margin_left, margin_right)

    h, w, _ = new_size

    for y in range(h):
        new_left[y, np.arange(xmin[y], w), :] = new_right[y, np.arange(xmin[y], w), :]

    # visualize the stitch
    # for y in range(h):
    #     new_left[y, xmin[y], :] = RED
    cv2_imshow(new_left)

    return new_left

def task5_snippet():
    points_left, points_right = task4_snippet()

    img = stitch_images(
        points_left, 
        points_right, 
        filenames=['manual_stitching/img2.png', 'manual_stitching/img1.png']
    )

    cv2.imwrite('task_solutions/task5_solution.png', img)

# # Task 6
import subprocess

def unpack_npz(npz, best_points = 15):
    k0, k1, m, mc = (
        npz['keypoints0'], 
        npz['keypoints1'], 
        npz['matches'],
        npz['match_confidence']
    )

    order = np.argsort(mc)[::-1]
    k0, m, mc = k0[order], m[order], mc[order]
    if (best_points < len(m)):
        k0, m, mc = k0[:best_points], m[:best_points], mc[:best_points]
    
    print(f'Found {len(m)} matches above {mc[-1]} confidence')

    return k0, k1, m, mc

def get_sg_point_pairs(names):
    names = [n[:-4] for n in names]
    path = f'superglue_result/{'_'.join(names)}_matches.npz'
    print(path)
    npz = np.load(path)
    kleft, kright, matches, conf = unpack_npz(npz)
    right_matches = kright[matches]

    return kleft, right_matches

def stitch2_superglue(imgs = None, filenames = None, script_debug = False):
    left, right = inputs_parse(imgs, filenames)

    folder = 'superglue_input/'
    names = ['left.png', 'right.png']

    for name, file in list(zip(names, [left, right])):
        cv2.imwrite(f'{folder}{name}', file)

    with open(f'{folder}/pairs', "w") as pairs:
        pairs.write(' '.join(names))

    result = subprocess.run(['./script.sh'], capture_output=True, text=True)

    if script_debug:
        print("Output from script:")
        print(result.stdout)
        if result.stderr:
            print("Errors from script:")
            print(result.stderr)

    points_right, points_left = get_sg_point_pairs(names)
    return stitch_images(points_right, points_left, imgs=[left, right])

def task6_snippet():
    img = stitch2_superglue(filenames=['data3/img3.png', 'data3/img2.png'])
    cv2.imwrite('task_solutions/task6_solution.png', img)

# # Task 7

def stitch5_from_middle(imgs = None, filenames = None, script_debug = False):
    imgs = np.array(inputs_parse(imgs, filenames))
    
    first, *rest = imgs[[2, 3, 1, 4, 0]]
    inputs = list(zip(rest, ['R', 'L', 'R', 'L']))

    current = first
    for file, side in inputs:
        argfiles = [current, file]
        argfiles = argfiles if side == 'R' else argfiles[::-1]

        current = stitch2_superglue(imgs = argfiles)

def stitch5_from_sides(imgs = None, filenames = None, script_debug = False):
    imgs = inputs_parse(imgs, filenames)
    
    pairs = [
        [0, 1],
        [3, 4],
        [5, 2],
        [-1, -2],
    ]

    for i1, i2 in pairs:
        argfiles = [imgs[i1], imgs[i2]]
        current = stitch2_superglue(imgs = argfiles)

        imgs += [current]

    return imgs[-1]

def task7_snippet():
    filenames = [f'data3/img{i}.png' for i in range(5,10)][::-1]
    print(filenames)
    img = stitch5_from_sides(filenames=filenames)
    cv2.imwrite('task_solutions/task7_solution.png', img)

task1_snippet()
task2_snippet()
task3_snippet()
task4_snippet()
task5_snippet()
task6_snippet()
task7_snippet()




