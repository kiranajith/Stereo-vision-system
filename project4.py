
import cv2
import numpy as np
from numpy.linalg import svd
import datetime
import uuid


def normalize_points(points):
    if points.shape[1] == 2:
        points = np.hstack((points, np.ones((points.shape[0], 1))))

    mean = np.mean(points, axis=0)
    rms = np.sqrt(np.mean(np.sum((points - mean) ** 2, axis=1)))
    scale = np.sqrt(2) / rms
    T = np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]])
    normalized_points = (T @ points.T).T
    return normalized_points, T


def eight_point_algorithm(pts1, pts2):
    pts1_normalized, T1 = normalize_points(pts1)
    pts2_normalized, T2 = normalize_points(pts2)

    A = np.zeros((len(pts1_normalized), 9))
    for i in range(len(pts1_normalized)):
        A[i, :] = np.kron(pts1_normalized[i], pts2_normalized[i])

    _, _, V = svd(A)
    F_normalized = V[-1].reshape(3, 3)

    U, S, Vt = svd(F_normalized)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt

    F = T2.T @ F_rank2 @ T1
    return F / F[2, 2]

def ransac_fundamental_matrix(pts1, pts2, threshold=1, max_iterations=1000):
    best_inliers = 0
    best_F = None

    for _ in range(max_iterations):
        sample_indices = np.random.choice(len(pts1), 8, replace=False)
        sample_pts1 = pts1[sample_indices]
        sample_pts2 = pts2[sample_indices]

        F = eight_point_algorithm(sample_pts1, sample_pts2)

        pts1_h = np.hstack((pts1, np.ones((len(pts1), 1))))
        pts2_h = np.hstack((pts2, np.ones((len(pts2), 1))))
        errors = np.abs(np.sum((pts2_h @ F) * pts1_h, axis=1))
        inliers = np.sum(errors < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_F = F

    return best_F
def visualize_epipolar_lines(img1, img2, F, pts1, pts2, kp1, kp2):
    img_size = img1.shape[::-1][:2]  

    ret, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1),np.float32(pts2), F, img_size)

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img1_epilines = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    img2_epilines = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

    for (r1, pt1), (r2, pt2) in zip(zip(lines1, pts1), zip(lines2, pts2)):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -r1[2] / r1[1]])
        x1, y1 = map(int, [gray1.shape[1], -(r1[2] + r1[0] * gray1.shape[1]) / r1[1]])
        img1_epilines = cv2.line(img1_epilines, (x0, y0), (x1, y1), color, 1)
        img1_epilines = cv2.circle(img1_epilines, tuple(map(int, pt1)), 5, color, -1)

        x0, y0 = map(int, [0, -r2[2] / r2[1]])
        x1, y1 = map(int, [gray2.shape[1], -(r2[2] + r2[0] * gray2.shape[1]) / r2[1]])
        img2_epilines = cv2.line(img2_epilines, (x0, y0), (x1, y1), color, 1)
        img2_epilines = cv2.circle(img2_epilines, tuple(map(int, pt2)), 5, color, -1)

    cv2.imwrite('combined.jpg',cv2.hconcat([img1_epilines, img2_epilines]))
    cv2.imwrite('image1_epilines.jpg', img1_epilines)
    cv2.imwrite('image2_epilines.jpg', img2_epilines)

    print('\nHomography Matrices:')
    print("--------------------")
    print("H1:")
    print(H1)
    print("H2:")
    print(H2)

def estimate_essential_matrix(F, cam0, cam1):
    E = np.dot(np.dot(cam1.T, F), cam0)
    U, S, Vt = np.linalg.svd(E)
    S[0] = 1
    S[1] = 1
    S[2] = 0
    E = np.dot(np.dot(U, np.diag(S)), Vt)
    return E

def decompose_essential_matrix(E):
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = np.dot(np.dot(U, W), Vt)
    R2 = np.dot(np.dot(U, W.T), Vt)
    t1 = U[:, 2]
    t2 = -U[:, 2]
    if np.linalg.det(R1) < 0:
        R1 *= -1
    if np.linalg.det(R2) < 0:
        R2 *= -1
    print('-- Camera 1 --')
    print("\nRotation Matrix:\n",R1)
    print('\nTranslational Vector:\n',t1)
    print('\n-- Camera 2 --')
    print("\nRotation Matrix:\n",R2)
    print('\nTranslational Vector:\n',t2)
    return R1, t1, R2, t2

def read_calibration_parameters(calib_file):
    with open(calib_file, 'r') as f:
        params = {}
        lines = f.readlines()
        for line in lines:
            key, value = line.strip().split("=")
            if key.startswith("cam"):
                values = [float(v.strip(";")) for v in value.strip("[]\n").split()]
                value = np.array(values).reshape((3, 3))
            else:
                value = float(value)
            params[key] = value
    return params

def save_disparity_map(disparity_map, grayscale_filename, heatmap_filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4().hex
    grayscale_filename = f"{timestamp}_{unique_id}_{grayscale_filename}"
    heatmap_filename = f"{timestamp}_{unique_id}_{heatmap_filename}"
    normalized_disparity = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite(grayscale_filename, normalized_disparity)

    heatmap = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_filename, heatmap)
def compute_depth_map(disparity_map, focal_length, baseline):
    disparity_map[disparity_map == 0] = 0.01
    depth_map = (focal_length * baseline) / disparity_map
    return depth_map
def save_depth_map(depth_map, grayscale_filename, heatmap_filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4().hex
    grayscale_filename = f"{timestamp}_{unique_id}_{grayscale_filename}"
    heatmap_filename = f"{timestamp}_{unique_id}_{heatmap_filename}"

    normalized_depth = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(grayscale_filename, normalized_depth)
    heatmap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_filename, heatmap)

def compute_disparity_SSD(imgL, imgR, window_size=5, max_disp=64):
    h, w = imgL.shape[:2]

    hw = window_size // 2

    disparity = np.zeros((h, w), dtype=np.float32)

    imgL_padded = np.pad(imgL, ((hw, hw), (hw, hw)), mode='constant')
    imgR_padded = np.pad(imgR, ((hw, hw), (hw, hw)), mode='constant')

    for y in range(hw, h + hw):
        for x in range(hw, w + hw):
            patchL = imgL_padded[y-hw:y+hw+1, x-hw:x+hw+1]
            min_SSD = float('inf')
            best_disp = 0

            for d in range(max_disp):
                patchR = imgR_padded[y-hw:y+hw+1, x-hw-d:x+hw+1-d]
                SSD = np.sum((patchL - patchR) ** 2)
                if SSD < min_SSD:
                    min_SSD = SSD
                    best_disp = d

            disparity[y-hw, x-hw] = best_disp

    return disparity

image_sets = [
    {
        'name': 'chess',
        'img1_path': '/Users/kiranajith/Downloads/First link/chess/im0.png',
        'img2_path': '/Users/kiranajith/Downloads/First link/chess/im1.png',
        'calib_path': '/Users/kiranajith/Downloads/First link/chess/calib.txt',
    },
    {
        'name': 'ladder',
        'img1_path': '/Users/kiranajith/Downloads/First link/ladder/im0.png',
        'img2_path': '/Users/kiranajith/Downloads/First link/ladder/im1.png',
        'calib_path': '/Users/kiranajith/Downloads/First link/ladder/calib.txt',
    },
    {
        'name': 'artroom',
        'img1_path': '/Users/kiranajith/Downloads/First link/artroom/im0.png',
        'img2_path': '/Users/kiranajith/Downloads/First link/artroom/im1.png',
        'calib_path': '/Users/kiranajith/Downloads/First link/artroom/calib.txt',
    }
]
for image_set in image_sets:
    print(f"\nProcessing {image_set['name']} images")
    print('-----------------------------------')

    calibration_params = read_calibration_parameters(image_set['calib_path'])
    cam0 = calibration_params['cam0']
    cam1 = calibration_params['cam1']
    baseline = calibration_params['baseline']
    window_size = 5
    min_disp = 0
    max_disp = 255
    num_disp = max_disp - min_disp

    stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=32
    )

    img1 = cv2.imread(image_set['img1_path'])
    img2 = cv2.imread(image_set['img2_path'])
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.match(des1, des2)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F = ransac_fundamental_matrix(pts1, pts2)
    print('\nFundamental Matrix')
    print('--------------------')
    print(F)
    focal_length = cam0[0, 0]  
    disparity_map = stereo.compute(img1, img2).astype(np.float32) / 16.0

    depth_map = compute_depth_map(disparity_map, focal_length, baseline)

    save_depth_map(depth_map, 'depth_grayscale.png', 'depth_heatmap.png')
    save_disparity_map(disparity_map, 'disparity_grayscale.png', 'disparity_heatmap.png')


    E = estimate_essential_matrix(F, cam0, cam1)

    print('\nEssential Matrix')
    print('------------------')
    print(E)
    print('\nThe Essential matrix is decompposed into the following:')
    print('--------------------------------------------------------')
    decompose_essential_matrix(E)
    visualize_epipolar_lines(img1,img2,F,pts1,pts2,kp1,kp2)



