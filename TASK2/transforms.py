#####TUWIEN - CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List, Tuple
from numpy.linalg import inv
import numpy as np
import mapping
import random
import cv2


def get_geometric_transform(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Calculate a homography from the first set of points (p1) to the second (p2)

    Parameters
    ----------
    p1 : np.ndarray
        first set of points
    p2 : np.ndarray
        second set of points
    
    Returns
    ----------
    np.ndarray
        homography from p1 to p2
    """

    num_points = len(p1)
    A = np.zeros((2 * num_points, 9))
    for p in range(num_points):
        first = np.array([p1[p, 0], p1[p, 1], 1])
        A[2 * p] = np.concatenate(([0, 0, 0], -first, p2[p, 1] * first))
        A[2 * p + 1] = np.concatenate((first, [0, 0, 0], -p2[p, 0] * first))
    U, D, V = np.linalg.svd(A)
    H = V[8].reshape(3, 3)

    # homography from p1 to p2
    return (H / H[-1, -1]).astype(np.float32)


def get_transform(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[np.ndarray, List[int]]:
    """
    Estimate the homography between two set of keypoints by implementing the RANSAC algorithm
    HINT: random.sample(..), transforms.get_geometric_transform(..), cv2.perspectiveTransform(..)

    Parameters
    ----------
    kp1 : List[cv2.KeyPoint]
        keypoints left image ([number_of_keypoints] - KeyPoint)
    kp2 :  List[cv2.KeyPoint]
        keypoints right image ([number_of_keypoints] - KeyPoint)
    matches : List[cv2.DMatch]
        indices of matching keypoints ([number_of_matches] - DMatch)
    
    Returns
    ----------
    np.ndarray
        homographies from left (kp1) to right (kp2) image ([3 x 3] - float)
    List[int]
        inliers : list of indices, inliers in 'matches' ([number_of_inliers x 1] - int)
    """

    # student_code start
    N = 1000 
    T = 5.0

    trans = None
    inliers = []

    pts1_all = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)  # (M,2)
    pts2_all = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)  # (M,2)

    M = len(matches)
    if M < 4:
        return None, []

    for _ in range(N):
        idxs = random.sample(range(M), 4)

        p1 = pts1_all[idxs]
        p2 = pts2_all[idxs]

        H_candidate = get_geometric_transform(p1, p2)
        if H_candidate is None:
            continue

        pts1_h = pts1_all.reshape(-1, 1, 2)
        pts1_trans = cv2.perspectiveTransform(pts1_h, H_candidate).reshape(-1, 2)

        diff = pts2_all - pts1_trans
        dists = np.linalg.norm(diff, axis=1)

        inlier_idx = np.where(dists < T)[0]

        if len(inlier_idx) > len(inliers):
            inliers = inlier_idx
            trans = H_candidate

    if trans is None or len(inliers) < 4:
        return None, []
    else:
        p1_in = pts1_all[inliers]
        p2_in = pts2_all[inliers]
        trans = get_geometric_transform(p1_in, p2_in)

    # student_code end

    return trans, inliers


def to_center(desc: List[np.ndarray], kp: List[cv2.KeyPoint]) -> List[np.ndarray]:
    """
    Prepare all homographies by calculating the transforms from all other images
    to the reference image of the panorama (center image)
    First use mapping.calculate_matches(..) and get_transform(..) to get homographies between
    two consecutive images from left to right, then calculate and return the homographies to the center image
    HINT: inv(..), pay attention to the matrix multiplication order!!
    
    Parameters
    ----------
    desc : List[np.ndarray]
        list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    kp : List[cv2.KeyPoint]
        list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    
    Returns
    ----------
    List[np.ndarray]
        (H_center) list of homographies to the center image ( [number_of_images x 3 x 3] - float)
    """

    # student_code start
    num_images = len(desc)
    H_center = [np.zeros((3,3), dtype=np.float32) for _ in range(num_images)]

    H_cons = [np.zeros((3,3), dtype=np.float32) for _ in range(num_images)]
    for i in range(0, num_images - 1):
        matches = mapping.calculate_matches(desc[i], desc[i+1])
        H, _ = get_transform(kp[i], kp[i+1], matches)
        H_cons[i] = H

    H_center[num_images // 2] = np.eye(3, 3, dtype=np.float32)
    for i in range(num_images // 2 - 1, -1, -1):
        H_center[i] = H_cons[i] @ H_center[i + 1]

    for i in range(num_images // 2 + 1, num_images, 1):
        H_center[i] =  H_center[i - 1] @ inv(H_cons[i-1])
    # student_code end

    return H_center


def get_panorama_extents(images: List[np.ndarray], H: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    """
    Calculate the extent of the panorama by transforming the corners of every image
    and geht the minimum and maxima in x and y direction, as you read in the assignment description.
    Together with the panorama dimensions, return a translation matrix 'T' which transfers the
    panorama in a positive coordinate system. Remember that the origin of opencv images is in the upper left corner
    HINT: cv2.perspectiveTransform(..)

    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])

    Returns
    ---------
    np.ndarray
        T : transformation matrix to translate the panorama to positive coordinates ([3 x 3])
    int
        width of panorama (in pixel)
    int
        height of panorama (in pixel)
    """

    # student_code start
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for i in range(len(images)):
        img = images[i]
        h = img.shape[0]
        w = img.shape[1]

        corners = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float32)
        corners = corners.reshape((-1, 1, 2))

        corners = cv2.perspectiveTransform(corners, H[i])
        corners = corners.reshape((-1, 2))

        min_x = min(min_x, corners[:, 0].min())
        min_y = min(min_y, corners[:, 1].min())
        max_x = max(max_x, corners[:, 0].max())
        max_y = max(max_y, corners[:, 1].max())

    T = np.eye(3, 3, dtype=np.float32)
    T[0][2] = abs(min_x)
    T[1][2] = abs(min_y)

    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))

    # student_code end

    return T, width, height
