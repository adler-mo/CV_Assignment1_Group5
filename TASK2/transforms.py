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
    raise NotImplementedError("TO DO in transforms.py")
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
    raise NotImplementedError("TO DO in transforms.py")
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
    raise NotImplementedError("TO DO in transforms.py")
    # student_code end

    return T, width, height
