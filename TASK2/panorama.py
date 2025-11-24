#####TUWIEN - CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_simple(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Stitch the final panorama with the calculated panorama extents
    by transforming every image to the same coordinate system as the center image. Use the dot product
    of the translation matrix 'T' and the homography per image 'H' as transformation matrix.
    HINT: cv2.warpPerspective(..), cv2.addWeighted(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) panorama image ([height x width x 3])
    """
    
    # student_code start
    result = np.zeros((height, width, 3), dtype=np.uint8)
        
    for i in range(len(images)):
        img = images[i]
        M = T @ H[i]
        
        img = cv2.warpPerspective(img, M, (width, height))
        
        result = cv2.addWeighted(result, 1, img, 1, 0)
        
    # student_code end
        
    return result


def get_blended(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Use the equation from the assignment description to overlay transformed
    images by blending the overlapping colors with the respective alpha values
    HINT: ndimage.distance_transform_edt(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) blended panorama image ([height x width x 3])
    """
    
    # student_code start

    pan = np.zeros((height, width, 3), dtype=np.float32)
    weights_total = np.zeros((height, width), dtype=np.float32)

    for i in range(len(images)):
        img = images[i]
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), dtype=np.float32)
        mask[0, :] = 0
        mask[h-1, :] = 0
        mask[:, 0] = 0
        mask[:, w-1] = 0

        weights = ndimage.distance_transform_edt(mask)
        if weights.max() > 0:
            weights = weights / weights.max()

        M = T @ H[i]
        img = cv2.warpPerspective(img, M, (width, height)).astype(np.float32)
        weights = cv2.warpPerspective(weights, M, (width, height))

        weights_colors = np.dstack([weights] * 3)

        weighted_img = img * weights_colors
        pan += weighted_img
        weights_total += weights

    weights_total_non_zero = weights_total # [weights_total==0] = 1.0
    result = pan / np.dstack([weights_total_non_zero] * 3)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # student_code end

    return result
