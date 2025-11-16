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
    raise NotImplementedError("TO DO in panorama.py")
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
    raise NotImplementedError("TO DO in panorama.py")
    # student_code end

    return result
