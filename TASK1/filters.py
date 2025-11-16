#####TUWIEN - CV: Task1 - Scale-Invariant Blob Detection
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import Tuple
import numpy as np
import cv2


def create_log_kernel(size: int, sig: float) -> float:
    """
    Returns a rotationally symmetric Laplacian of Gaussian kernel
    with given 'size' and standard deviation 'sig'

    Parameters
    ----------
    size : int
        size of kernel (must be odd) (int)
    sig : int
        standard deviation (float)
    
    Returns
    --------
    float
        kernel: filter kernel (size x size) (float)

    """

    kernel = np.zeros((size, size), np.float64)
    halfsize = int(np.floor(size / 2))
    r = range(-halfsize, halfsize + 1, 1)
    for x in r:
        for y in r:
            hg = (np.power(np.float64(x), 2) + np.power(np.float64(y), 2)) / (2 * np.power(np.float64(sig), 2))
            kernel[x + halfsize, y + halfsize] = -((1.0 - hg) * np.exp(-hg)) / (np.pi * np.power(sig, 4))

    return kernel - np.mean(kernel)


def get_log_pyramid(img: np.ndarray, sigma: float, k: float, levels: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a LoG scale space of given image 'img' with depth 'levels'
    The filter parameter 'sigma' increases by factor 'k' per level
    HINT: np.multiply(..), cv2.filter2D(..)

    Parameters
    ----------
    img : np.ndarray
        input image (n x m x 1) (float)
    sigma : float
        initial standard deviation for filter kernel
    levels : int
        number of layers of pyramid 

    Returns
    ---------
    np.ndarray
        scale_space : image pyramid (n x m x levels - float)
    np.ndarray
        all_sigmas : standard deviation used for every level (levels x 1 - float)
    """

    # student_code start
    if img.ndim == 3:
        # if it's (H, W, 1) or (H, W, 3), collapse channels
        if img.shape[2] == 1:
            img2d = img[:, :, 0]
        else:
            # average RGB/BGR channels to get grayscale
            img2d = img.mean(axis=2)
    else:
        img2d = img

    img2d = img2d.astype(np.float64)

    # Now this is guaranteed to be 2D, so this will NOT raise:
    h, w = img2d.shape

    scale_space = np.zeros((h, w, levels), dtype=np.float64)
    all_sigmas = np.zeros(levels, dtype=np.float64)

    sigma_i = float(sigma)

    for i in range(levels):
        all_sigmas[i] = sigma_i

        # kernel size: [-3σ, 3σ] → 2⌊3σ⌋ + 1
        size = int(2 * np.floor(3 * sigma_i) + 1)

        log_kernel = create_log_kernel(size, sigma_i)
        # scale-normalize by σ² (as required in the assignment)
        log_kernel = (sigma_i ** 2) * log_kernel

        response = cv2.filter2D(
            img2d,
            ddepth=-1,
            kernel=log_kernel,
            borderType=cv2.BORDER_REPLICATE
        )

        scale_space[:, :, i] = np.abs(response)
        sigma_i *= k
    # student_code end

    return scale_space, all_sigmas
