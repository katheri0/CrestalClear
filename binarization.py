import cv2
import numpy as np

def binarizeDocumentImage(
    grayscaleImage: np.ndarray,
    windowSize: int = 25,
    k: float = 0.2
) -> np.ndarray:
    """
    Convert a grayscale document image into a binary image using
    Sauvola adaptive thresholding.

    Parameters
    ----------
    grayscaleImage : np.ndarray
        Input image in grayscale.
    windowSize : int
        Size of the local neighborhood window (must be odd).
    k : float
        Sauvola sensitivity parameter.

    Returns
    -------
    np.ndarray
        Binary image with text as white (255) and background as black (0).
    """

    if grayscaleImage.ndim != 2:
        raise ValueError("binarizeDocumentImage expects a grayscale image")

    # Normalize to float for numerical stability
    normalizedImage = grayscaleImage.astype(np.float32)

    # Compute local mean and standard deviation
    localMean = cv2.boxFilter(
        normalizedImage, ddepth=-1,
        ksize=(windowSize, windowSize)
    )

    localMeanSq = cv2.boxFilter(
        normalizedImage ** 2, ddepth=-1,
        ksize=(windowSize, windowSize)
    )

    variance = localMeanSq - localMean ** 2
    variance = np.maximum(variance, 0)
    localStdDev = np.sqrt(variance)

    # Sauvola threshold formula
    dynamicThreshold = localMean * (
        1 + k * ((localStdDev / 128) - 1)
    )

    binaryImage = np.where(
        normalizedImage > dynamicThreshold, 255, 0
    ).astype(np.uint8)

    return binaryImage
