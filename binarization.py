import cv2
import numpy as np

def binarizeDocumentImage(
    grayscaleImage: np.ndarray,
    windowSize: int = None,
    k: float = 0.25
) -> np.ndarray:
    """
    Convert a grayscale document image into a binary image using
    Sauvola adaptive thresholding, with dynamic window sizing
    and slight smoothing.

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
        Slightly smoothed binary image.
    """

    if grayscaleImage.ndim != 2:
        raise ValueError("binarizeDocumentImage expects a grayscale image")

    if windowSize is None:
        h, w = grayscaleImage.shape
        # ~2.5% of max dimension is a good default for Sauvola window
        windowSize = max(11, int(max(h, w) * 0.025))

    if windowSize % 2 == 0:
        windowSize += 1

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

    # Eradicate salt-and-pepper background noise with a fast median filter
    # This perfectly cleans stray scanning dust while preserving 100% sharp text edges
    cleanBinary = cv2.medianBlur(binaryImage, 3)

    return cleanBinary
