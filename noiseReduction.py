import cv2
import numpy as np


def reduceDocumentNoise(documentImage: np.ndarray) -> np.ndarray:
    """
    Sharpen text edges to prepare for binarization, heavily cutting compute time.
    Instead of heavily blurring background noise (which binarization destroys anyway),
    we use an Unsharp Mask to sharply contrast the text against the paper.
    """

    if documentImage.ndim == 3:
        grayscaleImage = cv2.cvtColor(documentImage, cv2.COLOR_BGR2GRAY)
    else:
        grayscaleImage = documentImage.copy()

    # Fast Unsharp Mask: (Original * 1.5) - (Blurred * 0.5)
    # This makes ink darker and background lighter instantly, skipping heavy math
    gaussianBlur = cv2.GaussianBlur(grayscaleImage, (5, 5), 0)
    sharpenedImage = cv2.addWeighted(grayscaleImage, 1.5, gaussianBlur, -0.5, 0)

    return sharpenedImage
