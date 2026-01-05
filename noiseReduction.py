import cv2
import numpy as np

def reduceDocumentNoise(documentImage: np.ndarray) -> np.ndarray:
    """
    Suppress scanning and sensor noise while preserving text edges.
    Uses median filtering suitable for document images.
    """

    if documentImage.ndim == 3:
        grayscaleImage = cv2.cvtColor(documentImage, cv2.COLOR_BGR2GRAY)
    else:
        grayscaleImage = documentImage.copy()

    denoisedImage = cv2.medianBlur(grayscaleImage, ksize=3)

    return denoisedImage
