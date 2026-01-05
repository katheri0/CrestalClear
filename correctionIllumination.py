import cv2
import numpy as np

def normalizeDocumentIllumination(
    documentImage: np.ndarray,
    kernelSize: int = 25
) -> np.ndarray:
    """
    Normalize uneven illumination using background division,
    which preserves stroke contrast better than subtraction.
    """

    if documentImage.ndim == 3:
        grayscaleImage = cv2.cvtColor(documentImage, cv2.COLOR_BGR2GRAY)
    else:
        grayscaleImage = documentImage.copy()

    grayscaleFloat = grayscaleImage.astype(np.float32)

    structuringElement = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernelSize, kernelSize)
    )

    backgroundEstimate = cv2.morphologyEx(
        grayscaleFloat, cv2.MORPH_CLOSE, structuringElement
    )

    # Avoid division by zero
    normalizedImage = grayscaleFloat / (backgroundEstimate + 1.0)

    normalizedImage = cv2.normalize(
        normalizedImage, None, 0, 255, cv2.NORM_MINMAX
    )

    return normalizedImage.astype(np.uint8)
