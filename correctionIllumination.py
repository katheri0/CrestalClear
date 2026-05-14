import cv2
import numpy as np

def normalizeDocumentIllumination(
    documentImage: np.ndarray,
    kernelSize: int = 75
) -> np.ndarray:
    """
    Normalize uneven illumination using CLAHE for local contrast 
    enhancement followed by background division with downscaling
    to efficiently support large kernel sizes.
    """

    if documentImage.ndim == 3:
        grayscaleImage = cv2.cvtColor(documentImage, cv2.COLOR_BGR2GRAY)
    else:
        grayscaleImage = documentImage.copy()

    # 1. Apply CLAHE to equalize lighting locally.
    # This prevents the darker side of the page from losing detail and washing out.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhancedImage = clahe.apply(grayscaleImage)

    grayscaleFloat = enhancedImage.astype(np.float32)

    # 2. Downscale for faster morphological operations and larger effective kernel
    h, w = grayscaleFloat.shape
    scale = 0.25
    smallImg = cv2.resize(grayscaleFloat, (int(w * scale), int(h * scale)))
    
    smallKernelSize = max(3, int(kernelSize * scale))
    structuringElement = cv2.getStructuringElement(
        cv2.MORPH_RECT, (smallKernelSize, smallKernelSize)
    )

    smallBackgroundEstimate = cv2.morphologyEx(
        smallImg, cv2.MORPH_CLOSE, structuringElement
    )

    # 3. Upscale the background estimate back to original size
    backgroundEstimate = cv2.resize(smallBackgroundEstimate, (w, h))
    
    # Smooth to remove blockiness from upscaling
    backgroundEstimate = cv2.GaussianBlur(backgroundEstimate, (15, 15), 0)

    # 4. Division to flatten the illumination
    normalizedImage = grayscaleFloat / (backgroundEstimate + 1e-5)

    normalizedImage = cv2.normalize(
        normalizedImage, None, 0, 255, cv2.NORM_MINMAX
    )

    return normalizedImage.astype(np.uint8)
