import cv2
import numpy as np

def deskewDocumentImage(documentImage: np.ndarray) -> np.ndarray:
    if documentImage.ndim == 3:
        grayscaleImage = cv2.cvtColor(documentImage, cv2.COLOR_BGR2GRAY)
    else:
        grayscaleImage = documentImage.copy()

    edges = cv2.Canny(grayscaleImage, 0, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if lines is None:
        return documentImage

    angles = []
    for line in lines:
        _, theta = line[0]
        angle = (theta - np.pi / 2) * 180 / np.pi
        angles.append(angle)

    medianAngle = np.median(angles)

    if abs(medianAngle) < 1.0:
        return documentImage  # <-- this is real disabling

    h, w = grayscaleImage.shape
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, medianAngle, 1.0)

    return cv2.warpAffine(
        documentImage,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=200
    )
