import cv2
import numpy as np

def deskewDocumentImage(documentImage: np.ndarray) -> np.ndarray:
    if documentImage.ndim == 3:
        grayscaleImage = cv2.cvtColor(documentImage, cv2.COLOR_BGR2GRAY)
    else:
        grayscaleImage = documentImage.copy()

    # 1. 90/180/270 Degree Coarse Rotation Correction
    # Threshold for projection profile
    _, bin_img = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate variances of projection profiles
    proj_h = np.sum(bin_img, axis=1)
    proj_v = np.sum(bin_img, axis=0)
    
    var_h = np.var(proj_h)
    var_v = np.var(proj_v)
    
    # If vertical variance is higher, it means text lines are vertical (rotated 90 or 270)
    if var_v > var_h:
        # Rotate 90 degrees counter-clockwise (assume 90 for now, coarse correction)
        documentImage = cv2.rotate(documentImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
        grayscaleImage = cv2.rotate(grayscaleImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
        _, bin_img = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj_h = np.sum(bin_img, axis=1) # Recalculate horizontal profile

    # Basic heuristic for 180 degree rotation:
    # Text usually has more pixels in the lower half of the text line band due to baseline.
    # A simple way without OCR: check pixel density distribution relative to mean line center
    total_upper = 0
    total_lower = 0
    lines = np.where(proj_h > np.mean(proj_h))[0]
    # Simple split testing
    if len(lines) > 0:
        half_idx = len(lines) // 2
        total_upper = np.sum(proj_h[lines[:half_idx]])
        total_lower = np.sum(proj_h[lines[half_idx:]])
        # This is a very coarse heuristic. For perfect 180 detection, OCR is needed.
        # But we apply it if there's a huge disparity indicating upside down.
        if total_upper > total_lower * 1.5:  
            documentImage = cv2.rotate(documentImage, cv2.ROTATE_180)
            grayscaleImage = cv2.rotate(grayscaleImage, cv2.ROTATE_180)
            _, bin_img = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Fine Deskew via minAreaRect on contours
    # Dilate to connect characters into solid text blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for cnt in contours:
        x, y, w, h_box = cv2.boundingRect(cnt)
        if w > 50 and h_box > 10:  # Filter for reasonable text block sizes
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            
            # minAreaRect angle is always between [0, 90). Adjust it to actual skew
            if angle > 45:
                angle = angle - 90
            
            # Only consider small skews for fine correction
            if -15.0 <= angle <= 15.0:
                angles.append(angle)

    if not angles:
        return documentImage

    medianAngle = np.median(angles)
    
    if abs(medianAngle) < 0.5:
        return documentImage  # No fine deskew needed
        
    h, w = documentImage.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, medianAngle, 1.0)
    
    if documentImage.ndim == 3:
        borderValue = (255, 255, 255)
    else:
        borderValue = 255
        
    return cv2.warpAffine(
        documentImage,
        matrix,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderValue=borderValue
    )
