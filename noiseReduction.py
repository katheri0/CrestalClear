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
        
    img_h, img_w = grayscaleImage.shape
    max_dim = max(img_h, img_w)
    
    # Scale windows smoothly with image size. Base sizes (7, 21) are good for ~1000px images.
    template_win = max(5, int(max_dim * 0.007))
    if template_win % 2 == 0:
        template_win += 1
        
    search_win = max(15, int(max_dim * 0.021))
    if search_win % 2 == 0:
        search_win += 1

    # Non-local means preserves text edges better than median blur
    denoisedImage = cv2.fastNlMeansDenoising(
        grayscaleImage, None, h=10.0, 
        templateWindowSize=template_win, 
        searchWindowSize=search_win
    )

    return denoisedImage
