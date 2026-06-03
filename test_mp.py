import cv2
import numpy as np
import time
import concurrent.futures

def process_img(img):
    return cv2.fastNlMeansDenoising(img, None, h=10.0, templateWindowSize=7, searchWindowSize=21)

if __name__ == "__main__":
    imgs = [np.random.randint(0, 255, (1600, 1200), dtype=np.uint8) for _ in range(3)]
    
    # Sequential
    start = time.time()
    for img in imgs:
        process_img(img)
    seq_time = time.time() - start
    print(f"Sequential time: {seq_time:.2f}s")
    
    # MP
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_img, imgs))
    mp_time = time.time() - start
    print(f"Multiprocessing time: {mp_time:.2f}s")
    
    cv2.setNumThreads(1)
    # MP with cv2 threading disabled
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_img, imgs))
    mp_time_1t = time.time() - start
    print(f"MP (cv2 single thread) time: {mp_time_1t:.2f}s")
