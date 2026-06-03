import cv2
import numpy as np
import time

img = np.random.randint(0, 255, (2500, 1500), dtype=np.uint8)

print("OpenCL available:", cv2.ocl.haveOpenCL())

# CPU
print("Starting CPU run...")
start = time.time()
cpu_res = cv2.fastNlMeansDenoising(img, None, h=10.0, templateWindowSize=7, searchWindowSize=21)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.2f}s")


# GPU
print("Starting GPU (UMat) run...")
cv2.ocl.setUseOpenCL(True)
img_umat = cv2.UMat(img)
start = time.time()
gpu_res_umat = cv2.fastNlMeansDenoising(img_umat, None, h=10.0, templateWindowSize=7, searchWindowSize=21)
gpu_res = gpu_res_umat.get()
gpu_time = time.time() - start
print(f"GPU time: {gpu_time:.2f}s")

print(f"Speedup: {cpu_time/gpu_time:.2f}x")
