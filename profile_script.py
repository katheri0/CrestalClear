import cProfile
import pstats
import cv2
import numpy as np
import time

from main import processSingleDocumentImage

# Create dummy document image (1600x1200) to simulate a standard document
img = np.ones((1600, 1200, 3), dtype=np.uint8) * 255

# Add some fake text lines
for i in range(100):
    cv2.line(img, (20, 20 + i*15), (1180, 20 + i*15), (0, 0, 0), 2)

# Add scanning noise
noise = np.random.randint(0, 50, (1600, 1200, 3), dtype=np.uint8)
img = np.clip(img.astype(np.int16) - noise, 0, 255).astype(np.uint8)

print("Checking OpenCL availability:", cv2.ocl.haveOpenCL())
print("Is OpenCL enabled?", cv2.ocl.useOpenCL())

print("Starting profiling...", flush=True)

start = time.time()
profiler = cProfile.Profile()
profiler.enable()
processSingleDocumentImage(img)
profiler.disable()
end = time.time()

print(f"Total time: {end-start:.2f}s", flush=True)

with open('profile.txt', 'w') as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats('cumtime')
    stats.print_stats(20)
