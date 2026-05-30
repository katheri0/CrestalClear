# the Idea

a program that process images to make them Crestal Clear and turn them to a pdf file .
best for pictures of notes and books (documents in general) 

```
CrestalClear/
│    ├──templates/
│    │      └──upload.html
│    │
│    ├──static/
│    │      └──assets/(images)
│    │
│    ├── binarization.py (enabled)
│    ├── correctionIllumination.py
│    ├── geometricNormalization.py
│    └── noiseReduction.py
```
Issue fixed: Now suitable for all images with dynamic parameter scaling, advanced geometric normalization, and smoothed binarization.