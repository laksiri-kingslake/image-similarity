# Image Similarity
Check image similariry using two approaches.

## Approach 1: SSIM-based Approach
app.py uses SSIM-based similarity comparrision. </br>

1. Pixel-based structural analysis
Compares images by analyzing:
* Luminance patterns
* Contrast relationships
* Structural composition
* Local patterns in pixel neighborhoods

2. Works directly on image matrices
Uses raw pixel values (after grayscale conversion) without creating any feature vectors

3. Spatial relationship-aware
Conserves spatial information by comparing pixel regions in their original positions