# videoProcessor

This package is a work in progress at achieving real time localized frequency detection in high resolution, high framerate, camera feeds using python, numpy and openCV. OpenCV is used to handling the camera stream and image display, but the intention is to recreate all computer vision functions using optimized numpy array slicing. 

Thus far Laplacian Edge Detectors, Gaussian Image Pyramids, and Difference of Gaussian Image Pyramids have been successfully implemented in real time. Currently, the next step is to implement a SIFT style feature point detector/descriptor algorithm using multiresolution difference of Gaussian Pyramids to efficiently track feature points in multiple overlapping images.
