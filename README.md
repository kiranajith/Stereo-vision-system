# Stereo Vision and 3D Perception for Autonomous Robots

This project is part of the ENPM673 - Perception for Autonomous Robots coursework. It focuses on implementing stereo vision concepts using Python and OpenCV. It involves matching features in stereo images, estimating fundamental and essential matrices, rectifying the images, estimating disparity, and computing depth maps.

## Dependencies
- Python 3.7+
- NumPy
- OpenCV

## Instructions to Run the Code
Ensure you have the required dependencies installed in your environment. Python scripts and sample images are provided in the repository.

1. Clone this repository using the command: `git clone https://github.com/kiranajith/Stereo-vision-system-.git`
2. Navigate to the directory: `cd YourRepositoryName`
3. Run the Python script using the command: `python project4.py`

Please ensure you include the correct relative path of the images and calibration text file.

## Project Pipeline

### Calibration
- Feature matching in stereo images.
- Estimation of the Fundamental and Essential matrices.
- Decomposition of Essential matrix into translation T and rotation R.

### Rectification
- Application of perspective transformation to align the epipolar lines horizontally.
- Printing of the homography matrices H1 and H2 for both left and right images.

### Correspondence
- Implementation of the matching window concept on each epipolar line.
- Calculation of Disparity and saving the resulting images.

### Depth Map Computation
- Use of disparity information to compute the depth map.
- Saving the depth map as grayscale and colour images.

A detailed explanation of the steps involved is given in the report title 'kiran99_proj4.pdf'


## References
D. Scharstein, R. Szeliski and R. Zabih, "A taxonomy and evaluation of dense two-frame stereo correspondence algorithms," Proceedings IEEE Workshop on Stereo and Multi-Baseline Vision (SMBV 2001), Kauai, HI, USA, 2001, pp. 131-140, doi: 10.1109/SMBV.2001.988771.
