# AR-Tag-Tracking-and-Transformation

The project focuses on detecting ArUco markers and get the position and rotation data. There are 2 modes of detection. The first one is base, which is used for base coordinates by detecting 3 markers. The second one is dynamic, which is used for detecting a movement of a marker. All the data of markers is stored in csv files and will be used to calculate the transformation of one coordinate system to another.
It is followed by 2 steps:
1. Calibration : using chessboard (ex. 9x6)
2. Detection : using ArUco filters (dynamic, base)

# Calibration
It uses a chessboard for calibration. `cv.drawChessboardCorners` helps to detect the corders of the board. You need to take several images of different angles.
The calibration is followed by 2 steps.:
1. Saving images : calibration_campera.py
2. Calculation process : calibration.py

# Detection
It uses the calibration result and the marker information to detect the exact center location and rotation. In this program the default setting of marker type is `aruco.DICT_4X4_50`.

<img width="200" height="200" alt="aruco_marker_0" src="https://github.com/user-attachments/assets/9a6379bd-b10c-455e-9177-dabeb21b0a5c" />

There are 4*4 grid of squares including a padding of one square along the edge.

In the aruco_filter_dynamic.py file, it uses 3 markers that have a width * height size of 5cm * 5cm.
The functions for detection is as follows:
- `aruco.ArucoDetector`: it uses a marker type and other default parameters to be initialized.
- `ArucoDetector.detectMarkers` : it uses grayscale image to find the location of 4 corners for each marker and returns its ID as well.
- `aruco.estimatePoseSingleMarkers` : it uses the location of corners, marker size and other calibration parameters to detect the rotation and location of the center position for each marker.

Additional functions
- `rotationMatrixToEulerAngles` : it transforms the rotation matrix to euler angles for a marker.
- `reference corner` : it can detect an additional position of a marker which is used as a reference point. The offset distance for x, y, z coordinates will be used for this calculation.

In the main.py file, you can choose the type of ArucoFilter as dynamic or base.
1. aruco_filter_base.py : It detects 3 markers as base positions in the coordinate system of ArUco marker.
2. aruco_filter_dynamic.py : It detect 1 marker that move its position over time and detect the offset position of the reference marker.
The main class which is Cameo has two managers:
1. WindowManager : it manages the cv windows and controls the keyboard callback function.
2. CaptureManager : it manages the `cv.VideoCapture` object and uses WindowManager to show the frame that has been captured.
