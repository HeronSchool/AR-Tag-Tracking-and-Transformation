#%%
import cv2 as cv
import os
import numpy as np

# chess board size
CHESS_BOARD_DIM = (8, 5)

# the size of square in the chess board
SQUARE_SIZE = 13 # millimeters

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

calib_data_path = "C://Users//user//Desktop//Python_work//aruco//calibration_data"
CHECK_DIR = os.path.isdir(calib_data_path)

if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f"{calib_data_path} Directory is created")
else:
    print(f"{calib_data_path} Directory already exists")
    
#%% prepare object points
# x (left -> right) -> 0 14 28 42 56 70 84 98 112
# y (up -> down)-> 0 14 28 42 56 70
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)
obj_3D[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
obj_3D *= SQUARE_SIZE

# arrays to store object points and image points from all the images
obj_points_3D = [] # 3d points in real world space
img_points_2D = [] # 2d points in image plane

#%%
image_path = "C://Users//user//Desktop//Python_work//aruco//image//image0.png"
image = cv.imread(image_path)
grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(grayImage, CHESS_BOARD_DIM)

if ret:
    # 정밀 조정
    corners2 = cv.cornerSubPix(grayImage, 
                              corners, # 초기 코너 위치
                              (3, 3), # 검색 윈도우 크기 (width, height)
                              (-1, -1), # 검색 중 중심 영역을 무시할 영역 (None)
                              criteria)
    
    image = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners, ret)
    cv.imshow("corners", image)   
    cv.waitKey(0)
    cv.destroyAllWindows()
    
#%% read all images
image_dir_path = "C://Users//user//Desktop//Python_work//aruco//image"
files = os.listdir(image_dir_path)
for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)
    image = cv.imread(imagePath)
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(grayImage, CHESS_BOARD_DIM)
    
    if ret:
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)
        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

print(len(obj_points_3D))
print(len(img_points_2D))

# %% calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points_3D, img_points_2D, grayImage.shape[::-1], None, None)
print("--------------------------------------------------------")
print("calibrated - dumping the data into one files using numpy")
np.savez(
    f"{calib_data_path}/MultiMatrix",
    camMatrix = mtx,
    distCoef = dist,
    rVector = rvecs,
    tVector = tvecs
)

# loading
print("loading data stored using numpy savez function")
data = np.load(f"{calib_data_path}/MultiMatrix.npz")
camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]
print("calibration data loaded successfully")
# %%
