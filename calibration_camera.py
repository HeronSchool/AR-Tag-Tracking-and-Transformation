#%%
import cv2 as cv
import os

CHESS_BOARD_DIM = (8, 5)
n = 0 # image counter (more than 40)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria

#%% check if imager dir exists or not
image_dir_path = "C://Users//user//Desktop//Python_work//aruco//image"
CHECK_DIR = os.path.isdir(image_dir_path)

if not CHECK_DIR:
    os.makedirs(image_dir_path)
    print(f"{image_dir_path} Directory is created.")
else:
    print(f"{image_dir_path} Directory already exists.")

#%% camera
cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    copyFrame = frame.copy()
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(grayFrame, CHESS_BOARD_DIM)
    
    if ret:
        frame = cv.drawChessboardCorners(frame, CHESS_BOARD_DIM, corners, ret)
    
    cv.putText(frame, f"saved image: {n}", (30, 40), cv.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 0), 2, cv.LINE_AA)
    cv.imshow("frame", frame)
    
    key = cv.waitKey(1)
    
    if key == ord("q"):
        break
    elif key == ord("s") and ret:
        # store the checker board image
        cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)
        print(f"saved image number {n}")
        n += 1

cap.release()
cv.destroyAllWindows()

# %%
