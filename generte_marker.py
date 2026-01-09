import cv2 as cv
import numpy as np

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
marker_id = 4
marker_size = 400 # pixel

marker_img = cv.aruco.generateImageMarker(
    aruco_dict,
    marker_id,
    marker_size
)

cv.imwrite(f"aruco_marker_{marker_id}.png", marker_img)

cv.imshow("ArUco Marker", marker_img)
cv.waitKey(0)
cv.destroyAllWindows()
