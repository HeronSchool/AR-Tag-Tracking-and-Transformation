import cv2 as cv
from cv2 import aruco
import numpy as np
import math
import time
import datetime
import csv

print(cv.__version__)

class ArucoFilter(object):
    def __init__(self, markerType=aruco.DICT_4X4_50, markerSize=5,
                 calibPath = "C://Users//user//Desktop//Python_work//aruco//calibration_data//MultiMatrix.npz"):
        self._markerDict = aruco.getPredefinedDictionary(markerType)
        self._parameters = aruco.DetectorParameters()
        self._detector = cv.aruco.ArucoDetector(self._markerDict, self._parameters)

        self._markerSize = markerSize # marker physical size (cm)
        self._calibData = np.load(calibPath)
        self._camMatrix = self._calibData["camMatrix"]
        self._distCoef = self._calibData["distCoef"]
        self._rVectors = self._calibData["rVector"]
        self._tVectors = self._calibData["tVector"]
        
        self._marker_0 = [] # id 0
        self._marker_1 = [] # id 1
        self._marker_2 = [] # id 2
        self._markerData = [] # time, reference - x, y, z, offset - x, y, z
        self._starTime = time.time()
        self._frameCount = 0

    def drawMarkers(self, frame, showLabel=True):
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = self._detector.detectMarkers(grayFrame)
        
        # estimate pose (rotation & translation vectors)
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, self._markerSize, self._camMatrix, self._distCoef)
        
        if marker_IDs is not None:
            # clear arrays
            self._marker_0 = []
            self._marker_1 = []
            self._marker_2 = []
            
            # markers in one frame
            marker_index = range(0, marker_IDs.size)
            for id, corner, i in zip(marker_IDs, marker_corners, marker_index):
                
                # draw detected marker border
                cv.polylines(frame, [corner.astype(np.int32)], True, (255, 255, 255), 4, cv.LINE_AA)
                
                # draw 3D coordinate axes on the marker
                cv.drawFrameAxes(frame, self._camMatrix, self._distCoef, rVec[i], tVec[i], 4, 4)
                
                # (1, 4, 2) -> (4, 2)
                corner = corner.reshape(4, 2).astype(np.int32)
                
                # display marker ID
                if showLabel:
                    cv.putText(frame, f"id: {id}", corner[0], cv.FONT_HERSHEY_PLAIN, 1.3, (200, 200, 200), 2, cv.LINE_AA)
    
                # show base point    
                if id == 0:
                    self._marker_0.append(round(tVec[i][0][0], 6))
                    self._marker_0.append(round(tVec[i][0][1], 6))
                    self._marker_0.append(round(tVec[i][0][2], 6))
                elif id == 1:
                    self._marker_1.append(round(tVec[i][0][0], 6))
                    self._marker_1.append(round(tVec[i][0][1], 6))
                    self._marker_1.append(round(tVec[i][0][2], 6))
                elif id == 2:
                    self._marker_2.append(round(tVec[i][0][0], 6))
                    self._marker_2.append(round(tVec[i][0][1], 6))
                    self._marker_2.append(round(tVec[i][0][2], 6))
                    
            # after one frame
            timeElapsed = time.time() - self._starTime
            if self._marker_0 and self._marker_1 and self._marker_2:
                coordinates = np.hstack((timeElapsed, self._marker_0, self._marker_1, self._marker_2))
                
                print("coordinates:", coordinates)
                self._markerData.append(coordinates)
                self._frameCount += 1
            
    def saveCoordinateData(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        with open(f"./data/data_{now}.csv", "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            attr_name = ["Time(s)"]
            for i in range(0, 3):
                attr_name.append(f"marker_{i} : X (cm)")
                attr_name.append(f"marker_{i} : Y (cm)")
                attr_name.append(f"marker_{i} : Z (cm)")
            csv_writer.writerow(attr_name)
            
            for frame in range(self._frameCount):
                csv_writer.writerow(self._markerData[frame])
            csv_file.close()

if __name__ == "__main__":
    ar = ArucoFilter()
    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        ar.drawMarkers(frame)
        cv.imshow("frame", frame)

        key = cv.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
    # ar.saveCoordinateData()