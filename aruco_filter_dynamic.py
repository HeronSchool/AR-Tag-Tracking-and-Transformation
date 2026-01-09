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
        
        self._marker_reference = []
        self._marker_offset = []
        self._markerData = [] # time, reference - x, y, z, offset - x, y, z
        self._starTime = time.time()
        self._frameCount = 0
    
    def rotationMatrixToEulerAngles(R):
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

        singular = sy < 1e-6

        if not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z]) * (180/np.pi) # move from radians to degrees
    
    def drawMarkers(self, frame, showLabel=True, reference_point=1, reference_corner=[3, 2.5, 0]):
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = self._detector.detectMarkers(grayFrame)
        
        # estimate pose (rotation & translation vectors)
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, self._markerSize, self._camMatrix, self._distCoef)
        
        if marker_IDs is not None:
            # clear arrays
            self._marker_reference = []
            self._marker_offset = []
            
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
    
                # show reference point    
                if id == reference_point:
                    cv.putText(frame, f"Pos X = {round(tVec[i][0][0], 2)}", (0, 100), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                    cv.putText(frame, f"Pos Y = {round(tVec[i][0][1], 2)}", (0, 150), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                    cv.putText(frame, f"Pos Z = {round(tVec[i][0][2], 2)}", (0, 200), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)

                    rMat, _ = cv.Rodrigues(rVec[i][0])
                    rotation_array = ArucoFilter.rotationMatrixToEulerAngles(rMat)
                    
                    cv.putText(frame, f"Rot X = {round(rotation_array[0], 2)}", (0, 250), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                    cv.putText(frame, f"Rot Y = {round(rotation_array[1], 2)}", (0, 300), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                    cv.putText(frame, f"Rot Z = {round(rotation_array[2], 2)}", (0, 350), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)

                    self._marker_reference.append(round(tVec[i][0][0], 6))
                    self._marker_reference.append(round(tVec[i][0][1], 6))
                    self._marker_reference.append(round(tVec[i][0][2], 6))
                    
                if reference_corner and id == reference_point:
                    rMat, _ = cv.Rodrigues(rVec[i][0])
                    
                    # x, y, z distance from the reference point
                    local_offset = np.array(reference_corner) # x, y, z
                    
                    # transform from local to global coordinate
                    global_offset = rMat.dot(local_offset) + tVec[i][0]
                    
                    cv.putText(frame, f"Pos X = {round(global_offset[0], 2)}", (200, 100), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                    cv.putText(frame, f"Pos Y = {round(global_offset[1], 2)}", (200, 150), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                    cv.putText(frame, f"Pos Z = {round(global_offset[2], 2)}", (200, 200), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)

                    self._marker_offset.append(round(global_offset[0], 6))
                    self._marker_offset.append(round(global_offset[1], 6))
                    self._marker_offset.append(round(global_offset[2], 6))
                
                    
            # after one frame
            timeElapsed = time.time() - self._starTime
            if self._marker_reference and self._marker_offset:
                coordinates = np.hstack((timeElapsed, self._marker_reference, self._marker_offset))
                
                print("coordinates:", coordinates)
                self._markerData.append(coordinates)
                self._frameCount += 1
            
    def saveCoordinateData(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        with open(f"./data/data_{now}.csv", "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            attr_name = ["Time(s)"]
            attr_name.append(f"marker_reference : X (cm)")
            attr_name.append(f"marker_reference : Y (cm)")
            attr_name.append(f"marker_reference : Z (cm)")
            attr_name.append(f"marker_offset : X (cm)")
            attr_name.append(f"marker_offset : Y (cm)")
            attr_name.append(f"marker_offset : Z (cm)")
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