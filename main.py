import cv2 as cv
from managers import WindowManager, CaptureManager
import datetime
from cv2 import aruco
import numpy as np
import aruco_filter_dynamic
import aruco_filter_base
import time

'''
installation
for estimatePoseSingleMarkers
-m pip install opencv-contrib-python
'''
print(cv.__version__)

class Cameo(object):
    def __init__(self, dynamic=True):
        self._windowManager = WindowManager('Aruco', self.onKeypress)
        self._captureManager = CaptureManager(cv.VideoCapture(0), self._windowManager, False)
        self._isDynamic = dynamic
        if dynamic:
            self._arucoFilter = aruco_filter_dynamic.ArucoFilter()
        else: 
            self._arucoFilter = aruco_filter_base.ArucoFilter()
    # key press callback
    def onKeypress(self, keycode):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if keycode == 32: # space
            self._captureManager.writeImage(f'./material/screenshot_{now}.png')
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()
    
    def run(self):
        self._windowManager.createWindow()
        
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
            if frame is not None:
                if self._isDynamic:
                    self._arucoFilter.drawMarkers(frame, showLabel=True, reference_point=1, reference_corner=[3, 0, 0])
                else:
                    self._arucoFilter.drawMarkers(frame, showLabel=True)
            self._captureManager.exitFrame()
            self._windowManager.processEvents()
        self._arucoFilter.saveCoordinateData()    

if __name__ == "__main__":
    
    cameo = Cameo(dynamic=False)
    cameo.run()
    
    

'''
    cap = cv.VideoCapture(0)
    cv.namedWindow("hello")
    window_created = True
    
    # detector
    markerType=aruco.DICT_4X4_50
    markerDict = aruco.getPredefinedDictionary(markerType)
    parameters = aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(markerDict, parameters)
    
    # estimator
    markerSize = 5
    calibPath = "C://Users//user//Desktop//Python_work//aruco//calibration_data//MultiMatrix.npz"
    calibData = np.load(calibPath)
    camMatrix = calibData["camMatrix"]
    distCoef = calibData["distCoef"]
    rVectors = calibData["rVector"]
    tVectors = calibData["tVector"]
    reference_point = 0
    start_time = time.time()
    
    # marker data
    reference = []
    offset = []
    marker_data = []

    while window_created:
        frame_entered = cap.grab()
        _, frame = cap.retrieve()
        
        if frame is not None:
            grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            marker_corners, marker_IDs, reject = detector.detectMarkers(grayFrame)
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, markerSize, camMatrix, distCoef)

            # clear arrays
            reference = []
            offset = []
            if marker_IDs is not None:
                marker_index = range(0, marker_IDs.size)
                for id, corner, i in zip(marker_IDs, marker_corners, marker_index):
                    
                    # draw borders
                    cv.polylines(frame, [corner.astype(np.int32)], True, (255, 255, 255), 4, cv.LINE_AA)
                    cv.drawFrameAxes(frame, camMatrix, distCoef, rVec[i], tVec[i], 4, 4)
                    corner = corner.reshape(4, 2).astype(np.int32)
                    cv.putText(frame, f"id: {id}", corner[0], cv.FONT_HERSHEY_PLAIN, 1.3, (200, 200, 200), 2, cv.LINE_AA)
                    
                    if id == reference_point:
                        cv.putText(frame, f"Pos X = {round(tVec[i][0][0], 2)}", (0, 100), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                        cv.putText(frame, f"Pos Y = {round(tVec[i][0][1], 2)}", (0, 150), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                        cv.putText(frame, f"Pos Z = {round(tVec[i][0][2], 2)}", (0, 200), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)

                        rMat, _ = cv.Rodrigues(rVec[i][0])
                        rotation_array = aruco_filter.ArucoFilter.rotationMatrixToEulerAngles(rMat) # x, y, z
                        
                        cv.putText(frame, f"Rot X = {round(rotation_array[0], 2)}", (0, 250), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                        cv.putText(frame, f"Rot Y = {round(rotation_array[1], 2)}", (0, 300), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                        cv.putText(frame, f"Rot Z = {round(rotation_array[2], 2)}", (0, 350), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                        
                        # from the reference point, 3cm to x direction 2.5cm to y direction
                        local_offset = np.array([3, 2.5, 0]) # x, y, z
                        
                        # transform from local to global coordinate
                        global_offset = rMat.dot(local_offset) + tVec[i][0]
                        
                        cv.putText(frame, f"Pos X = {round(global_offset[0], 2)}", (200, 100), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                        cv.putText(frame, f"Pos Y = {round(global_offset[1], 2)}", (200, 150), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                        cv.putText(frame, f"Pos Z = {round(global_offset[2], 2)}", (200, 200), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
                        
                        reference.append(round(tVec[i][0][0], 2))
                        reference.append(round(tVec[i][0][1], 2))
                        reference.append(round(tVec[i][0][2], 2))
                        offset.append(round(global_offset[0], 2))
                        offset.append(round(global_offset[1], 2))
                        offset.append(round(global_offset[2], 2))
                        
                timeElapsed = time.time() - start_time
                
                if reference and offset:
                    markerCoordinates = np.hstack((timeElapsed, reference, offset))
                    print(markerCoordinates)
                
                if reference and offset:
                    coordinates = np.hstack((timeElapsed, reference, offset))
                    marker_data.append(coordinates)
                    
        cv.imshow("hello", frame)
        frame = None
        frame_entered = None
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyWindow("hello")
'''