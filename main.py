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
