import cv2 as cv
import numpy
import time

class WindowManager(object):
    def __init__(self, windowName, keypressCallback=None):
        self._windowName = windowName
        self._isWindowCreated = False
        
        self.keypressCallback = keypressCallback
        
    @property
    def isWindowCreated(self):
        return self._isWindowCreated
    
    def createWindow(self):
        cv.namedWindow(self._windowName)
        self._isWindowCreated = True
    
    def destroyWindow(self):
        cv.destroyWindow(self._windowName)
        self._isWindowCreated = False
        
    def show(self, frame):
        cv.imshow(self._windowName, frame)
    
    def processEvents(self):
        keycode = cv.waitKey(1)
        
        if self.keypressCallback is not None and keycode != -1:
            # discard any non-ASCII info encoded  by GTK
            keycode &= 0xFF
            self.keypressCallback(keycode)
        
        
class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        self._capture = capture
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        
        self._enteredFrame = False
        self._frame = None
        
        self._imageFilename = None
        
        self._starTime = None
        self._framesElapsed = int(0)
        self._fpsEstimate = None
        
    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame
    
    @property
    def isWritingVideo(self):
        return self._videoFilename is not None
    
    @property
    def isWritingImage(self):
        return self._imageFilename is not None
    
    def writeImage(self, filename):
        self._imageFilename = filename
        
    def enterFrame(self):
        
        # check that any previous frame was exited
        assert not self._enteredFrame
        
        if self._capture is not None:
            self._enteredFrame = self._capture.grab() # True, False
    
    def exitFrame(self):
        
        # check whether any grabbed fram is retrieved
        if self._frame is None:
            self._enteredFrame = False
            return
        
        # update the FPS estimate and related variables
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        
        self._framesElapsed += 1
        
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = numpy.fliprlr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)
        
        # write to the image file, if any
        if self.isWritingImage:
            cv.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None
        
        # release the frame
        self._frame = None
        self._enteredFrame = False