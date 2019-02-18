# import cv
import cv2
from PyQt5 import QtCore
import numpy as np
import os


class CameraDevice(QtCore.QObject):
    _DEFAULT_FPS = 30

    # newFrame = QtCore.pyqtSignal(cv.iplimage)
    newFrame = QtCore.pyqtSignal(np.ndarray)
    video_time_out = QtCore.pyqtSignal()

    def __init__(self, cameraId=0, video_path=None, mirrored=False, parent=None):
        super(CameraDevice, self).__init__(parent)

        self.mirrored = mirrored

        if video_path:
            self._cameraDevice = cv2.VideoCapture(video_path)
        else:
            self._cameraDevice = cv2.VideoCapture(cameraId)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(1000 / self.fps)

        self.paused = False

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        '''
        frame = cv.QueryFrame(self._cameraDevice)
        if self.mirrored:
            mirroredFrame = cv.CreateImage(cv.GetSize(frame), frame.depth, \
                                           frame.nChannels)
            cv.Flip(frame, mirroredFrame, 1)
            frame = mirroredFrame
        '''
        ret, frame = self._cameraDevice.read()
        if ret:
            self.newFrame.emit(frame)
        else:
            self._cameraDevice.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_time_out.emit()

    @property
    def paused(self):
        return not self._timer.isActive()

    @paused.setter
    def paused(self, p):
        if p:
            self._timer.stop()
        else:
            self._timer.start()

    @property
    def frameSize(self):
        w = cv2.cv.GetCaptureProperty(self._cameraDevice, cv2.CAP_PROP_FRAME_WIDTH)
        h = cv2.cv.GetCaptureProperty(self._cameraDevice, cv2.CAP_PROP_FRAME_HEIGHT)
        return int(w), int(h)

    @property
    def fps(self):
        fps = 0
        if not fps > 0:
            fps = self._DEFAULT_FPS
        return fps

    def set_video_path(self, video_path):
        if os.path.exists(video_path):
            self.paused = True
            self._cameraDevice = cv2.VideoCapture(video_path)
            self.paused = False
        else:
            print('video path ' + video_path + ' does not exist!')

    def set_image_path(self, image_path):
        if os.path.exists(image_path):
            frame = cv2.imread(image_path)
            if frame is not None:
                self.newFrame.emit(frame)
        else:
            print('image path ' + image_path + ' does not exist!')
