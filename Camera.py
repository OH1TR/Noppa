import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
from PyQt4 import QtGui
import numpy as np


class Camera():

    def __init__(self):
        super(Camera, self).__init__()
        self.imageLabel = None
        self.camImage = None
        self.runtime = None

    def initCam(self):
        init = zcam.PyInitParameters()
        self.cam = zcam.PyZEDCamera()
        if not self.cam.is_opened():
            print("Opening ZED Camera...")
        status = self.cam.open(init)
        if status != tp.PyERROR_CODE.PySUCCESS:
            print(repr(status))
            exit()

        self.runtime = zcam.PyRuntimeParameters()
        self.camImage = core.PyMat()


    def toQImage(self, im, copy=False):
        if im is None:
            return QtGui.QImage()

        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_Indexed8)
                qim.setColorTable(QtGui.gray_color_table)
                return qim.copy() if copy else qim

            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
                    return qim.copy() if copy else qim


    def PixmapFromArray(self,arr):
        if(type(arr) is np.ndarray):
            return (QtGui.QPixmap.fromImage(self.toQImage(np.asarray(arr), False)))
        else:
            return(QtGui.QPixmap.fromImage(self.toQImage(arr,False)))

    def GrabArray(self):
        if self.camImage is None:
            self.initCam()

        # ZED does some buffering? And first image is quite old.
        for x in range(1,10):
            err = self.cam.grab(self.runtime)
            if err == tp.PyERROR_CODE.PySUCCESS:
                self.cam.retrieve_image(self.camImage)

        while err!=tp.PyERROR_CODE.PySUCCESS:
            err = self.cam.grab(self.runtime)

        if err == tp.PyERROR_CODE.PySUCCESS:
            self.cam.retrieve_image(self.camImage)
            return(self.camImage.get_data())
        else:
            print('Error capturing image')

    def GrabPixmap(self):
        return (self.PixmapFromArray(self.GrabArray()))