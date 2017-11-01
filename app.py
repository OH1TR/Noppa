import sys
from PyQt4 import QtGui
import cv2
from Camera import *



class MyWindow(QtGui.QWidget):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.camera=Camera()
        self.setupUi()

    def setupUi(self):
        image = QtGui.QPixmap()
        self.setGeometry(100, 100, 700, 700)
        self.imageLabel =QtGui.QLabel(self)
        self.imageLabel.move(50, 20)
        self.imageLabel.setFixedHeight(600)
        self.imageLabel.setFixedWidth(600)

        button = QtGui.QPushButton(self)
        button.clicked.connect(self.onClick)
        button.setText('Capture')
        self.setWindowTitle('Noppa')
        self.show()


    def onClick(self):
        arr=self.camera.GrabArray()
        x,y,z=arr.shape
        img=self.camera.PixmapFromArray(arr[int(x/2-113):int((x/2+114)),int(y/2-113):int(y/2+114),0:4].copy())
        self.imageLabel.setPixmap(img)


    def keyPressEvent(self, event):
        key = event.key()
        print(key)



if __name__=='__main__':
  app = QtGui.QApplication(sys.argv)
  ui = MyWindow()

  sys.exit(app.exec_())
