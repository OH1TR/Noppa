import sys
from PyQt4 import QtGui
import cv2
from Camera import *
from PyQt4.QtCore import QTimer
import os
from lib.googlenet import GoogLeNet
from lib.config import params_setup
import os, json, yaml, requests, math
from PIL import Image
import codecs

class MyWindow(QtGui.QWidget):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.timer=None
        self.camera=Camera()
        args = params_setup()
        self.gnet = GoogLeNet(args=args)
        self.setupUi()

    def setupUi(self):
        self.setGeometry(100, 100, 700, 700)
        self.imageLabel =QtGui.QLabel(self)
        self.imageLabel.move(20, 20)
        self.imageLabel.setFixedHeight(300)
        self.imageLabel.setFixedWidth(300)

        self.setGeometry(100, 100, 700, 700)
        self.imageLabel2 =QtGui.QLabel(self)
        self.imageLabel2.move(350, 20)
        self.imageLabel2.setFixedHeight(300)
        self.imageLabel2.setFixedWidth(300)

        button = QtGui.QPushButton(self)
        button.clicked.connect(self.onClick)
        button.setText('Capture')

        self.logOutput = QtGui.QTextEdit(self)
        self.logOutput.setReadOnly(True)
        self.logOutput.setLineWrapMode(QtGui.QTextEdit.NoWrap)
        self.logOutput.move(20,350)
        self.logOutput.setFixedWidth(500)
        self.logOutput.setFixedHeight(300)

        font = self.logOutput.font()
        font.setFamily("Courier")
        font.setPointSize(10)


        self.setWindowTitle('Noppa')
        self.show()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(300)


    def tick(self):
        self.onClick()


    def getClippedArray(self):
        arr=self.camera.GrabArray()
        x,y,z=arr.shape
        return(arr[int(x/2-113):int((x/2+114)),int(y/2-113):int(y/2+114),0:4].copy())


    def onClick(self):
        img=self.getClippedArray()
        bitmap=self.camera.PixmapFromArray(img)
        self.imageLabel.setPixmap(bitmap)

        bitmap.save("image.jpg","JPG")
        i=Image.open("image.jpg")
        i.resize((227, 227), Image.ANTIALIAS)
        i.load()
        im=np.asarray(i, dtype="float32")
        #im2 = np.asarray(i, dtype="uint8")
        #json.dump(im.tolist(), codecs.open('app.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

        #self.imageLabel2.setPixmap(self.camera.PixmapFromArray(im2))

        im /= 255

        probs = self.gnet.predict([im])[0]
        cnt = int(sum([math.exp(i+4) * probs[i] for i in range(len(probs))]))
        probs = [(i, round(100*p, 1)) for i, p in enumerate(probs)]

        txt=''
        for i in probs:
            n,v= i
            txt += '%i : %f\n' % (n+1,v)
        self.logOutput.clear()
        self.logOutput.insertPlainText(txt)


    def keyPressEvent(self, event):
        key = event.key()
        print(key)
        if(key==81):
            sys.exit(0)

        if(key>=49 and key<=54):
            img=self.camera.PixmapFromArray(self.getClippedArray())
            path=self.makeNewImagePath(key-48)
            img.save(path,'JPG')

    def makeNewImagePath(self,category):
        for i in range(1,100000):
            path='images/noppa/jpg/%i/image%i.jpg' % (category,i)
            if(not os.path.exists(path)):
                return(path)


if __name__=='__main__':
  app = QtGui.QApplication(sys.argv)
  ui = MyWindow()

  sys.exit(app.exec_())
