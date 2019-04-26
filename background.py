#https://www.riverbankcomputing.com/static/Docs/PyQt5/
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import *
from PyQt5 import uic
import video2
from selectGender import *
import subprocess
import test_main
import dlib

import datetime
import pickle

ui = uic.loadUiType("background.ui")[0] #파일이 여러개 있을 수 있어서 0번째

class Form(QWidget, ui):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        pixmap = QPixmap("./input/testvideo1.png")
        self.imageViewer_1.setPixmap(pixmap)
        self.imageViewer_1.setScaledContents(True)
        self.pushButton_1.clicked.connect(self.clicked_video1)
        
        pixmap = QPixmap("./input/testvideo2.png")
        self.imageViewer_2.setPixmap(pixmap)
        self.imageViewer_2.setScaledContents(True)
        self.pushButton_2.clicked.connect(self.clicked_video2)
        
        pixmap = QPixmap("./input/testvideo3.png")
        self.imageViewer_3.setPixmap(pixmap)
        self.imageViewer_3.setScaledContents(True)
        self.pushButton_3.clicked.connect(self.clicked_video3)
        
        pixmap = QPixmap("./input/testvideo4.png")
        self.imageViewer_4.setPixmap(pixmap)
        self.imageViewer_4.setScaledContents(True)
        self.pushButton_4.clicked.connect(self.clicked_video4)        

    def clicked_video1(self):
        backgroundvideo = "./input/testvideo1.mp4"
        backgroundvideo_name = "testvideo1"
        with open('global.pickle', 'rb') as pf:
            data = pickle.load(pf)
            data['backgroundvideo'] = backgroundvideo
            data['backgroundvideo_name'] = backgroundvideo_name
            with open('global.pickle', 'wb') as pf:
                pickle.dump(data, pf)
        test_main.mix_video()

    def clicked_video2(self):
        backgroundvideo = "./input/testvideo2.mp4"
        backgroundvideo_name = "testvideo2"
        with open('global.pickle', 'rb') as pf:
            data = pickle.load(pf)
            data['backgroundvideo'] = backgroundvideo
            data['backgroundvideo_name'] = backgroundvideo_name
            with open('global.pickle', 'wb') as pf:
                pickle.dump(data, pf)
        test_main.mix_video()

    def clicked_video3(self):
        backgroundvideo = "./input/testvideo3.mp4"
        backgroundvideo_name = "testvideo3"
        with open('global.pickle', 'rb') as pf:
            data = pickle.load(pf)
            data['backgroundvideo'] = backgroundvideo
            data['backgroundvideo_name'] = backgroundvideo_name
            with open('global.pickle', 'wb') as pf:
                pickle.dump(data, pf)
        test_main.mix_video()

    def clicked_video4(self):
        backgroundvideo = "./input/testvideo4.mp4"
        backgroundvideo_name = "testvideo4"
        with open('global.pickle', 'rb') as pf:
            data = pickle.load(pf)
            data['backgroundvideo'] = backgroundvideo
            data['backgroundvideo_name'] = backgroundvideo_name
            with open('global.pickle', 'wb') as pf:
                pickle.dump(data, pf)
        test_main.mix_video()