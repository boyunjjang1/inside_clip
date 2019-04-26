from PyQt5.QtWidgets import *
from PyQt5 import uic
# import video2
import selectPersonNumber


ui = uic.loadUiType("video.ui")[0] #파일이 여러개 있을 수 있어서 0번째




class Form(QMainWindow, ui):

    def startButtonClicked(self):
        # self.v2 = video2.Form()
        # self.v2.show()
        self.sp = selectPersonNumber.Form()
        self.sp.show()

    def __init__(self):
        super().__init__()
        self.setupUi(self)