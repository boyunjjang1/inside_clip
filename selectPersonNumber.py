from PyQt5.QtWidgets import *
from PyQt5 import uic
import selectGender
import pickle

ui = uic.loadUiType("selectPersonNumber.ui")[0] #파일이 여러개 있을 수 있어서 0번째

class Form(QMainWindow, ui):
    def person1_button(self):
        number = 1
        with open('global.pickle', 'rb') as pf:
            data = pickle.load(pf)
            data['personnumber'] = number
            with open('global.pickle', 'wb') as pf:
                pickle.dump(data, pf)

        self.sp = selectGender.FormGender()
        self.sp.show()

    def person2_button(self):
        number = 2
        with open('global.pickle', 'rb') as pf:
            data = pickle.load(pf)
            data['personnumber'] = number
            with open('global.pickle', 'wb') as pf:
                pickle.dump(data, pf)
        self.sp = selectGender.FormGender()
        self.sp.show()

    def __init__(self):
        super().__init__()
        self.setupUi(self)


