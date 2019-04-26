#https://www.riverbankcomputing.com/static/Docs/PyQt5/
import sys

from PyQt5.QtWidgets import *
import video
import pickle

if __name__ == "__main__":
    # data = {}
    # with open('global.pickle', 'rb') as pf:
    #     data = pickle.load(pf)

    # with open('global.pickle', 'wb') as pf:
    #     pickle.dump(data, pf)

    app = QApplication(sys.argv)
    mainForm = video.Form() # 클래스 이름이 Form
    mainForm.show()
    app.exec_()