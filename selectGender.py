from PyQt5.QtWidgets import *
from PyQt5 import uic
import background
import video2

import csv
import numpy as np
import pickle

import os
import win32api
import pickle
import sys

ui = uic.loadUiType("selectGender.ui")[0] #파일이 여러개 있을 수 있어서 0번째
genderNum = 0

class FormGender(QMainWindow, ui):
    def womanSelect(self):
        genderNum = 2 #여자
        print (genderNum)
        with open('global.pickle', 'rb') as pf:
            data = pickle.load(pf)
            data['gender'] = genderNum
            with open('global.pickle', 'wb') as pf:
                pickle.dump(data, pf)

        self.sp = video2.Form()
        self.sp.show()

    def manSelect(self):
        genderNum = 1 #남자
        with open('global.pickle', 'rb') as pf:
            data = pickle.load(pf)
            data['gender'] = genderNum
            with open('global.pickle', 'wb') as pf:
                pickle.dump(data, pf)
        self.sp = video2.Form()
        self.sp.show()

    def __init__(self):
        super().__init__()
        self.setupUi(self)


def reading_csv(csvfile_name):
    data = {}
    if os.path.getsize('global.pickle') > 0:
        with open('global.pickle', 'rb') as pf:
            data = pickle.load(pf)
    else:
        win32api.MessageBox(0, '값이 제대로 전달되지 않았습니다. 처음부터 다시 시작해주세요', 'warning', 0x00001000) 
        sys.exit(0)

    genderNum = int(data['gender']) #1: 남자, 2:여자
    personnumber = int(data['personnumber'])

    f = open('facial_points/' + csvfile_name + '.csv', 'r')
    lines = csv.reader(f)
    cnt = 0
    ret = []

    isOnePerson = True
    if personnumber == 2:
        isOnePerson = False

    for line in lines:
        print(cnt)
        # type if line : list
        # line[0] : frame_number
        # line[1] : annotation -1 : none 1 : 남자 , 2 : 여자
        # line[2~69] : x
        # line[70~137] : y

        # 맨 위 정보는 저장 안하기
        if cnt == 0:
           print('pass')
        else:
            x_points = []
            y_points = []
            full_facialpoints = []

            # 얼굴 인식이 안되었을 떄
            if int(line[1]) == -1:
                ret.append(None)
                continue
            # 1만 불러왔을때 오류 :
            # d : [None]
            if not isOnePerson:
                 # x값 넣기
                for i in range(2, 70):
                    x_points.append(int(line[i]))

                # y값 넣기
                for i in range(70, 138):
                    y_points.append(int(line[i]))

                full_facialpoints.append(x_points)
                full_facialpoints.append(y_points)
                # print (type(full_facialpoints))
                # print(full_facialpoints)

                ret2 = np.array(full_facialpoints, dtype=np.int64)
                ret.append(ret2)
            else:
                if int(line[1]) == genderNum: #1: 남자, 2:여자
                    # x값 넣기
                    for i in range(2, 70):
                        x_points.append(int(line[i]))

                    # y값 넣기
                    for i in range(70, 138):
                        y_points.append(int(line[i]))

                    full_facialpoints.append(x_points)
                    full_facialpoints.append(y_points)
                    # print (type(full_facialpoints))
                    # print(full_facialpoints)

                    ret2 = np.array(full_facialpoints, dtype=np.int64)
                    ret.append(ret2)
                else:
                    ret.append(None)
        cnt += 1

    f.close()

    return ret

# print(reading_csv("testvideo1_annotation"))
