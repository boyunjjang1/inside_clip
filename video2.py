from PyQt5.QtWidgets import *
from PyQt5 import uic
import datetime
import cv2
import background
import video_to_images as vtoi
import sys
import pickle

now = datetime.datetime.now().strftime("%d_%H")
recapture = False #"True" --> True
from_cam = True

ui = uic.loadUiType("video2.ui")[0] #파일이 여러개 있을 수 있어서 0번째

class Form(QMainWindow, ui):
    def cameraConnected(self):
        capture = None
        if not from_cam:
            capture = cv2.VideoCapture('C:/Users/yoon/Desktop/창의자율과제_테스트/04_16_modify_강보윤/input/16_18.mp4')    
        else:
            capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        record = False
        frame_cnt = 0
        now = datetime.datetime.now().strftime("%d_%H")
        while True:
            if not from_cam:
                if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                    frame_cnt = 0
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ret, frame = capture.read()
            cv2.imshow("VideoFrame", frame)

            key = cv2.waitKey(33)
            if key == 27: # esc
                break
            elif key == 24: #ctrl + x
                print("start recording")
                record = True
                video =  cv2.VideoWriter("./input/" + str(now) + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 25,(frame.shape[1], frame.shape[0]), True)

            elif key == 26: #ctrl + z #3: ctrl + c
                print("stop recording")
                record = False
                video.release()
                temp = str(now) + ".avi"
                recapture = vtoi.video_to_images(temp)
                if recapture:
                    msg = QMessageBox(QMessageBox.Information, 'Notice', '성공적으로 인식되었습니다')
                    msg.exec()

                    with open('global.pickle', 'rb') as pf:
                        data = pickle.load(pf)
                        data['userfolder'] = str(now)+'.avi'+'_images'
                        with open('global.pickle', 'wb') as pf:
                            pickle.dump(data, pf)
                    
                    break
                else:
                    msg = QMessageBox(QMessageBox.Information, 'Notice', '다시 촬영해 주세요')
                    msg.exec()
                    if not from_cam:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if record == True:
                print("recordig..")
                video.write(frame)
            frame_cnt += 1

        cv2.destroyWindow('VideoFrame')

    def openBackground(self):
        self.sp = background.Form()
        self.sp.show()

    def __init__(self):
        super().__init__()
        self.setupUi(self)
