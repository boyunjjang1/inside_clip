import dlib
import cv2
import numpy as np

import csv
import pandas as pd

import models
import NonLinearLeastSquares
import ImageProcessing

import FaceRendering
import utils

from os import rename, listdir

# 키포인트 인식 모델
predictor_path = "data/shape_predictor_68_face_landmarks.dat"

#이미지 사이즈가 작을수록 처리속도가 빨라짐
#너무 작으면 얼굴 인식이 안됨
maxImageSizeForDetection = 320

# 얼굴 인식기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("data/candide.npz")

projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
lockedTranslation = False
drawOverlay = False
writer = None

backgroundvideo ="testvideo3"

# 콤비네이션: VC:Video,Cam | CI: Cam,Image | CC: Cam, Cam | VI: Video,Image
cap_background = cv2.VideoCapture("input/"+backgroundvideo+".mp4") # Video for background
# cap_background = cv2.VideoCapture(0) # WebCAM for background
cameraImg = cap_background.read()[1]

modelParams = np.zeros(20)



with open("facial_points/"+backgroundvideo+".csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['frame_number','unknown(0)/male(1)/female(2)','facial_landmarks(x-56,y-56)'])
    framecnt = 0

    while True:
        #프레임 세기
        framecnt += 1
        print(framecnt)
        # 배경으로 사용할 영상의 프레임 이미지 읽기
        ret, cameraImg = cap_background.read()
        
        #영상이 끝나면 반복문 탈출
        if ret == False:
            break

        # 영상에서 얼굴을 인식하고, 키포인트를 추출함
        shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)
        
        #shapes2D가 none이면 얼굴인식이 안되는 상황,
        #none이 아니면 얼굴인식이 되서 페이셜포인트 찾은 상황
        if shapes2D is not None:

            frame = [framecnt,0]
            print (frame)
  
            rows =np.append(frame,shapes2D[0][0])
            rows = np.append(rows,shapes2D[0][1]) 
            csvwriter.writerow(rows)
        
        else :
            #인식 못했을 경우 빈 줄
            csvwriter.writerow("")