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

backgroundvideo = "testvideo3"
# 콤비네이션: VC:Video,Cam | CI: Cam,Image | CC: Cam, Cam | VI: Video,Image
cap_background = cv2.VideoCapture("input/"+backgroundvideo+".mp4") # Video for background
# cap_background = cv2.VideoCapture(0) # WebCAM for background
cameraImg = cap_background.read()[1]


framecnt = 0
modelParams = np.zeros(20)

csvfile = open("facial_points/"+backgroundvideo+".csv", "w", newline="\n")
while True:
    # 배경으로 사용할 영상의 프레임 이미지 읽기
    cameraImg = cap_background.read()[1]
    
    # 영상에서 얼굴을 인식하고, 키포인트를 추출함
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)
  
    #shapes2D가 none이면 얼굴인식이 안되는 상황,
    #none이 아니면 얼굴인식이 되서 페이셜포인트 찾은 상황
    if shapes2D is not None:
        #얼굴 키포인트를 프린트해봄
        print(shapes2D)
        print(type(shapes2D)) #list
        #CSVfile 로 저장하기
        csvwriter = csv.writer(csvfile)
        for row in shapes2D:
            csvwriter.writerow(row)
    else 
csvfile.close()
