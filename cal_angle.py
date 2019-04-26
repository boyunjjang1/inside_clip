#input 사진의 얼굴 각도 알아내기
#0~9.png의 사진이 input파일 안에 있을 때,
#각 사진의 각도를 확인
import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

import FaceRendering
import utils
import pickle

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

data_save = []
with open('input/user_test1_images' +'/data.pickle', 'rb') as f:
        data_save = pickle.load(f)

for i in range(0,9):
    img="input/user_test1_images/"+str(i)+".jpg"
    textureImg = cv2.imread(img)
    # height, width, channels = textureImg.shape
    # user_shapes2D = utils.getFaceKeypointsWithDetectedFace(textureImg, (0,0,width,height), predictor, maxImageSizeForDetection)
    (user_shapes2D, face_angle) = data_save[i]
    if user_shapes2D:
        print("{0} angle: {1}".format(img,utils.getFaceAngle(user_shapes2D)))