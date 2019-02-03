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

backgroundvideo = "testvideo1"
# 콤비네이션: VC:Video,Cam | CI: Cam,Image | CC: Cam, Cam | VI: Video,Image
cap_background = cv2.VideoCapture("input/"+backgroundvideo+".mp4") # Video for background
# cap_background = cv2.VideoCapture(0) # WebCAM for background
cameraImg = cap_background.read()[1]

# 본인의 여러 얼굴을 메모리에 저장함 (0~9)
textureImgs = []
for i in range(0,9):
    img="input/"+str(i)+".png"
    textureImgs.append(cv2.imread(img)) # Image for face
    textureImg = textureImgs[i]
    user_shapes2D = utils.getFaceKeypoints(textureImg, detector, predictor, maxImageSizeForDetection)
    if user_shapes2D is not None:
        print("{0} angle: {1}".format(img,utils.getFaceAngle(user_shapes2D)))