import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils

import datetime
import time

print("Press T to draw the keypoints and the 3D model") #결과 영상에 얼굴합성 과정 표시
print("Press R to start recording to a video file") #녹화

#shape_predictor_64_face_landmarks.dat 여기서 다운받아서 압축해제
#http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2rr

# 키포인트 인식 모델
predictor_path = "data/shape_predictor_68_face_landmarks.dat"

#이미지 사이즈가 작을수록 처리속도가 빨라짐
#너무 작으면 얼굴 인식이 안됨
maxImageSizeForDetection = 320

# 얼굴 인식기 로드
detector = dlib.get_frontal_face_detector()
#detector = dlib.cnn_face_detection_model_v1("data/mmod_human_face_detector.dat")
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

# 본인의 여러 얼굴을 메모리에 저장함 (0~9)
textureImgs = []
for i in range(0,9):
    textureImgs.append(cv2.imread("input/"+str(i)+".png")) # Image for face
output_video_name = 'output/'+ backgroundvideo +'_'+ datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.avi'
writer = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'XVID'), 25, (cameraImg.shape[1], cameraImg.shape[0]), True) 

framecnt = 0
modelParams = np.zeros(20)
startTime = time.time()
while True:
    # 배경으로 사용할 영상의 프레임 이미지 읽기
    ret, cameraImg = cap_background.read()
    
    #영상이 끝나면 반복문 탈출
    if ret == False:
        break
    
    # 영상에서 얼굴을 인식하고, 키포인트를 추출함
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

    #shapes2D가 비어있지 않다면
    if shapes2D:
        textureImg = textureImgs[utils.getFaceAngle(shapes2D)]

        # 유저 얼굴 영상에서 얼굴을 인식하고, 키포인트를 추출함
        user_shapes2D = utils.getFaceKeypoints(textureImg, detector, predictor, maxImageSizeForDetection)
        # 배경영상 얼굴의 각도를 이용하여 유저 얼굴이미지 선택
        if user_shapes2D is not None:
            print("user_face_angle: {}".format(utils.getFaceAngle(user_shapes2D)))

        textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
        renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)
        
        for shape2D in shapes2D:

            if framecnt == 0:
                #3D 모델 파라미터 초기화 (영상에서 인식된 얼굴로부터 3D 모델 생성을 위해)
                modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])
            #3D 모델 파라미터 최적화 기능 (배경의 얼굴과 입력된 얼굴의 키포인트 간의 거리를 최소화 하도록 머신러닝으로 최적화)
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

            #위의 모델을 이용해 입력된 이미지의 얼굴을 3D객체로 바꿔 배경의 얼굴 위치로 보정
            shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D)

            #배경 영상과 입력된 얼굴 이미지를 합성 (합성과정에서 색변환, 이미지 블랜딩 기법 사용)
            mask = np.copy(renderedImg[:, :, 0])
            renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
            cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)
       
            # 3D매쉬와 키포인트를 화면 위에 그림
            if drawOverlay:
                #초록색 
                drawPoints(cameraImg, shape2D.T)
                #?
                drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)
            framecnt += 1
    if writer is not None:
        writer.write(cameraImg)

    # 얼굴 합성된 영상 출력
    cv2.imshow('image', cameraImg)

endTime = time.time() - startTime
writer.release()
print("endTime: ")
print(endTime) 
