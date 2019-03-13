#input 동영상에서 각도에 따라 자동으로 사진 저장
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
vidcap = cv2.VideoCapture('input/borahvideo.mp4')
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



angle = 0
while(vidcap.isOpened()):
    ret, image = vidcap.read()
    cameraImg = cap_background.read()[1]
    user_shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)
    if user_shapes2D is not None:
        face_angle=utils.getFaceAngle(user_shapes2D)
        print("angle: {}".format(face_angle))

        #get() 함수를 이용해 전체 프레임 중 1/n의 프레임만 가져와 저장
        if face_angle == angle:
            #캡쳐된 이미지를 저장하는 함수
            cv2.imwrite("input/borahvideo_images/%d.jpg" % angle,image)

            print('angle %d.jpg saved' % angle)
            angle += 1
    if angle==10 : 
        break

vidcap.release()
