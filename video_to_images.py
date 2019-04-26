#input 동영상에서 각도에 따라 자동으로 사진 저장
import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

import FaceRendering
import utils
import os
import shutil
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

#input 동영상에서 각도에 따라 자동으로 사진 저장하는 함수
#prameter: 사진으로 바꿀 동영상 이름(str)
#각도 0~8 이 모두 정상적으로 저장됐을 경우 true return
#아닐 경우 false return -> ui에서 사용자가 다시 동영상 찍을 것 요구
#비디오 사이즈가 1:1에 가까워야지 얼굴인식을 하는것 같다..
def video_to_images(user_video):
    vidcap = cv2.VideoCapture("./input/" + str(user_video))

    if not vidcap.isOpened():
        raise Exception("Could not open video device")

    angle = 0
    #각도 0~8 까지 모두 추출해 낼 수 있는지 확인
    video_check = []

    folder_name = "./input/"+str(user_video)+"_images"
    #폴더가 없으면 만든다
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    while(vidcap.isOpened()):
        ret, cameraImg = vidcap.read()
        #유저 쉐입
        user_shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

        #영상이 끝나면 반복문 탈출
        if ret == False:
            break

        #얼굴 정면이 모두 인식된다면
        if user_shapes2D:
            face_angle = utils.getFaceAngle(user_shapes2D)
            #유저 텍스처
            if face_angle == angle:
                #캡쳐된 이미지를 저장하는 함수
                cv2.imwrite(folder_name+"/%d.jpg" % angle,cameraImg)
                print('angle %d.jpg saved' % angle)
                video_check.append(angle)

                textureCoords = utils.getFaceTextureCoords_v2(cameraImg, mean3DShape, blendshapes, idxs2D, idxs3D, user_shapes2D, predictor)#, maxImgSizeForDetection=320)
                data_save.append((cameraImg, face_angle, user_shapes2D, textureCoords))
                
                if angle == 8: #8
                    break
                else :
                    angle += 1
            else :
                pass
        else :
            print("no face")
    vidcap.release()

    with open(folder_name +'/userdata.pickle', 'wb') as f:
        pickle.dump(data_save,f)

    #모든 각도의 값이 저장되었는지 확인
    check = True
    print (video_check)
    if len(video_check) == 9:
        for i in range (len(video_check)):
            if not video_check[i] == i:
                check = False
    else :
        check = False

    #모든각도 추출 못했으면 폴더 지우기
    if not check:
        try:
            shutil.rmtree(folder_name)
        except OSError as e:
            if e.errno == 2:
                # 파일이나 디렉토리가 없음!
                print ("No such file or directory to remove")
            else:
                raise
    return check

# print(video_to_images(""))
