import numpy as np
import cv2
import models
from dlib import rectangle
import NonLinearLeastSquares
import math


def getNormal(triangle):
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    axisX = b - a
    axisX = axisX / np.linalg.norm(axisX)
    axisY = c - a
    axisY = axisY / np.linalg.norm(axisY)
    axisZ = np.cross(axisX, axisY)
    axisZ = axisZ / np.linalg.norm(axisZ)

    return axisZ


def flipWinding(triangle):
    return [triangle[1], triangle[0], triangle[2]]


def fixMeshWinding(mesh, vertices):
    for i in range(mesh.shape[0]):
        triangle = mesh[i]
        normal = getNormal(vertices[:, triangle])
        if normal[2] > 0:
            mesh[i] = flipWinding(triangle)

    return mesh


def getShape3D(mean3DShape, blendshapes, params):
    # skalowanie
    s = params[0]
    # rotacja
    r = params[1:4]
    # przesuniecie (translacja)
    t = params[4:6]
    w = params[6:]

    # macierz rotacji z wektora rotacji, wzor Rodriguesa
    R = cv2.Rodrigues(r)[0]
    shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

    shape3D = s * np.dot(R, shape3D)
    shape3D[:2, :] = shape3D[:2, :] + t[:, np.newaxis]

    return shape3D


def getMask(renderedImg):
    mask = np.zeros(renderedImg.shape[:2], dtype=np.uint8)


def load3DFaceModel(filename):
    faceModelFile = np.load(filename)
    mean3DShape = faceModelFile["mean3DShape"]
    mesh = faceModelFile["mesh"]
    idxs3D = faceModelFile["idxs3D"]
    idxs2D = faceModelFile["idxs2D"]
    blendshapes = faceModelFile["blendshapes"]
    mesh = fixMeshWinding(mesh, mean3DShape)

    return mean3DShape, blendshapes, mesh, idxs3D, idxs2D


def getFaceKeypointsWithDetectedFace(img, faceRectangle, predictor, maxImgSizeForDetection=640):
    shapes2D = []
    # 얼굴의 특징점 검출
    dlibShape = predictor(img, rectangle(faceRectangle))
    shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
    # 모양이 2 x n이 아닌 n x 2가되도록 전치하면 나중에 계산이 쉬워집니다.
    shape2D = shape2D.T
    shapes2D.append(shape2D)
    return shapes2D


def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=320):
    imgScale = 1
    scaledImg = img
    if img is not None:
        if max(img.shape) > maxImgSizeForDetection:
            imgScale = maxImgSizeForDetection / float(max(img.shape))
            scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

        # detekcja twarzy
        dets = detector(scaledImg, 1)
        # print("number of face: {}".format(len(dets)))

    else:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale),
                                  int(det.bottom() / imgScale))
        # 얼굴의 특징점 검출
        dlibShape = predictor(img, faceRectangle)
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        # 모양이 2 x n이 아닌 n x 2가되도록 전치하면 나중에 계산이 쉬워집니다.
        shape2D = shape2D.T
        shapes2D.append(shape2D)

    return shapes2D


def getFaceTextureCoords(img, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor, maxImgSizeForDetection=320):
    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

    keypoints = getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection)[0]
    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], keypoints[:, idxs2D])
    modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, (
    [mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], keypoints[:, idxs2D]), verbose=0)
    textureCoords = projectionModel.fun([mean3DShape, blendshapes], modelParams)

    return textureCoords


def getFaceTextureCoords_v2(img, mean3DShape, blendshapes, idxs2D, idxs3D, myshape, predictor):
    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

    keypoints = myshape[0]
    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], keypoints[:, idxs2D])
    modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, (
    [mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], keypoints[:, idxs2D]), verbose=0)
    textureCoords = projectionModel.fun([mean3DShape, blendshapes], modelParams)

    return textureCoords


def getFaceAngle(shapes2D):
    # 왼쪽, 오른쪽의 기준은 영상 기준이 아니라 영상을 보는 사람의 기준
    # 두 점 사이의 길이:root((x1-x2)^2+(y1-y2)^2)
    # 왼쪽 볼 길이 (32번 포인트 - 3번 포인트)
    leftCheekLen = math.sqrt(
        math.pow(shapes2D[0][0][31] - shapes2D[0][0][2], 2) + math.pow(shapes2D[0][1][31] - shapes2D[0][1][2], 2))
    # 오른쪽 볼 길이 (15번 포인트 - 36번 포인트)
    rightCheekLen = math.sqrt(
        math.pow(shapes2D[0][0][14] - shapes2D[0][0][35], 2) + math.pow(shapes2D[0][1][14] - shapes2D[0][1][35], 2))

    # print("leftCheekLen: {}  rightCheekLen: {}".format(leftCheekLen,rightCheekLen))

    # 얼굴 수평 길이 (15번 포인트 - 3번 포인트)
    faceParallelLen = math.sqrt(
        math.pow(shapes2D[0][0][14] - shapes2D[0][0][2], 2) + math.pow(shapes2D[0][1][14] - shapes2D[0][1][2], 2))
    # (비율) 왼쪽 볼 길이 / 얼굴 수평 길이
    leftCheekRatio = leftCheekLen / faceParallelLen
    # (비율) 오른쪽 볼 길이 / 얼굴 수평 길이
    rightCheekRatio = rightCheekLen / faceParallelLen
    # print("leftCheekRatio: {}  rightCheekRatio: {}".format(leftCheekRatio,rightCheekRatio))

    # 왼쪽 볼 비율- 오른 쪽 비율(정면 보고 있다고 가정, x축만)
    face_x_dif = leftCheekRatio - rightCheekRatio
    # print("face_x_dif: {}".format(face_x_dif))

    # face_x_dif
    ret = 0
    # ~-0.4: 왼쪽으로 0
    if (face_x_dif < -0.4):
        ret = 0

    # -0.4~-0.3: 왼쪽으로 1
    elif (-0.4 <= face_x_dif) & (face_x_dif < -0.3):
        ret = 1

    # -0.3~-0.2: 왼쪽으로 2
    elif (-0.3 <= face_x_dif) & (face_x_dif < -0.2):
        ret = 2

    # -0.2~-0.1: 왼쪽으로 3
    elif (-0.2 <= face_x_dif) & (face_x_dif < -0.1):
        ret = 3

    # -0.1~0.1: 정면 4
    elif (-0.1 <= face_x_dif) & (face_x_dif <= 0.1):
        ret = 4

    # 0.1 ~ 0.2: 오른쪽으로  5
    elif (0.1 <= face_x_dif) & (face_x_dif < 0.2):
        ret = 5

    # 0.2 ~ 0.3: 오른쪽으로 6
    elif (0.2 <= face_x_dif) & (face_x_dif < 0.3):
        ret = 6

    # 0.3 ~ 0.4: 오른쪽으로 7
    elif (0.3 <= face_x_dif) & (face_x_dif < 0.4):
        ret = 7

    # 0.4~: 오른쪽으로 8
    else:
        ret = 8

    # print("angle: {}".format(ret))
    return ret

