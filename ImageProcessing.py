import numpy as np
import cv2

# src 이미지는 dst 이미지 위에 붙여 넣을 이미지
# 깃 양은 무게를 측정 할 영역의 크기를 제어하는 데 사용되는 백분율입니다.
def blendImages(src, dst, mask, featherAmount=0.2):
    # 검정색이 아닌 픽셀 마스크의 인덱스 (마스크 이미지의 가운데 부분)
    maskIndices = np.where(mask != 0) # 0이 아닌 값의 index

    # a = np.array([1, 2, 3]), b = np.array([4, 5, 6]) 일떄 np.hstack(a, b) = array([1, 2, 3, 4, 5, 6])
    # x[:, np.newaxis] : 1 차원 늘림 array([0, 1, 2, 3, 4]) --> array([[0], [1], [2], [3], [4]])
    # maskIndices[0] = x축 좌표, maskIndices[1] = y축 좌표
    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis])) 

    # maskPts = [[x1,y1],[x2,y2],[x3,y3],[x4,y4], ...] : (2=axis0,n=axis1) 
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)

    # width 또는 height의 크기 중 큰값
    featherAmount = featherAmount * np.max(faceSize)

    # 외곽선(엄밀히말하면 아님) 구하기
    hull = cv2.convexHull(maskPts)
    
    dists = np.zeros(maskPts.shape[0]) # shape: (n, 2)

    for i in range(maskPts.shape[0]):
        # 이미지상의 어느 한 점과 어느 한 Contour와의 최단 거리를 구해주는 함수
        # 세번째 인자가 True면, 최단 거리를 리턴
        # dists: 외곽선과의 최단거리
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    # np.clip: array a, a_min, a_max
    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    # weight값만큼 원래 이미지의 픽셀 rgb값을 곱해 blending 시킴. src에서 weight만큼빠진 만큼 dst는 채워줘야함
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    # 외곽선 그리기
    #composedImg = cv2.drawContours(composedImg, hull, -1, (0,0,255), 10)

    return composedImg

#src는 색상을 가져올 이미지입니다.
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    # 검정색 픽셀이 아닌 마스크의 인덱스 (가운데 이미지 부분)
    maskIndices = np.where(mask != 0)
    
    # src[maskIndices [0], maskIndices [1]] 검정색 픽셀이 아닌 마스크의 픽셀 rgb값을 반환합니다.
    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    # maskedSrc.shape : (n, 3)
    # 모든 픽셀의 중간값
    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    # 칼라 블랜딩
    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst

