import datetime
import cv2
import video_to_images as vtoi


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

now = datetime.datetime.now().strftime("%d_%H")



while True:
    if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open()

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    now = datetime.datetime.now().strftime("%d_%H") # 전역 변수로 빼줘야함
    key = cv2.waitKey(33)

    if key == 27:
        break
    elif key == 24:
        print("녹화시작")
        record = True
        video = cv2.VideoWriter("./input/" + str(now) + ".avi", fourcc, 20, (frame.shape[1],frame.shape[0]))
    elif key == 3:
        print("녹화 중지")
        record = False
        video.release()
        temp = str(now) + ".avi"
        vtoi.video_to_images(temp)

    if record == True:
        print("녹화 중..")
        video.write(frame)

capture.release()
cv2.destroyAllWindows()

