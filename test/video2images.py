import numpy as np
import cv2
from PIL  import Image

cap = cv2.VideoCapture("test/data/01d51606163cd72070e670a01f569de3.mp4")  # 获取视频对象
isOpened = cap.isOpened  # 判断是否打开
# 视频信息获取
fps = cap.get(cv2.CAP_PROP_FPS)
fps_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)

imageNum = 0
sum=0
timef=fps_total // 100  #隔15帧保存一张图片

while (isOpened):

    sum+=1

    (frameState, frame) = cap.read()  # 记录每帧及获取状态

    if frameState == True and (sum % timef==0):

        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        frame = np.array(frame)

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        imageNum = imageNum + 1
        fileName = 'data/nerf_llff_data/popmart/images/' + str(imageNum) + '.jpg'  # 存储路径
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(fileName + " successfully write in")  # 输出存储状态

    elif frameState == False:
        break

print('finish!')
cap.release()
