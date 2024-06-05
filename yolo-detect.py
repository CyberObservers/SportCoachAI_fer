import cv2
from ultralytics import YOLO
import torch
import numpy as np
from util import resize_image
from models import VGG, DenseNet121

RATIO_X = 0
RATIO_Y = 0

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('./weights/yolov8n-face.pt').to(device)
# fer = torch.load('./weights/vgg_it100.pkl', map_location=device)
fer = torch.load('./weights/model_it60.pkl', map_location=device)
emo_dict = np.array(['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
fer.eval()


# 打开摄像头
cap = cv2.VideoCapture(0)
cv2.namedWindow("preview")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width, height)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    with torch.no_grad():
        # 使用YOLOv8模型进行检测
        results = model(frame)

    # 在图像上绘制检测结果
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取单个边界框的坐标
            print(x1,y1,x2,y2, frame.shape)
            # w = x2 - x1
            # h = y2 - y1
            # x1 = max(x1 - int(w * RATIO_X), 0)
            # x2 = min(x2 + int(w * RATIO_X), width)
            # y1 = max(y1 - int(h * RATIO_Y), 0)
            # y2 = min(y2 + int(h * RATIO_Y), height)

            gray = frame[y1:y2, x1:x2]
            if gray is None:
                continue
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            img = np.array([resize_image(gray, (48, 48))])
            img = torch.tensor(img).to(torch.float32).to(device)
            with torch.no_grad():
                pred = fer.forward(img)
            index = np.argmax(pred.data.cpu().numpy(), axis=1)
            res = emo_dict[index][0]

            conf = box.conf.item()  # 获取置信度
            cls = int(box.cls.item())  # 获取类别标签
            label = f'{model.names[cls]} {conf:.2f} {res}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('preview', frame)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
