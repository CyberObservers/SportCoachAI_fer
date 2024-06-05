import argparse
import cv2
from ultralytics import YOLO
import torch
import numpy as np
from util import resize_image, extract_detections
import deep_sort.deep_sort.deep_sort as ds
from models import VGG, DenseNet121


def run(
        yolo_path: str,
        fer_path: str,
        deepsort_path: str,
        use_deepsort: int
):
    # Select Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # YOLO
    model = YOLO(yolo_path).to(device)

    # FER
    fer = torch.load(fer_path, map_location=device)
    emo_dict = np.array(['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    fer.eval()

    # DeepSort
    tracker = ds.DeepSort(deepsort_path, use_cuda=False)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("preview")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)

    if not cap.isOpened():
        raise RuntimeError("Could not open web camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        with torch.no_grad():
            # 使用YOLOv8模型进行检测
            results = model(frame)

        # 使用DeepSort
        if use_deepsort == 1:
            # 从预测结果中提取检测信息。
            detections, confarray = extract_detections(results, 0)

            # 使用deepsort模型对检测到的目标进行跟踪。
            resultsTracker = tracker.update(detections, confarray, frame)

            for x1, y1, x2, y2, Id in resultsTracker:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 将位置信息转换为整数。

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

                # conf = box.conf.item()  # 获取置信度
                # cls = int(box.cls.item())  # 获取类别标签
                label = f'{model.names[0]} {res} Id: {Id}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # 不使用DeepSort模块，cpu上推理速度更快
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取单个边界框的坐标
                    # print(x1,y1,x2,y2, frame.shape)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--YOLO", type=str, default='./weights/yolov8n-face.pt', help="YOLO detector")
    parser.add_argument("--tracker", type=str, default="./deep_sort/deep_sort/deep/checkpoint/ckpt.t7",
                        help="DeepSORT tracker")
    parser.add_argument("--fer", type=str, default="./weights/model_it60.pkl", help="FER classifier")
    parser.add_argument("--use_tracker", type=int, default=1, help="Decide whether to use DeepSort")

    opt = parser.parse_args()
    run(opt.YOLO, opt.fer, opt.tracker, opt.use_tracker)
