import gradio as gr
import numpy as np
import cv2
import os
import torch
from detect import peekFace

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #图像宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #图像高度

isEnd = False
one_turn_res = "" 

SCALE = 0.05
       
def real_time_emotion_detect():
    global isEnd
    isEnd = False

    feeling = []
    frame_count = 0
    emotion_list = np.array(['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    emotion_result = ""
    
    # 加载模型到 CUDA 设备上
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu')
    print(device)
    model = torch.load('weights/vgg_it100.pkl', map_location=device)
    model.to(device)
    
    faceCascade = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while not isEnd:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )
        for (x, y, w, h) in faces:
            OFF_SET_X = int(w * SCALE)
            OFF_SET_Y = int(h * SCALE)
            x -= OFF_SET_X
            y -= OFF_SET_Y
            w += OFF_SET_X + OFF_SET_X
            h += OFF_SET_Y + OFF_SET_Y
            if x <= 0 or y <= 0 or x + w >= 640 or y + h >= 480:
                continue

            img = gray[x:x + w, y:y + h]
            if img is None or img.shape[0]<=0 or img.shape[1]<=0:
                continue
            img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA).reshape(1, 1, 48, 48) / 255.0

            # 将图像张量移动到 CUDA 设备上
            img = torch.tensor(img).to(torch.float32).to(device)
            with torch.no_grad():
                # 在 CUDA 设备上进行推理
                pred = model.forward(img)
            index = np.argmax(pred.cpu().data.numpy(), axis=1)  # 将结果移回 CPU
            emotion = emotion_list[index][0]
            frame_count += 1

            if len(feeling) < 30:
                feeling.append(emotion)
            else:
                feeling.pop(0)
                feeling.append(emotion)
                
            emotion_result = ""
            if frame_count >= 30:
                
                emo_dict = dict()
                for f in emotion_list:
                    emo_dict[f] = 0
                for i in feeling:
                    emo_dict[i] += 1

                easy_count = emo_dict["happy"] + emo_dict["surprise"]
                normal_count = emo_dict["neutral"]
                hard_count = emo_dict["sad"]+emo_dict["fear"]+emo_dict["anger"]+emo_dict["disgust"]
                
                if max(easy_count, normal_count, hard_count) == easy_count:
                    emotion_result = "go on!!!!!"
                elif max(easy_count, normal_count, hard_count) == normal_count:
                    emotion_result = "hold on:)"
                else:
                    emotion_result = "relax-_-"
                frame_count = 0

            cv2.putText(frame, f"result={emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if emotion_result != "" :
            global one_turn_res
            one_turn_res = emotion_result
        # Display the resulting frame
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame, one_turn_res

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()

def stop():
    global isEnd
    isEnd = True
    while isEnd :
        if os.path.exists("R-C.jpg"):
            img = cv2.cvtColor(cv2.imread("R-C.jpg",1), cv2.COLOR_BGR2RGB)
        yield img

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            video = gr.Image(
                label = '视频',
                sources = 'upload',
                width = 800,
                height = 600,
                interactive = False            
            )
            
        with gr.Column():
            with gr.Row():
                description = gr.Textbox(
                    label = "介绍",
                    value = "在这里添加介绍",
                    interactive = False
                )
            with gr.Row():
                result = gr.Textbox(
                    label = "识别结果",
                    value = "",
                    interactive = False
                )
            with gr.Row():
                start_btn = gr.Button("Start")
                start_btn.click(real_time_emotion_detect, outputs = [video, result])
            with gr.Row():
                stop_btn = gr.Button("Stop")
                stop_btn.click(stop, outputs = [video])
    
    demo.launch()
    cap.release()
    cv2.destroyAllWindows()
