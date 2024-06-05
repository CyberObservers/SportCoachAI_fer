import cv2
import torch
import numpy as np
from util import resize_image
from models import VGG, FaceCNN, DenseNet121


# OFF_SET_X = 20
# OFF_SET_Y = 20
emo_dict = np.array(['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])


def peekFace():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'
    # model = torch.load('weights/vgg_it100.pkl', map_location=device)
    model = torch.load('weights/model_it60.pkl', map_location=device)
    model.eval()
    faceCascade = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    width = 1920
    height = 1080
    cap.set(3, width)  # set Width
    cap.set(4, height)  # set Height
    cv2.namedWindow("preview")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )
        for (x, y, w, h) in faces:
            OFF_SET_X = int(w*0.05)
            OFF_SET_Y = int(h*0.05)
            x -= OFF_SET_X
            y -= OFF_SET_Y
            w += OFF_SET_X + OFF_SET_X
            h += OFF_SET_Y + OFF_SET_Y

            x1 = 0 if x <= 0 else x
            y1 = 0 if y <= 0 else y
            x2 = width if x + w > width else x + w
            y2 = height if y + h > height else y + h

            img = gray[x1:x2, y1:y2]
            # img_w, img_h = img.shape
            if img is None or img.shape[0]<=0 or img.shape[1]<=0:
                continue
            img = np.array([resize_image(img, (48, 48))])
            # img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA).reshape(1, 1, 48, 48) / 255.0

            img = torch.tensor(img).to(torch.float32).to(device)
            with torch.no_grad():
                pred = model.forward(img)
            index = np.argmax(pred.data.cpu().numpy(), axis=1)
            res = f"result={emo_dict[index][0]}"

            cv2.putText(frame, res, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("preview", frame)

        key = cv2.waitKey(24)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    peekFace()
