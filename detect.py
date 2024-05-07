import cv2
import torch
import numpy as np
from models import VGG, FaceCNN


OFF_SET_X = 20
OFF_SET_Y = 20
emo_dict = np.array(['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'
    model = torch.load('weights/vgg_it100.pkl', map_location=device)
    faceCascade = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # set Width
    cap.set(4, 480)  # set Height
    cv2.namedWindow("preview")

    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )
        for (x, y, w, h) in faces:
            x -= OFF_SET_X
            y -= OFF_SET_Y
            w += OFF_SET_X + OFF_SET_X
            h += OFF_SET_Y + OFF_SET_Y
            if x <= 0 or y <= 0 or x + w >= 640 or y + h >= 480:
                continue

            img = gray[x:x + w, y:y + h]
            if img is None:
                continue
            img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA).reshape(1, 1, 48, 48) / 255.0

            img = torch.tensor(img).to(torch.float32).to(device)
            with torch.no_grad():
                pred = model.forward(img)
            index = np.argmax(pred.data.cpu().numpy(), axis=1)
            res = f"result={emo_dict[index][0]}"

            cv2.putText(frame, res, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("preview", frame)

        key = cv2.waitKey(24)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
