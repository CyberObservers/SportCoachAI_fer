import cv2
import torch
import numpy as np
# import argparse
from models import VGG, FaceCNN


OFF_SET_X = 20
OFF_SET_Y = 20
emo_dict = np.array(['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--img', type=str)
    # opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'
    model = torch.load('weights/vgg_it100.pkl', map_location=device)
    faceCascade = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')

    cv2.namedWindow("preview")
    frame = cv2.imread('./img.png')
    height, width, _ = frame.shape
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

        x1 = 0 if x <= 0 else x
        y1 = 0 if y <= 0 else y
        x2 = width if x+w > width else x+w
        y2 = height if y+h > height else y+h

        img = gray[x1:x2, y1:y2]
        if img is None:
            continue
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA).reshape(1, 1, 48, 48) / 255.0

        img = torch.tensor(img).to(torch.float32).to(device)
        with torch.no_grad():
            pred = model.forward(img)
        index = np.argmax(pred.data.cpu().numpy(), axis=1)
        res = f"result={emo_dict[index][0]}"

        cv2.putText(frame, res, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("preview", frame)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
