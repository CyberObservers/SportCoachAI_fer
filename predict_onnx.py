import onnxruntime
import cv2
import numpy as np
from detect import resize_image

emo_dict = np.array(['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
image = cv2.imread('R-C.jpg', cv2.IMREAD_GRAYSCALE)
image = resize_image(image, (48, 48))
image.reshape((1, 48, 48))
img = np.array([image])

model = './weights/vgg_it100.onnx'
session = onnxruntime.InferenceSession(model)
inputs = {session.get_inputs()[0].name: img}
outs = session.run(None, inputs)[0]

print('onnx weights', outs)
print('onnx prediction', emo_dict[outs.argmax(axis=1)[0]])
