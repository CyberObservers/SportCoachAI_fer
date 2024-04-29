import pandas as pd
import numpy as np


path = '../data.csv'

data = pd.read_csv(path)

pixels = data['pixels'].tolist()
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.array(face).reshape(48,48)
    print(face.shape)
