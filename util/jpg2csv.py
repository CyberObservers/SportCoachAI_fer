import os
from PIL import Image
import numpy as np
import pandas as pd
import random

emo2num = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}


def get_jpg_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            files.append(os.path.join(path, file))
    random.shuffle(files)
    return files


def resize_image(files):
    size = (48, 48)
    jpgData = {
        'emotion': [],
        'pixels': []
        # 'set': []
    }

    # num = len(files)
    # train = int(num * 0.7)

    # i = 0
    for file in files:
        # 用灰度图读入
        img = Image.open(file).convert('L').resize(size)
        img = np.array(img)

        jpgData['pixels'].append(" ".join(map(str, img.reshape(-1))))
        jpgData['emotion'].append(emo2num[file.split('_')[1]])
        # jpgData['set'].append('train' if i < train else 'test')
        # i += 1

    df = pd.DataFrame(jpgData)

    # 将DataFrame写入CSV文件
    csv_file_path = '../data.csv'
    df.to_csv(csv_file_path, index=False)



if __name__ == "__main__":
    folder_path = "../data/visual/"  # 文件夹路径
    jpg_files = get_jpg_files(folder_path)
    resize_image(jpg_files)
