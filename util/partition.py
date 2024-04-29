import os
import glob
import pandas as pd


def get_jpg_files(directory):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为 .jpg
            if file.lower().endswith('.jpg'):
                # 构建文件的绝对路径并添加到列表中
                file_path = os.path.join(root, file)
                jpg_files.append(file_path)
    return jpg_files


train = []
test = []

files = get_jpg_files('./data/visual/')

train_num = int(len(files)*0.7)

for i in range(len(files)):
    if i < train_num:
        train.append(files[i])
    else:
        test.append(files[i])

print(f"Total number: {len(files)}, train set number: {len(train)}, test set number: {len(test)}")

pd.Series(train).to_csv('../train.txt', index=False, header=False)
pd.Series(test).to_csv('../test.txt', index=False, header=False)
