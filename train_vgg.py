""" CNN_face.py 基于卷积神经网络的面部表情识别(Pytorch实现) """
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from models import VGG

emo_dict = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}
emo_tensor = {
    'anger': torch.tensor([1., 0, 0, 0, 0, 0, 0]).float(),
    'disgust': torch.tensor([0, 1., 0, 0, 0, 0, 0]).float(),
    'fear': torch.tensor([0, 0, 1., 0, 0, 0, 0]).float(),
    'happy': torch.tensor([0, 0, 0, 1., 0, 0, 0]).float(),
    'sad': torch.tensor([0, 0, 0, 0, 1., 0, 0]).float(),
    'surprise': torch.tensor([0, 0, 0, 0, 0, 1., 0]).float(),
    'neutral': torch.tensor([0, 0, 0, 0, 0, 0, 1.]).float()
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaceDataset(data.Dataset):
    def __init__(self, DataPath):
        super(FaceDataset, self).__init__()

        data_frame = pd.read_csv(DataPath, header=None)
        column_index = 0
        self.path = data_frame.iloc[:, column_index].tolist()

    def __getitem__(self, item):
        face = Image.open(self.path[item]).convert('L').resize((48, 48))
        face = np.array(face) / 255.0
        face = face.reshape(1, 48, 48)

        face_tensor = torch.from_numpy(face)
        face_tensor = face_tensor.type('torch.FloatTensor')

        label = emo_tensor[self.path[item].split('_')[1]]
        return face_tensor, label

    def __len__(self):
        return len(self.path)


def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    lossData = {
        'epoch': [],
        'loss': []
    }
    accData = {
        'epoch': [],
        'train': [],
        'val': []
    }

    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    # model = FaceCNN().to(device)
    model = VGG('VGG19').to(device)
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        scheduler.step()  # 学习率衰减
        model.train()  # 模型训练
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model.forward(images)
            # 误差计算
            # print(output.shape, labels.shape)
            loss_rate = loss_function(output, labels)
            # 误差的反向传播
            loss_rate.backward()
            # 更新参数
            optimizer.step()

        # 打印每轮的损失
        print('After {} epochs , the loss_rate is : '.format(epoch + 1), loss_rate.item())
        lossData['epoch'].append(epoch + 1)
        lossData['loss'].append(loss_rate.item())
        if (epoch + 1) % 5 == 0 and epoch != 0:
            model.eval()  # 模型评估
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('After {} epochs , the acc_train is : '.format(epoch + 1), acc_train)
            print('After {} epochs , the acc_val is : '.format(epoch + 1), acc_val)
            accData['epoch'].append(epoch + 1)
            accData['train'].append(acc_train)
            accData['val'].append(acc_val)
        if (epoch + 1) % 20 == 0 and epoch != 0:
            torch.save(model, f'./vgg_res/model_it{epoch + 1}.pkl')
    pd.DataFrame(lossData).to_csv('./vgg_res/loss.csv')
    pd.DataFrame(accData).to_csv('./vgg_res/acc.csv')
    return model


def main():
    # 数据集实例化(创建数据集)
    train_dataset = FaceDataset('./fer_train.txt')
    val_dataset = FaceDataset('./fer_val.txt')
    # 超参数可自行指定
    model = train(train_dataset, val_dataset, batch_size=128, epochs=100, learning_rate=0.1, wt_decay=0)
    # 保存模型
    torch.save(model, './model_net.pkl')


if __name__ == '__main__':
    main()
