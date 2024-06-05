import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from models import VGG, DenseNet121, ResNet101, DenseNet169
# from util import AsianDataset


emo_dict = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happiness': 3,
    'sadness': 4,
    'surprise': 5,
    'neutral': 6,
    "contempt": 1
}
# emo_dict = np.array(['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral', "contempt"])

emo_tensor = {
    'anger': torch.tensor([1., 0, 0, 0, 0, 0, 0]).float(),
    'disgust': torch.tensor([0, 1., 0, 0, 0, 0, 0]).float(),
    'fear': torch.tensor([0, 0, 1., 0, 0, 0, 0]).float(),
    'happiness': torch.tensor([0, 0, 0, 1., 0, 0, 0]).float(),
    'sadness': torch.tensor([0, 0, 0, 0, 1., 0, 0]).float(),
    'surprise': torch.tensor([0, 0, 0, 0, 0, 1., 0]).float(),
    'neutral': torch.tensor([0, 0, 0, 0, 0, 0, 1.]).float(),
    'contempt': torch.tensor([0, 1., 0, 0, 0, 0, 0]).float()
}
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu')

# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 将outputs和labels从GPU移至CPU，然后转换为numpy
            outputs = outputs.cpu()
            labels = labels.cpu()
            pred = np.argmax(outputs.data.numpy(), axis=1)
            truth = np.argmax(labels.data.numpy(), axis=1)
            # result += np.sum((pred == labels.numpy()))
            result += np.sum((pred == truth))
            num += len(images)
    acc = result / num
    return acc


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

        label = emo_tensor[self.path[item].split('/')[6]]
        return face_tensor, label

    def __len__(self):
        return len(self.path)


def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    lossData = {
        'epoch':[],
        'loss':[]
    }
    accData = {
        'epoch':[],
        'train':[],
        'val':[]
    }
    path = './dense169-plus/'
    if not os.path.exists(path):
        # 如果目录不存在，则创建目录
        os.makedirs(path)

    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    # model = FaceCNN().to(device)
    # model = VGG('VGG19-M').to(device)
    model = DenseNet169().to(device)
    # model = ResNet101().to(device)
    # model = torch.load('./vgg_res/model_it60.pkl', map_location=device)
    # model = torch.load('./vgg_res/_it100.pkl').to(device)

    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        model.train()  # 模型训练
        for images, labels in tqdm(train_loader, desc=f'epoch:{epoch+1}'):
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
            scheduler.step()  # 学习率衰减

        # 打印每轮的损失
        print('After {} epochs , the loss_rate is : '.format(epoch + 1), loss_rate.item())
        lossData['epoch'].append(epoch+1)
        lossData['loss'].append(loss_rate.item())
        if (epoch+1) % 10 == 0 and epoch != 0:
            model.eval()  # 模型评估
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('After {} epochs , the acc_train is : '.format(epoch + 1), acc_train)
            print('After {} epochs , the acc_val is : '.format(epoch + 1), acc_val)
            accData['epoch'].append(epoch+1)
            accData['train'].append(acc_train)
            accData['val'].append(acc_val)
        else:
            acc_val = validate(model, val_dataset, batch_size)
            accData['epoch'].append(epoch+1)
            accData['train'].append(-1.)
            accData['val'].append(acc_val)
        pd.DataFrame(lossData).to_csv(path + 'loss.csv')
        pd.DataFrame(accData).to_csv(path + 'acc.csv')
        if (epoch+1) % 10 == 0 and epoch != 0:
            torch.save(model, path + f'model_it{epoch+1}.pkl')
    pd.DataFrame(lossData).to_csv(path + 'loss.csv')
    pd.DataFrame(accData).to_csv(path + 'acc.csv')
    return model


def main():
    # 数据集实例化(创建数据集)
    train_dataset = FaceDataset('./trainplus.txt')
    val_dataset = FaceDataset('./testplus.txt')
    # train_dataset = AsianDataset('./train-asian.txt')
    # val_dataset = AsianDataset('./test-asian.txt')
    # 超参数可自行指定
    model = train(train_dataset, val_dataset, batch_size=128, epochs=80, learning_rate=0.1, wt_decay=0.0001)
    # 保存模型
    # torch.save(model, './asian_net.pkl')


if __name__ == '__main__':
    main()
