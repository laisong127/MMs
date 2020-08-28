import torch
import numpy as np
from numpy import long
from torch.optim import lr_scheduler
from torchvision.transforms import transforms as T
import argparse  # argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
from Newunet import Insensee_3Dunet
from torch import optim
import MRIdataset
from torch.utils.data import DataLoader
from advanced_model import DeepSupervision_U_Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 5e-4


def train_model(model, criterion, optimizer, dataload, num_epochs=200):
    # model.load_state_dict(torch.load('./3dunet_model_save/weights_199.pth'))
    for epoch in range(num_epochs):
        save_loss = []
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('learning_rate:', optimizer.state_dict()['param_groups'][0]['lr'])
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for x, y,_,_ in dataload:  # 分100次遍历数据集，每次遍历batch_size=4
            x = torch.squeeze(x)
            y = torch.squeeze(y)
            loss = 0
            for z in range(x.shape[0]):
                img_2d = x[z, :, :]
                img_2d = torch.unsqueeze(img_2d, 0)
                img_2d = torch.unsqueeze(img_2d, 0)

                label_2d = y[z, :, :]
                label_2d = torch.unsqueeze(label_2d, 0)

                optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
                loss_2d = 0
                inputs = img_2d.float().to(device)
                labels = label_2d.long().to(device)
                outputs = model(inputs)  # 前向传播

                # print(outputs.shape, labels.shape)
                loss_2d = criterion(outputs, labels)
                # print(loss)

                loss_2d.backward()  # 梯度下降,计算出梯度
                optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
                loss += loss_2d.item()
            epoch_loss += loss
            save_loss.append(loss)
            step += 1
            print("%d/%d,train_loss:%0.6f" % (step, dataset_size // dataload.batch_size, loss))
        np.savetxt('./3dunet_model_save/loss_%d.txt' % epoch, save_loss)
        print("epoch %d loss:%0.6f" % (epoch, epoch_loss))

        if epoch % 100 == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.8
        if epoch % 10 == 0:
            torch.save(model.state_dict(), './3dunet_model_save/weights_%d.pth' % epoch)  # 返回模型的所有内容

    return model


def train():
    model = DeepSupervision_U_Net(in_channels=1, out_channels=4).to(device)
    batch_size = 1
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss().cuda(0)
    # 梯度下降
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    scheduler = lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.98)
    liver_dataset = MRIdataset.LiverDataset(MRIdataset.imagepath, MRIdataset.labelpath, MRIdataset.img_ids,
                                            MRIdataset.label_ids, False)
    dataloader = DataLoader(liver_dataset, batch_size=1, shuffle=True, num_workers=4)
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
    train_model(model, criterion, optimizer, dataloader)


if __name__ == '__main__':
    train()