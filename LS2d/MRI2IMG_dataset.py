import torch
import torch.utils.data as data
import os
import numpy as np
import PIL.Image as Image
import nibabel as nib
from scipy.ndimage import rotate
from torch.utils.data import DataLoader
from PIL import Image
from tool.makelist import makedatalist
from torchvision import transforms

train_imagepath = r'/home/laisong/MRI2IMG/TRAIN_IMG(A)'
train_labelpath = r'/home/laisong/MRI2IMG/TRAIN_LABEL(A)'
imgid = r'/home/laisong/MRI2IMG/train_img.txt'
labelid = r'/home/laisong/MRI2IMG/train_label.txt'
trainimg_ids = [i_id.strip() for i_id in open(imgid)]
trainlabel_ids = [i_id.strip() for i_id in open(labelid)]

val_imagepath = r'/home/laisong/MRI2IMG/VAL_IMG(A)'
val_labelpath = r'/home/laisong/MRI2IMG/VAL_LABEL(A)'
imgid = r'/home/laisong/MRI2IMG/val_img.txt'
labelid = r'/home/laisong/MRI2IMG/val_label.txt'
valimg_ids = [i_id.strip() for i_id in open(imgid)]
vallabel_ids = [i_id.strip() for i_id in open(labelid)]

def normalization(data):
    data = np.asarray(data).squeeze()
    max, min = np.max(data), np.min(data)
    data = (data - min) / (max - min)
    return data


class LiverDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, imagepath, labelpath, imgids, labelids):
        n = len(os.listdir(imagepath))  # os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整
        imgdirs = os.listdir(imagepath)
        labeldirs = os.listdir(labelpath)
        imgs = []
        labels = []
        imgfile = []
        labelfile = []

        for j, imfile in enumerate(imgdirs):
            image = Image.open(imagepath+'/'+imfile)
            image_1 = np.asarray(image, np.float32)
            imgs.append(image_1)


        for j, lafile in enumerate(labeldirs):
            label = Image.open(labelpath+'/'+lafile)
            label_1 = np.asarray(label, np.float32)
            label_2 = label_1/255
            labels.append(label_2)

        self.imgs = imgs
        self.labels = labels
        self.imgfile = imgids
        self.labelfile = labelids

    def __getitem__(self, index):
        imgfile = self.imgfile[index]
        labelfile = self.labelfile[index]
        image = self.imgs[index]
        image_normal = normalization(image)  # normalization
        image_addchanel = image_normal[np.newaxis,...]

        label = self.labels[index]

        return image_addchanel, label, imgfile, labelfile  # tensor 0~1

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]


if __name__ == '__main__':
    liver_dataset = LiverDataset(val_imagepath, val_labelpath, valimg_ids, vallabel_ids)
    dataloader = DataLoader(liver_dataset, batch_size=64, shuffle=False, num_workers=4)
    print(len(dataloader))

    for img, lab, imgfile, labelfile in dataloader:
        print(img.shape,lab.shape)
        # label_one = lab[5].numpy()*3
        # label_as_tensor = torch.from_numpy(label_one)
        # print(torch.max(label_as_tensor,0))



