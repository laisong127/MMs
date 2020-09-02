import random

import torch
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np


def imageaug(img_label):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 左右翻转
        iaa.Flipud(0.5),  # 上下翻转
        iaa.Sometimes(0.3, iaa.Affine(
            rotate=(-10, 10),  # 旋转一定角度
            shear=(-10, 10),  # 拉伸一定角度（矩形变为平行四边形状）
            order=0,  # order=[0, 1],   #使用最邻近差值或者双线性差值
            cval=0,  # cval=(0, 255),  #全白全黑填充
            mode='constant'  # mode=ia.ALL  #定义填充图像外区域的方法
        )),
        # iaa.Crop(percent=(0, 0.1)),
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0, 10.0), sigma=(4.0, 6.0)))  # 把像素移动到周围的地方
    ])
    label = img_label[1]
    imglab_aug = seq.augment_images(img_label)
    imglab_aug = imglab_aug.transpose((3,0,1,2))
    img_aug = imglab_aug[0]
    img_aug_shape = img_aug.shape
    img_aug = img_aug.reshape(img_aug_shape[0],1,img_aug_shape[1],img_aug_shape[2])
    lab_aug = imglab_aug[1]
    lab_aug_shape = lab_aug.shape
    lab_aug = lab_aug.reshape(lab_aug_shape[0],1,lab_aug_shape[1],lab_aug_shape[2])
    # print(img_aug.shape)
    lab_aug = np.clip(np.round(lab_aug), np.min(label), np.max(label))
    return img_aug, lab_aug


if __name__ == '__main__':
    img = plt.imread('/home/laisong/cvlab/trainCvlab/img/train001.png')
    label = plt.imread('/home/laisong/cvlab/trainCvlab/lab/train001.png')
    img_reshape = np.reshape(img, (img.shape[0], img.shape[1], 1))
    label_reshape = np.reshape(label, (label.shape[0], label.shape[1], 1))
    imglab = np.concatenate((img_reshape, label_reshape), axis=2)
    print(imglab.shape)
    img_aug, lab_aug = imageaug(imglab)

    print(np.max(lab_aug))

    plt.figure()
    plt.subplot(2, 2, 3)
    plt.imshow(img)
    plt.subplot(2, 2, 4)
    plt.imshow(label)
    plt.subplot(2, 2, 1)
    plt.title('aug0')
    plt.imshow(img_aug)
    plt.subplot(2, 2, 2)
    plt.title('aug1')
    plt.imshow(lab_aug)
    plt.show()
