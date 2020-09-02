from torch.autograd import Variable
import nibabel as nib
import numpy as np
from PIL import Image

from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

from MRI2IMG_dataset import LiverDataset, val_imagepath, val_labelpath, valimg_ids, vallabel_ids, train_imagepath, \
    train_labelpath, trainimg_ids, trainlabel_ids
from loss import DiceLoss
from metrics import dice_coeff
import torch
from Newunet import Insensee_3Dunet
from criterions import softmax_dice_loss
import MRIdataset
from torch.utils.data import DataLoader
from HSC82 import CleanU_Net
from advanced_model import DeepSupervision_U_Net
from ResNetUNet import ResNetUNet
from loss import make_one_hot
from metrics import dice_coeff

from criterions import sigmoid_dice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def saveresult2d(test_loader, model):
    i = 0
    for img, label, _, labelfile in test_loader:
        if i < 50:
            label3d = []
            img = torch.squeeze(img)
            label = torch.squeeze(label)
            for z in range(img.shape[0]):
                img_2d = img[z, :, :]
                label_2d = label[z, :, :]
                img_2d = torch.unsqueeze(img_2d, 0)
                img_2d = torch.unsqueeze(img_2d, 0)
                img_2d = img_2d.to(device)
                output = model(img_2d)
                # print(output.shape)
                pred = torch.argmax(output, 1)
                # print(pred.shape, label.shape)
                pred = pred.cpu()
                # print(type(pred))
                # label3d.append(pred)
                if z == 0:
                    label3d = pred
                else:
                    label3d = torch.cat((label3d, pred), dim=0)
            print(label3d.shape)
        label = label3d.numpy().astype(float)
        label1 = label.transpose(1, 2, 0)

        new_image = nib.Nifti1Image(label1, np.eye(4))
        nib.save(new_image, '../saveresult/pred_{}'.format(labelfile[0]))

        i += 1


def test_model():
    MAX = 0
    save = 0
    model = CleanU_Net(1,4).cuda(0)
    model.load_state_dict(torch.load('/home/laisong/github/MMs/LS2d/3dunet_model_save/weights_149.pth'))
    liver_dataset = LiverDataset(val_imagepath, val_labelpath, valimg_ids, vallabel_ids)
    val_dataloader = DataLoader(liver_dataset, batch_size=1, shuffle=False, num_workers=4)

    train_dataset = LiverDataset(train_imagepath, train_labelpath, trainimg_ids, trainlabel_ids)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

    # train_iter = enumerate(train_loader)
    model.eval()
    LV_Dice = 0
    LV_Jac = 0
    RV_Dice = 0
    RV_Jac = 0
    Myo_Dice = 0
    Myo_Jac = 0
    i = 0


    for img, label, _, _ in val_dataloader:
        img_val_tensor = img
        label_val_tensor = label * 3
        inputs = img_val_tensor.float().to(device)
        labels = label_val_tensor.long().to(device)
        outputs = model(inputs)
        # print(outputs.shape)
        pred = torch.argmax(outputs, 1)
        # print(pred.shape, labels.shape)
        # LV_dice, LV_jac, RV_dice, RV_jac, Myo_dice, Myo_jac = dice_coeff(pred,labels)
        # print(pred.shape)

        pred = pred.unsqueeze(0)
        pred_onehot = make_one_hot(pred,4).cpu()
        labels = labels.cpu()
        LV_dice, RV_dice, Myo_dice,_ = sigmoid_dice(pred_onehot,labels)

        print('LV_dice:{:.4f} | RV_dice:{:.4f} | Myo_dice:{:.4f}'.format(LV_dice, RV_dice, Myo_dice))
        LV_Dice += LV_dice
        RV_Dice += RV_dice
        Myo_Dice += Myo_dice

    print('===============================================')
    print('LV_Dice_avg:', LV_Dice / i, 'RV_Dice_avg:', RV_Dice / i, 'Myo_Dice_avg:', Myo_Dice / i)
    # if (LV_Dice/i+RV_Dice/i+Myo_Dice/i) > MAX:
    #     MAX = LV_Dice/i+RV_Dice/i+Myo_Dice/i
    #     save = index


if __name__ == '__main__':
    # model = CleanU_Net(1, 4).to(device)
    # model.load_state_dict(torch.load('./3dunet_model_save/weights_200.pth'))
    # test_dataset = MRIdataset.LiverDataset(MRIdataset.test_imagepath, MRIdataset.test_labelpath,
    #                                        MRIdataset.testimg_ids, MRIdataset.testlabel_ids, False)
    #
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    # saveresult2d(test_loader, model)
    test_model()

    # img = nib.load(r'/home/peng/Desktop/CROP/pred/pred_2.nii.gz').get_data()
    # img = img.flatten()
    # for i in range(len(img)):
    #     if img[i]!=0:
    #         print(img[i])
