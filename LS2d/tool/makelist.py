import os

def makedatalist(imgpath, listpath):
    num = 0
    file_path = imgpath
    path_list = os.listdir(file_path)
    for file_name in path_list:
        path = imgpath + file_name+'/'
        if os.path.isdir(path):
            num = num + 1
            path_list.extend(os.listdir(path))
        else:
            break

    del path_list[:num]
    path_list.sort()

    with open(listpath, 'a') as f:
        f.seek(0)
        f.truncate()
        for file_name in path_list:
            f.write(file_name + '\n')

    f.close()

imagepath = r'/home/laisong/MRI2IMG/TRAIN_IMG(A)'
listpath = r'/home/laisong/MRI2IMG//train_img.txt'
makedatalist(imagepath,listpath)
imagepath = r'/home/laisong/MRI2IMG/TRAIN_LABEL(A)'
listpath = r'/home/laisong/MRI2IMG//tain_label.txt'
makedatalist(imagepath,listpath)
imagepath = r'/home/laisong/MRI2IMG/VAL_IMG(A)'
listpath = r'/home/laisong/MRI2IMG//val_img.txt'
makedatalist(imagepath,listpath)
imagepath = r'/home/laisong/MRI2IMG/VAL_LABEL(A)'
listpath = r'/home/laisong/MRI2IMG//val_label.txt'
makedatalist(imagepath,listpath)
# print('%.1f' % 0.96)