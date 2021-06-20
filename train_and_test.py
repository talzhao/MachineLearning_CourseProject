# coding: utf-8

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset

from our_model import Unet
from trainers import Trainer

import warnings
from PIL import Image
import os

Path = os.path.abspath(os.path.dirname(__file__))


class Segmentation_Dataset(Dataset):

    def __init__(self, path_img=None, path_label=None, transforms=None):
        self.train = None
        self.labels = None
        self.transforms = transforms

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        img = self.train[idx]
        label = self.labels[idx]

        if self.transforms:
            img = self.transforms(img)
            target = self.transforms(label)

        return img, target


## get the dataset
def get_numpy(path, num, augment=False):
    for i in range(num):
        tmp_path_img = path + str(i) + '.png'
        tmp_train = Image.open(tmp_path_img)
        if augment == False:
            # without augmentation
            tmp_train_array = np.array(tmp_train)
            tmp_train_array = np.expand_dims(tmp_train_array, axis = 0)
            tmp_train_array = np.expand_dims(tmp_train_array, axis = -1)

            if i == 0 :
                train_pic = tmp_train_array
            else:
                train_pic = np.concatenate((train_pic, tmp_train_array), axis=0)
        else:
            # with augmentation
            for j in range(2):
                # tmp_train.save(Path + "\\train" + str(i) + "before.png")
                tmp_train_rotate= tmp_train.rotate(180 * j)
                # tmp_train_rotate.save(Path + "\\train" + str(i) + "after.png")

                tmp_train_array = np.array(tmp_train_rotate)
                tmp_train_array = np.expand_dims(tmp_train_array, axis = 0)
                tmp_train_array = np.expand_dims(tmp_train_array, axis = -1)

                if i == 0 and j == 0:
                    train_pic = tmp_train_array
                else:
                    train_pic = np.concatenate((train_pic, tmp_train_array), axis=0)
    #         print (train_pic.shape)
    return train_pic


# store the prediction
def store_pred(num, isbi_test):
    img, target = isbi_test[num]
    y_pred = trainer.predict(img)
    thresh = 0.5
    y_pred[y_pred >= thresh] = 1
    y_pred[y_pred < thresh] = 0


    test_pred = y_pred.reshape(512, 512)
    test_pred = np.cast['uint8'](test_pred * 255)
    im = Image.fromarray(test_pred)

    im.save(Path + "\\result" + str(num) + ".png")

if __name__ == "__main__":
    path_train_img = Path + '\dataset\\train_img\\'
    path_train_label = Path + '\dataset\\train_label\\'
    path_test_img = Path + '\dataset\\test_img\\'
    path_test_label = Path + '\dataset\\test_label\\'
    train_img = get_numpy(path_train_img, 25, augment=True)
    train_label = get_numpy(path_train_label, 25, augment=True)
    test_img = get_numpy(path_test_img, 5, augment=False)
    test_label = get_numpy(path_test_label, 5, augment=False)

    print(train_label.shape)
    print(train_img.shape)
    print(test_label.shape)
    print(test_img.shape)

    warnings.filterwarnings("ignore")
    transform = transforms.Compose([transforms.ToTensor()])

    ## load_the data
    train_Dataset = Segmentation_Dataset(None , None , transforms=transform)
    test_Dataset = Segmentation_Dataset(None, None, transforms=transform)
    ## load_the data
    train_Dataset.train = train_img
    train_Dataset.labels = train_label
    test_Dataset.train = test_img
    test_Dataset.labels = test_label

    ## train the network
    unet = Unet()
    unet.cuda()
    trainer = Trainer(unet)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(trainer.model.parameters(), lr=1e-3)
    loss_history = trainer.fit_generator(train_Dataset, criterion, optimizer,  n_epochs=25, batch_size = 1)
    print(loss_history)

    ## output the result
    for i in range(5):
        store_pred(i, test_Dataset)
