import sys

import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
import h5py
from scipy.misc import imread, imsave, imresize
from PIL import Image
import scipy.io as sci
from utils_torch import downgrade_images

Image.MAX_IMAGE_PIXELS = None
import cv2
import matplotlib.pyplot as plt


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", '.tif', '.mat'])


def load_tif(filepath):
    img = imread(filepath)
    # img = np.expand_dims(img,0)
    return img


def load_extif(filepath):
    img = imread(filepath)
    img = np.expand_dims(img, 0)
    return img


def load_mat(filepath):
    img = sci.loadmat(filepath)['i']
    return img


def augment(img_in, img_tar, img_bic, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, img_bic, info_aug



class DatasetFromFullHdf5(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(DatasetFromFullHdf5, self).__init__()
        dataset = h5py.File(image_dir, 'r')
        gt_orig = dataset.get('gt')
        pan_orig = dataset.get('pan')
        self.gt = np.transpose(gt_orig, [0, 3, 1, 2])
        self.PAN = np.expand_dims(pan_orig, 1)
        self.transform = transform

    def __getitem__(self, index):
        return torch.from_numpy(self.gt[index, :, 0:66, 0:66] / 255.0).float(), \
               torch.from_numpy(self.PAN[index, :, 0:264, 0:264] / 255.0).float()

    def __len__(self):
        return self.gt.shape[0]


#######QB
#############GF2
class DatasetFromHdf5(data.Dataset):
    def __init__(self, gt_dir, panl_dir, ms_dir, pan_dir, transform=None):
        super(DatasetFromHdf5, self).__init__()
        self.gt1 = [join(gt_dir+'/1', x) for x in listdir(gt_dir+'/1') if is_image_file(x)]
        self.ms_filenames1 = [join(ms_dir+'/1', x) for x in listdir(ms_dir+'/1') if is_image_file(x)]
        self.gt2 = [join(gt_dir + '/2', x) for x in listdir(gt_dir + '/2') if is_image_file(x)]
        self.ms_filenames2 = [join(ms_dir + '/2', x) for x in listdir(ms_dir + '/2') if is_image_file(x)]
        self.gt3 = [join(gt_dir + '/3', x) for x in listdir(gt_dir + '/3') if is_image_file(x)]
        self.ms_filenames3 = [join(ms_dir + '/3', x) for x in listdir(ms_dir + '/3') if is_image_file(x)]
        self.gt4 = [join(gt_dir + '/4', x) for x in listdir(gt_dir + '/4') if is_image_file(x)]
        self.ms_filenames4 = [join(ms_dir + '/4', x) for x in listdir(ms_dir + '/4') if is_image_file(x)]

        # for i in range(8):
        self.PAN = [join(pan_dir, x) for x in listdir(pan_dir) if is_image_file(x)]
        self.PAN_label = [join(panl_dir, x) for x in listdir(panl_dir) if is_image_file(x)]
        self.transform = transform

    def __getitem__(self, index):
        ms = np.concatenate((load_extif(self.ms_filenames1[index]), load_extif(self.ms_filenames2[index]),
                             load_extif(self.ms_filenames3[index]), load_extif(self.ms_filenames4[index])), 0)
        gt = np.concatenate((load_extif(self.gt1[index]), load_extif(self.gt2[index]),
                             load_extif(self.gt3[index]), load_extif(self.gt4[index])), 0)

        pan = load_extif(self.PAN[index])
        pan_label = load_extif(self.PAN_label[index])

        return torch.from_numpy(gt/255.0).float(), torch.from_numpy(pan_label/255.0).float(), \
               torch.from_numpy(ms/255.0).float(), torch.from_numpy(pan/255.0).float(), self.gt1[index]

    def __len__(self):
        return len(self.gt1)
#
class DatasetFromTestHdf5(data.Dataset):
    def __init__(self, gt_dir, panl_dir, ms_dir, pan_dir, transform=None):
        super(DatasetFromTestHdf5, self).__init__()
        self.PAN = [join(pan_dir, x) for x in listdir(pan_dir) if is_image_file(x)]
        self.PAN_label = [join(panl_dir, x) for x in listdir(panl_dir) if is_image_file(x)]
        self.gt = [join(gt_dir, x) for x in listdir(gt_dir) if is_image_file(x)]
        self.ms = [join(ms_dir, x) for x in listdir(ms_dir) if is_image_file(x)]
        self.transform = transform

    def __getitem__(self, index):
        pan = load_extif(self.PAN[index])
        pan_label = load_extif(self.PAN_label[index])
        ms = load_tif(self.ms[index])
        ms = np.transpose(ms, (2, 0, 1))
        gt = load_tif(self.gt[index])
        gt = np.transpose(gt, (2, 0, 1))

        return torch.from_numpy(gt/255.0).float(), torch.from_numpy(pan_label/255.0).float(), \
               torch.from_numpy(ms/255.0).float(), torch.from_numpy(pan/255.0).float(), self.gt[index]

    def __len__(self):
        return len(self.gt)



####################################WV2
# class DatasetFromHdf5(data.Dataset):
#     def __init__(self, gt_dir, panl_dir, ms_dir, pan_dir, transform=None):
#         super(DatasetFromHdf5, self).__init__()
#         self.gt1 = [join(gt_dir+'/1', x) for x in listdir(gt_dir+'/1') if is_image_file(x)]
#         self.ms_filenames1 = [join(ms_dir+'/1', x) for x in listdir(ms_dir+'/1') if is_image_file(x)]
#         self.gt2 = [join(gt_dir + '/2', x) for x in listdir(gt_dir + '/2') if is_image_file(x)]
#         self.ms_filenames2 = [join(ms_dir + '/2', x) for x in listdir(ms_dir + '/2') if is_image_file(x)]
#         self.gt3 = [join(gt_dir + '/3', x) for x in listdir(gt_dir + '/3') if is_image_file(x)]
#         self.ms_filenames3 = [join(ms_dir + '/3', x) for x in listdir(ms_dir + '/3') if is_image_file(x)]
#         self.gt4 = [join(gt_dir + '/4', x) for x in listdir(gt_dir + '/4') if is_image_file(x)]
#         self.ms_filenames4 = [join(ms_dir + '/4', x) for x in listdir(ms_dir + '/4') if is_image_file(x)]
#         self.gt5 = [join(gt_dir + '/5', x) for x in listdir(gt_dir + '/5') if is_image_file(x)]
#         self.ms_filenames5 = [join(ms_dir + '/5', x) for x in listdir(ms_dir + '/5') if is_image_file(x)]
#         self.gt6 = [join(gt_dir + '/6', x) for x in listdir(gt_dir + '/6') if is_image_file(x)]
#         self.ms_filenames6 = [join(ms_dir + '/6', x) for x in listdir(ms_dir + '/6') if is_image_file(x)]
#         self.gt7 = [join(gt_dir + '/7', x) for x in listdir(gt_dir + '/7') if is_image_file(x)]
#         self.ms_filenames7 = [join(ms_dir + '/7', x) for x in listdir(ms_dir + '/7') if is_image_file(x)]
#         self.gt8 = [join(gt_dir + '/8', x) for x in listdir(gt_dir + '/8') if is_image_file(x)]
#         self.ms_filenames8 = [join(ms_dir + '/8', x) for x in listdir(ms_dir + '/8') if is_image_file(x)]
#
#         # for i in range(8):
#         self.PAN = [join(pan_dir, x) for x in listdir(pan_dir) if is_image_file(x)]
#         self.PAN_label = [join(panl_dir, x) for x in listdir(panl_dir) if is_image_file(x)]
#         self.transform = transform
#
#     def __getitem__(self, index):
#         # print(index)
#         # print(len(self.ms_filenames1))
#         # print(self.ms_filenames1[index])
#         # print(load_extif(self.ms_filenames1[index]).shape)
#         ms = np.concatenate((load_extif(self.ms_filenames1[index]), load_extif(self.ms_filenames2[index]),
#                              load_extif(self.ms_filenames3[index]), load_extif(self.ms_filenames4[index]),
#                              load_extif(self.ms_filenames5[index]), load_extif(self.ms_filenames6[index]),
#                              load_extif(self.ms_filenames7[index]), load_extif(self.ms_filenames8[index])), 0)
#         gt = np.concatenate((load_extif(self.gt1[index]), load_extif(self.gt2[index]),
#                              load_extif(self.gt3[index]), load_extif(self.gt4[index]),
#                              load_extif(self.gt5[index]), load_extif(self.gt6[index]),
#                              load_extif(self.gt7[index]), load_extif(self.gt8[index])), 0)
#
#         pan = load_extif(self.PAN[index])
#         pan_label = load_extif(self.PAN_label[index])
#         # print(self.gt1[index])
#         # a=self.gt1[index][50:-4]
#         a = self.gt1[index][60:-4]
#         # print(a)
#         # sys.exit(0)
#         # ms, pan=downgrade_images(np.expand_dims(gt, 0), np.expand_dims(pan_label, 0), 4, 'WV2')
#         # imsave('/data/gmq/pansharpening/MKD/data/TrainFolder_rand/ms_blur/1/'+str(a)+'.tif', ms[0, 0, :, :])
#         # imsave('/data/gmq/pansharpening/MKD/data/TrainFolder_rand/ms_blur/2/' + str(a) + '.tif', ms[0, 1, :, :])
#         # imsave('/data/gmq/pansharpening/MKD/data/TrainFolder_rand/ms_blur/3/' + str(a) + '.tif', ms[0, 2, :, :])
#         # imsave('/data/gmq/pansharpening/MKD/data/TrainFolder_rand/ms_blur/4/' + str(a) + '.tif', ms[0, 3, :, :])
#         # imsave('/data/gmq/pansharpening/MKD/data/TrainFolder_rand/ms_blur/5/' + str(a) + '.tif', ms[0, 4, :, :])
#         # imsave('/data/gmq/pansharpening/MKD/data/TrainFolder_rand/ms_blur/6/' + str(a) + '.tif', ms[0, 5, :, :])
#         # imsave('/data/gmq/pansharpening/MKD/data/TrainFolder_rand/ms_blur/7/' + str(a) + '.tif', ms[0, 6, :, :])
#         # imsave('/data/gmq/pansharpening/MKD/data/TrainFolder_rand/ms_blur/8/' + str(a) + '.tif', ms[0, 7, :, :])
#         # imsave('/data/gmq/pansharpening/MKD/data/TrainFolder_rand/pan_blur/' + str(a) + '.tif', pan[0, 0, :, :])
#
#         # imsave('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/up_mslabel/1/'+str(a)+'.tif',
#         #        imresize(load_tif(self.gt1[index]), size=4.0, interp='bicubic'))
#         # imsave('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/up_mslabel/2/'+str(a)+'.tif',
#         #        imresize(load_tif(self.gt2[index]), size=4.0, interp='bicubic'))
#         # imsave('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/up_mslabel/3/'+str(a)+'.tif',
#         #        imresize(load_tif(self.gt3[index]), size=4.0, interp='bicubic'))
#         # imsave('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/up_mslabel/4/'+str(a)+'.tif',
#         #        imresize(load_tif(self.gt4[index]), size=4.0, interp='bicubic'))
#         # imsave('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/up_mslabel/5/'+str(a)+'.tif',
#         #        imresize(load_tif(self.gt5[index]), size=4.0, interp='bicubic'))
#         # imsave('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/up_mslabel/6/'+str(a)+'.tif',
#         #        imresize(load_tif(self.gt6[index]), size=4.0, interp='bicubic'))
#         # imsave('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/up_mslabel/7/'+str(a)+'.tif',
#         #        imresize(load_tif(self.gt7[index]), size=4.0, interp='bicubic'))
#         # imsave('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/up_mslabel/8/'+str(a)+'.tif',
#         #        imresize(load_tif(self.gt8[index]), size=4.0, interp='bicubic'))
#         # imsave('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/down_pan/'+str(a)+'.tif',
#         #        imresize(load_tif(self.PAN[index]), size=0.25, interp='bicubic'))
#         # print(ms.dtype)
#         # print(ms.shape)
#         # print(np.max(ms))
#         # print(np.min(ms))
#         # sys.exit(0)
#         return torch.from_numpy(gt/255.0).float(), torch.from_numpy(pan_label/255.0).float(), \
#                torch.from_numpy(ms/255.0).float(), torch.from_numpy(pan/255.0).float(), self.gt1[index]
#
#     def __len__(self):
#         return len(self.gt1)
#
# class DatasetFromFolderEval(data.Dataset):
#     def __init__(self, gt_dir, ms_dir, pan_dir, transform=None):
#         super(DatasetFromFolderEval, self).__init__()
#         # self.gt_filenames = [join(gt_dir, x) for x in listdir(gt_dir) if is_image_file(x)]
#         # self.ms_filenames = [join(ms_dir, x) for x in listdir(ms_dir) if is_image_file(x)]
#         self.pan_filenames = [join(pan_dir, x) for x in listdir(pan_dir) if is_image_file(x)]
#         self.gt1 = [join(gt_dir+'/1', x) for x in listdir(gt_dir+'/1') if is_image_file(x)]
#         self.ms_filenames1 = [join(ms_dir+'/1', x) for x in listdir(ms_dir+'/1') if is_image_file(x)]
#         self.gt2 = [join(gt_dir + '/2', x) for x in listdir(gt_dir + '/2') if is_image_file(x)]
#         self.ms_filenames2 = [join(ms_dir + '/2', x) for x in listdir(ms_dir + '/2') if is_image_file(x)]
#         self.gt3 = [join(gt_dir + '/3', x) for x in listdir(gt_dir + '/3') if is_image_file(x)]
#         self.ms_filenames3 = [join(ms_dir + '/3', x) for x in listdir(ms_dir + '/3') if is_image_file(x)]
#         self.gt4 = [join(gt_dir + '/4', x) for x in listdir(gt_dir + '/4') if is_image_file(x)]
#         self.ms_filenames4 = [join(ms_dir + '/4', x) for x in listdir(ms_dir + '/4') if is_image_file(x)]
#         self.gt5 = [join(gt_dir + '/5', x) for x in listdir(gt_dir + '/5') if is_image_file(x)]
#         self.ms_filenames5 = [join(ms_dir + '/5', x) for x in listdir(ms_dir + '/5') if is_image_file(x)]
#         self.gt6 = [join(gt_dir + '/6', x) for x in listdir(gt_dir + '/6') if is_image_file(x)]
#         self.ms_filenames6 = [join(ms_dir + '/6', x) for x in listdir(ms_dir + '/6') if is_image_file(x)]
#         self.gt7 = [join(gt_dir + '/7', x) for x in listdir(gt_dir + '/7') if is_image_file(x)]
#         self.ms_filenames7 = [join(ms_dir + '/7', x) for x in listdir(ms_dir + '/7') if is_image_file(x)]
#         self.gt8 = [join(gt_dir + '/8', x) for x in listdir(gt_dir + '/8') if is_image_file(x)]
#         self.ms_filenames8 = [join(ms_dir + '/8', x) for x in listdir(ms_dir + '/8') if is_image_file(x)]
#         self.transform = transform
#
#     def __getitem__(self, index):
#         ms = np.concatenate((load_extif(self.ms_filenames1[index]), load_extif(self.ms_filenames2[index]),
#                              load_extif(self.ms_filenames3[index]), load_extif(self.ms_filenames4[index]),
#                              load_extif(self.ms_filenames5[index]), load_extif(self.ms_filenames6[index]),
#                              load_extif(self.ms_filenames7[index]), load_extif(self.ms_filenames8[index])),0)
#         gt = np.concatenate((load_extif(self.gt1[index]), load_extif(self.gt2[index]),
#                              load_extif(self.gt3[index]), load_extif(self.gt4[index]),
#                              load_extif(self.gt5[index]), load_extif(self.gt6[index]),
#                              load_extif(self.gt7[index]), load_extif(self.gt8[index])), 0)
#         pan = load_extif(self.pan_filenames[index])
#
#         return torch.from_numpy(gt/255.0).float(), torch.from_numpy(ms/255.0).float(),\
#                torch.from_numpy(pan/255.0).float(), self.gt1[index]
#
#     def __len__(self):
#         return len(self.gt1)
#
class DatasetFromFullEval(data.Dataset):
    def __init__(self, ms_dir, pan_dir, transform=None):
        super(DatasetFromFullEval, self).__init__()
        self.ms_filenames1 = [join(ms_dir+'/1', x) for x in listdir(ms_dir+'/1') if is_image_file(x)]
        self.ms_filenames2 = [join(ms_dir + '/2', x) for x in listdir(ms_dir + '/2') if is_image_file(x)]
        self.ms_filenames3 = [join(ms_dir + '/3', x) for x in listdir(ms_dir + '/3') if is_image_file(x)]
        self.ms_filenames4 = [join(ms_dir + '/4', x) for x in listdir(ms_dir + '/4') if is_image_file(x)]
        self.ms_filenames5 = [join(ms_dir + '/5', x) for x in listdir(ms_dir + '/5') if is_image_file(x)]
        self.ms_filenames6 = [join(ms_dir + '/6', x) for x in listdir(ms_dir + '/6') if is_image_file(x)]
        self.ms_filenames7 = [join(ms_dir + '/7', x) for x in listdir(ms_dir + '/7') if is_image_file(x)]
        self.ms_filenames8 = [join(ms_dir + '/8', x) for x in listdir(ms_dir + '/8') if is_image_file(x)]
        # self.ms_filenames = [join(ms_dir, x) for x in listdir(ms_dir) if is_image_file(x)]
        self.pan_filenames = [join(pan_dir, x) for x in listdir(pan_dir) if is_image_file(x)]
        self.transform = transform

    def __getitem__(self, index):
        ms = np.concatenate((load_tif(self.ms_filenames1[index]), load_tif(self.ms_filenames2[index]),
                             load_tif(self.ms_filenames3[index]), load_tif(self.ms_filenames4[index]),
                             load_tif(self.ms_filenames5[index]), load_tif(self.ms_filenames6[index]),
                             load_tif(self.ms_filenames7[index]), load_tif(self.ms_filenames8[index])), 0)
        pan = load_tif(self.pan_filenames[index])
        return torch.from_numpy(ms/255.0).float(), \
               torch.from_numpy(pan/255.0).float(), self.ms_filenames1[index]

    def __len__(self):
        return len(self.ms_filenames1)
