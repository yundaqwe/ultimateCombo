# encoding:utf-8
"""Implementation of sample attack."""

import kornia as K
from torchtoolbox.transform import Cutout, ImageNetPolicy, CIFAR10Policy
import imgaug.augmenters as iaa
import torchvision
from torchvision.transforms import *
from torchvision.prototype import transforms as T2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from torch.autograd import Variable as V
import math
# from torch.autograd.gradcheck import zero_gradients
from torch.utils import data
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from ipdb import set_trace
from PIL import Image, ImageFilter, ImageGrab
from torchvision import transforms
from utils import regularizer, rand_bbox
from colourspace import group_pca_color_augmention
import time
import datetime
from torch.utils.data import random_split
from ipdb import set_trace
import torch

import logging


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device="cpu"):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def patchShuffle(images, device, k=4, p=0.05):
    batches, c, h, w = images.size()
    # output_tensor = torch.zeros(batch_size, out_channels, output_height, output_width)

    for b in range(batches):
        for i in range(0, h - 4, k):
            for j in range(0, w - 4, k):
                # set_trace()
                if random.random() < p:
                    images[b, :, i:i + k, j:j + k] = shuffleKernel(images[b, :, i:i + k, j:j + k], device=device, k=k)
    return images


def shuffleKernel(tensor, device, k=4):
    # Generate random permutation indices for the rows and columns

    row_perm_index = torch.randperm(k).cuda(device)
    col_perm_index = torch.randperm(k).cuda(device)
    # version2
    tensor_shuffled_dim1 = torch.index_select(tensor, 1, row_perm_index)
    tensor_shuffled_dim2 = torch.index_select(tensor_shuffled_dim1, 2, col_perm_index)
    return tensor_shuffled_dim2
    # print(tensor_shuffled_dim2)


def zero_gradients(x):
    if x.grad is not None:
        x.grad.zero_()

    return x


def rounddown(number):
    return int(number * 10000) / 10000


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def mkdir(path):
    """Check if the folder exists, if it does not exist, create it"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def save_img(images, filenames, output_dir):
    """save high quality jpeg"""
    mkdir(output_dir)

    for i, filename in enumerate(filenames):
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(os.path.join(output_dir, filename), quality=100)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()

    x2 = x
    x3 = x
    x4 = x
    for i in range(x.shape[0]):
        tem = x2[i][1]
        x2[i][1] = x2[i][2]
        x2[i][2] = tem
    return x2


def input_diversity(X, p=0.5, image_width=299, image_resize=330):
    # AW optimize: change random to start instead of end
    if torch.rand(()) >= p:
        return X
    # if opt.dataset=='cifar-10':
    #     image_width =32
    #     image_resize = 35
    rnd = torch.randint(image_width, image_resize, ())
    rescaled = nn.functional.interpolate(X, [rnd, rnd])
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [image_width, image_width])
    # return padded if torch.rand(()) < p else X

    return padded


def admix(x, size=3):
    portion = 0.2
    # size=3 #mixup
    return torch.cat(tuple([(x + portion * x[torch.randperm(x.size(0))]) for _ in range(size)]), axis=0) / (
                1 + portion * size)


def Edge_Enhance(x, device):
    kernel = torch.unsqueeze(torch.tensor([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]]), 0)
    kernel = torch.unsqueeze(kernel, dim=0).cuda(device)
    kernel = torch.repeat_interleave(kernel, 3, dim=0)

    return F.conv2d(x, kernel, padding='same', groups=3)


def mycutout(img, device, p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False,
             inplace=False):
    if random.random() < p:
        # if True:

        batch, img_c, img_h, img_w = img.shape
        s = random.uniform(*scale)
        s = s * img_h * img_w
        r = random.uniform(*ratio)
        w = int(math.sqrt(s / r))
        h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)
        c = torch.tensor(0).to(device)

        for i in range(batch):
            img[i, :, left:left + w, top:top + h] = c
        # ##save_img(img, [str(i) + "cutout.jpeg" for i in range(img.shape[0])], opt.output_dir)
        return img
    else:
        return img


def cutmix(img, device, p=0.5):
    if random.random() < p:
        rand_index = torch.randperm(img.shape[0]).cuda(device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
        x = img

        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        return x
    else:
        return img


def AdvancedUltimateAdmixFGSM(model, img, img_n, label, using_aux_logit, order, opt, device):
    # SAFE BUT slow!!!
    logging.basicConfig(filename='log.txt', level=logging.INFO)
    print("order:")
    print(order)

    order = str(bin(order))[2:]
    extrazero = 48 - len(order)
    order = extrazero * '0' + order

    eps = opt.max_epsilon / 255.0

    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    batch, channel, H, W = X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])
    gradBetweenIters = []

    for i in range(num_iter):
        zero_gradients(noise)

        augmented = []
        augmented_SI = []
        augmented.append(X_pert + noise)
        ##save_img(X_pert + noise, ["0"+ "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

        # set_trace()
        for index, whethertouse in enumerate(order):
            possibility = 1
            if whethertouse == '1':
                if index == 1:
                    # pass
                    # TODO
                    x_cm = cutmix(X_pert + noise, p=possibility, device=device)
                    augmented.append(x_cm)
                    ##save_img(x_cm, [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")
                elif index == 2:
                    cropSize = 229
                    targetSize = 299

                    # 随机选择填充大小的范围，可根据实际情况设置
                    w_pad = targetSize - cropSize
                    h_pad = targetSize - cropSize

                    # 随机填充
                    left = torch.randint(0, w_pad + 1, (1,))
                    top = torch.randint(0, h_pad + 1, (1,))

                    transform = RandomCrop(size=cropSize)
                    x_RC = transform(X_pert + noise)

                    X_p = F.pad(x_RC, (left, w_pad - left, top, h_pad - top))
                    ##save_img(X_p ,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")
                    augmented.append(X_p)
                    # augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
                elif index == 3:
                    degree = random.randint(1, 359)
                    transform = RandomRotation(degrees=(-1 * degree, degree))
                    x_RR = transform(X_pert + noise)
                    ##save_img(x_RR ,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RR)
                    # x_neural = img_n.clone()
                    # augmented.append(x_neural + noise)
                elif index == 4:
                    transform = transforms.Grayscale(num_output_channels=3)
                    x_GS = transform(X_pert + noise)
                    ##save_img(x_GS,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_GS)
                    # augmented.append(Edge_Enhance(X_pert + noise))
                elif index == 5:
                    transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                    x_CJ = transform(X_pert + noise)
                    ##save_img(x_CJ,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_CJ)
                elif index == 6:
                    transform = transforms.GaussianBlur(kernel_size=(7, 7))
                    x_GB = transform(X_pert + noise)
                    ##save_img(x_GB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_GB)
                elif index == 7:

                    x_GB = channel_shuffle(X_pert + noise)
                    ##save_img(x_GB,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_GB)
                elif index == 8:

                    transform = transforms.RandomAffine(degrees=(-30, 30), translate=(0.2, 0.2),
                                                        shear=(-30, 30))  # given by gpt
                    x_RA = transform(X_pert + noise)
                    ##save_img(x_RA,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RA)
                elif index == 9:

                    transform = transforms.RandomPerspective(p=possibility)
                    x_RP = transform(X_pert + noise)
                    ##save_img(x_RP,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RP)
                elif index == 10:
                    x1, x2, x3 = group_pca_color_augmention(X_pert)
                    ##save_img(x1,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x1)
                elif index == 11:
                    # set_trace()
                    # pass
                    transform = T.ElasticTransform(alpha=200.0)
                    x_ET = transform(X_pert + noise)
                    ##save_img(x_ET,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_ET)
                elif index == 12:
                    transform = transforms.RandomHorizontalFlip(p=possibility)
                    x_RHP = transform(X_pert + noise)
                    ##save_img(x_RHP,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RHP)
                elif index == 13:
                    transform = transforms.RandomVerticalFlip(p=possibility)

                    x_RVP = transform(X_pert + noise)
                    ##save_img(x_RVP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RVP)
                elif index == 14:

                    augmented.append(admix(X_pert + noise, 1))
                elif index == 15:
                    transform = transforms.RandomInvert(p=possibility)
                    x_RI = transform(X_pert + noise)
                    ##save_img(x_RI,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RI)
                elif index == 16:

                    transform = transforms.RandomPosterize(p=possibility, bits=6)

                    x_RP2 = transform((255 * (X_pert + noise)).to(torch.uint8)) / 255
                    ##save_img(x_RP2,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RP2)
                elif index == 17:
                    transform = transforms.RandomSolarize(p=possibility, threshold=0.5)
                    x_RS = transform(X_pert + noise)
                    ##save_img(x_RS,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RS)
                elif index == 18:
                    transform = transforms.RandomEqualize(p=possibility)
                    x_RE = transform((255 * (X_pert + noise)).to(torch.uint8)) / 255
                    ##save_img(x_RE,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RE)
                elif index == 19:
                    transform = transforms.RandomErasing(p=possibility, scale=(0.02, 0.2), ratio=(0.3, 3.3),
                                                         value='random')
                    x_RE2 = transform(X_pert + noise)
                    ##save_img(x_RE2,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RE2)
                elif index == 20:
                    transform = transforms.Compose([
                        ImageNetPolicy,
                        transforms.ToTensor()
                    ])
                    pil_trans = ToPILImage()
                    x_AA = torch.zeros_like(img)
                    for i in range(img.shape[0]):
                        x_AA[i] = transform(pil_trans(X_pert[i])).cuda(device) + noise[i]

                    ##save_img(x_AA,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_AA)

                elif index == 21:
                    x_neural = img_n.clone()
                    ##save_img(x_neural,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_neural + noise)
                elif index == 22:
                    X_EE = Edge_Enhance(X_pert + noise, device)
                    ##save_img(X_EE,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(X_EE)
                elif index == 23:
                    X_CO = mycutout(X_pert + noise, p=possibility, ratio=(1, 1), value=(0, 1), device=device)
                    ##save_img(X_CO,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(X_CO)

                elif index == 24:

                    transform = iaa.JpegCompression(compression=(70, 99))
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_JC = transform(images=x_nd)
                    X_JC = torch.tensor(np.transpose(X_JC, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_JC, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testJC")

                    augmented.append(X_JC + noise)
                    # transform=K.enhance.ZCAWhitening()
                    # X_ZCA =transform(X_pert + noise,include_fit=True)
                    # #save_img(X_ZCA, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")
                    # augmented.append(X_ZCA)
                elif index == 25:

                    # rescale to 0-255 and transfer to np.ndarray
                    transform = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_DO = transform(images=x_nd)
                    X_DO = torch.tensor(np.transpose(X_DO, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_DO, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(X_DO + noise)

                elif index == 26:
                    transform = iaa.color.KMeansColorQuantization(n_colors=(8, 16), max_size=299)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_KCQ = transform(images=x_nd)
                    X_KCQ = torch.tensor(np.transpose(X_KCQ, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_KCQ, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(X_KCQ + noise)
                elif index == 27:
                    transform = iaa.segmentation.RelativeRegularGridVoronoi(max_size=299)  # to be determined

                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_RRGV = transform(images=x_nd)
                    X_RRGV = torch.tensor(np.transpose(X_RRGV, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_RRGV, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./RRGV")

                    augmented.append(X_RRGV + noise)
                    # pass
                elif index == 28:
                    transform = iaa.Superpixels(max_size=299)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_SP = transform(images=x_nd)
                    X_SP = torch.tensor(np.transpose(X_SP, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_SP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./SP")

                    augmented.append(X_SP + noise)
                elif index == 29:
                    transform = iaa.AverageBlur(k=(2, 7))  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_AB = transform(images=x_nd)
                    X_AB = torch.tensor(np.transpose(X_AB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_AB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./SP")

                    augmented.append(X_AB + noise)
                elif index == 30:
                    transform = iaa.MedianBlur(k=(3, 7))  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MB = transform(images=x_nd)
                    X_MB = torch.tensor(np.transpose(X_MB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./SP")

                    augmented.append(X_MB + noise)
                elif index == 31:
                    transform = iaa.BilateralBlur(d=(2, 7))  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_BB = transform(images=x_nd)
                    X_BB = torch.tensor(np.transpose(X_BB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_BB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./BB")

                    augmented.append(X_BB + noise)
                elif index == 32:
                    transform = iaa.MotionBlur()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MB2 = transform(images=x_nd)
                    X_MB2 = torch.tensor(np.transpose(X_MB2, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MB2, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MB2")

                    augmented.append(X_MB2 + noise)
                elif index == 33:
                    transform = iaa.MeanShiftBlur()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MSB = transform(images=x_nd)
                    X_MSB = torch.tensor(np.transpose(X_MSB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MSB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MSB")

                    augmented.append(X_MSB + noise)
                elif index == 34:
                    transform = iaa.Emboss()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_EB = transform(images=x_nd)
                    X_EB = torch.tensor(np.transpose(X_EB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_EB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./EB")

                    augmented.append(X_EB + noise)
                elif index == 35:
                    transform = iaa.EdgeDetect()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_ED = transform(images=x_nd)
                    X_ED = torch.tensor(np.transpose(X_ED, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_ED, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./ED")

                    augmented.append(X_ED + noise)
                elif index == 36:
                    transform = iaa.Canny()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_C = transform(images=x_nd)
                    X_C = torch.tensor(np.transpose(X_C, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_C, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./canny")

                    augmented.append(X_C + noise)
                elif index == 37:
                    transform = iaa.AveragePooling(keep_size=True)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_AP = transform(images=x_nd)
                    X_AP = torch.tensor(np.transpose(X_AP, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_AP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./AP")

                    augmented.append(X_AP + noise)
                elif index == 38:
                    transform = iaa.MaxPooling(keep_size=True)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MP = transform(images=x_nd)
                    X_MP = torch.tensor(np.transpose(X_MP, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MP")

                    augmented.append(X_MP + noise)
                elif index == 39:
                    transform = iaa.MinPooling(keep_size=True)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MP2 = transform(images=x_nd)
                    X_MP2 = torch.tensor(np.transpose(X_MP2, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MP2, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MP2")

                    augmented.append(X_MP2 + noise)
                elif index == 40:
                    transform = iaa.MedianPooling(keep_size=True)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MP3 = transform(images=x_nd)
                    X_MP3 = torch.tensor(np.transpose(X_MP3, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MP3, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MP3")

                    augmented.append(X_MP3 + noise)
                elif index == 41:
                    X_PS = patchShuffle(X_pert + noise, device=device)
                    # transform = iaa.Rain() # to be determined
                    # x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    # X_R= transform(images=x_nd)
                    # X_R = torch.tensor(np.transpose(X_R, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_PS, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./X_PS")
                    augmented.append(X_PS)

                elif index == 42:
                    transform = iaa.Clouds()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_CD = transform(images=x_nd)
                    X_CD = torch.tensor(np.transpose(X_CD, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_CD, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./CD")

                    augmented.append(X_CD + noise)
                elif index == 43:
                    transform = iaa.Fog()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_FG = transform(images=x_nd)
                    X_FG = torch.tensor(np.transpose(X_FG, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_FG, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./FG")

                    augmented.append(X_FG + noise)

                elif index == 44:
                    transform = iaa.imgcorruptlike.Frost()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_FT = transform(images=x_nd)
                    X_FT = torch.tensor(np.transpose(X_FT, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_FT, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./FT")

                    augmented.append(X_FT + noise)
                elif index == 45:
                    transform = iaa.imgcorruptlike.Snow()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_S = transform(images=x_nd)
                    X_S = torch.tensor(np.transpose(X_S, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_S, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./snow")

                    augmented.append(X_S + noise)
                elif index == 46:
                    transform = iaa.Rain()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_R = transform(images=x_nd)
                    X_R = torch.tensor(np.transpose(X_R, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_R, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./rain")

                    augmented.append(X_R + noise)
                elif index == 47:
                    transform = AddGaussianNoise(std=15 / 255, device=device)  # to be determined
                    X_AGN = transform(X_pert + noise)
                    # save_img(X_AGN, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./X_AGN")

                    augmented.append(X_AGN)

        label_2 = label  # torch.cat(tuple([label] * len(augmented)))
        # augmented = torch.cat(tuple(augmented))

        if order[0] == '1':
            for item in augmented:
                augmented_SI.append((item))
                augmented_SI.append((item / 2))
                augmented_SI.append((item / 4))
                augmented_SI.append((item / 8))
                augmented_SI.append((item / 16))
            # augmented_SI = torch.cat(tuple(augmented_SI))
            label_3 = label_2  # torch.cat(tuple([label_2] * 5))
            augmented_images_si_di = []

            for augmented_image in augmented_SI:
                augmented_image_si_di = input_diversity(augmented_image)
                augmented_images_si_di.append(augmented_image_si_di)
            augmented_SI = augmented_images_si_di  # torch.cat(tuple(augmented_images_si_di))

        else:
            augmented_SI = augmented
            # augmented_SI = torch.cat(tuple(augmented_SI))
            label_3 = label_2  # torch.cat(tuple([label_2] * 1))
        grad = 0
        for i, each_x in enumerate(augmented_SI):
            # noise.detach_()
            # noise=noise.data
            # noise= V(noise.data, requires_grad=True)
            # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
            output = model(each_x.float())
            if len(output[0].size()) == 1:
                for image_i in range(len(output)):
                    if opt.parallel:
                        output[image_i] = output[image_i].reshape(-1, 1001)
                    else:
                        output[image_i] = output[image_i].unsqueeze(0)

            loss = F.cross_entropy(output[0], label_3, reduction='sum')  # logit

            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label_3, reduction='sum')  # aux_logit
            loss.backward(retain_graph=True)
            grad = grad + noise.grad.data
            zero_gradients(noise)


        if order[0] == '1':
            # translation invariant
            grad = grad.reshape([1, batch * channel, H, W])
            grad = nn.functional.conv2d(grad, stack_kernel, padding='same', groups=channel * batch)
            grad = grad.reshape([batch, channel, H, W])

        # momentum
        grad = grad / torch.abs(grad).mean([1, 2, 3], keepdim=True)
        grad = momentum * old_grad + grad
        old_grad = grad

        noise = noise + alpha * torch.sign(grad)
        # Avoid out of bound
        noise = torch.clamp(noise, -eps, eps)
        x = img + noise
        x = torch.clamp(x, 0.0, 1.0)
        noise = x - img

        noise = V(noise, requires_grad=True)


    adv = img + noise.detach()
    # TODO:assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0A6000
    # TODO:assert(np.all(np.logical_and(adv-img>-eps, adv-img<eps)))
    assert rounddown(adv.max().item()) <= 1 and rounddown(adv.min().item()) >= 0
    assert rounddown((adv - img).min().item()) >= -eps and rounddown((adv - img).max()) <= eps

    batch, channel, H, W = img.shape
    if opt.cosine:
        grad = grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None


def ens_AdvancedUltimateAdmixFGSM(modelnames, img, img_n, label, order, opt, device, models):
    # 1/m \sum_{x_i} \nabla_x{} loss(x_i)
    logging.basicConfig(filename=f'{opt.log}.txt', level=logging.INFO)
    print("order:")
    print(order)

    order = str(bin(order))[2:]
    extrazero = 48 - len(order)
    order = extrazero * '0' + order

    eps = opt.max_epsilon / 255.0

    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    batch, channel, H, W = X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])

    for i in range(num_iter):
        zero_gradients(noise)

        augmented = []
        augmented_SI = []
        augmented.append(X_pert + noise)
        ##save_img(X_pert + noise, ["0"+ "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

        # set_trace()
        for index, whethertouse in enumerate(order):
            possibility = 1
            if whethertouse == '1':
                if index == 1:
                    # pass
                    # TODO
                    x_cm = cutmix(X_pert + noise, p=possibility, device=device)
                    augmented.append(x_cm)
                    ##save_img(x_cm, [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")
                elif index == 2:
                    cropSize = 229
                    targetSize = 299

                    # 随机选择填充大小的范围，可根据实际情况设置
                    w_pad = targetSize - cropSize
                    h_pad = targetSize - cropSize

                    # 随机填充
                    left = torch.randint(0, w_pad + 1, (1,))
                    top = torch.randint(0, h_pad + 1, (1,))

                    transform = RandomCrop(size=cropSize)
                    x_RC = transform(X_pert + noise)

                    X_p = F.pad(x_RC, (left, w_pad - left, top, h_pad - top))
                    ##save_img(X_p ,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")
                    augmented.append(X_p)
                    # augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
                elif index == 3:
                    degree = random.randint(1, 359)
                    transform = RandomRotation(degrees=(-1 * degree, degree))
                    x_RR = transform(X_pert + noise)
                    ##save_img(x_RR ,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RR)
                    # x_neural = img_n.clone()
                    # augmented.append(x_neural + noise)
                elif index == 4:
                    transform = transforms.Grayscale(num_output_channels=3)
                    x_GS = transform(X_pert + noise)
                    ##save_img(x_GS,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_GS)
                    # augmented.append(Edge_Enhance(X_pert + noise))
                elif index == 5:
                    transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                    x_CJ = transform(X_pert + noise)
                    ##save_img(x_CJ,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_CJ)
                elif index == 6:
                    transform = transforms.GaussianBlur(kernel_size=(7, 7))
                    x_GB = transform(X_pert + noise)
                    ##save_img(x_GB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_GB)
                elif index == 7:

                    x_GB = channel_shuffle(X_pert + noise)
                    ##save_img(x_GB,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_GB)
                elif index == 8:

                    transform = transforms.RandomAffine(degrees=(-30, 30), translate=(0.2, 0.2),
                                                        shear=(-30, 30))  # given by gpt
                    x_RA = transform(X_pert + noise)
                    ##save_img(x_RA,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RA)
                elif index == 9:

                    transform = transforms.RandomPerspective(p=possibility)
                    x_RP = transform(X_pert + noise)
                    ##save_img(x_RP,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RP)
                elif index == 10:
                    x1, x2, x3 = group_pca_color_augmention(X_pert)
                    ##save_img(x1,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x1)
                elif index == 11:
                    # set_trace()
                    # pass
                    transform = T.ElasticTransform(alpha=200.0)
                    x_ET = transform(X_pert + noise)
                    ##save_img(x_ET,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_ET)
                elif index == 12:
                    transform = transforms.RandomHorizontalFlip(p=possibility)
                    x_RHP = transform(X_pert + noise)
                    ##save_img(x_RHP,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RHP)
                elif index == 13:
                    transform = transforms.RandomVerticalFlip(p=possibility)

                    x_RVP = transform(X_pert + noise)
                    ##save_img(x_RVP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RVP)
                elif index == 14:

                    augmented.append(admix(X_pert + noise, 1))
                elif index == 15:
                    transform = transforms.RandomInvert(p=possibility)
                    x_RI = transform(X_pert + noise)
                    ##save_img(x_RI,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RI)
                elif index == 16:

                    transform = transforms.RandomPosterize(p=possibility, bits=6)

                    x_RP2 = transform((255 * (X_pert + noise)).to(torch.uint8)) / 255
                    ##save_img(x_RP2,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RP2)
                elif index == 17:
                    transform = transforms.RandomSolarize(p=possibility, threshold=0.5)
                    x_RS = transform(X_pert + noise)
                    ##save_img(x_RS,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RS)
                elif index == 18:
                    transform = transforms.RandomEqualize(p=possibility)
                    x_RE = transform((255 * (X_pert + noise)).to(torch.uint8)) / 255
                    ##save_img(x_RE,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RE)
                elif index == 19:
                    transform = transforms.RandomErasing(p=possibility, scale=(0.02, 0.2), ratio=(0.3, 3.3),
                                                         value='random')
                    x_RE2 = transform(X_pert + noise)
                    ##save_img(x_RE2,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_RE2)
                elif index == 20:
                    transform = transforms.Compose([
                        ImageNetPolicy,
                        transforms.ToTensor()
                    ])
                    pil_trans = ToPILImage()
                    x_AA = torch.zeros_like(img)
                    for i in range(img.shape[0]):
                        x_AA[i] = transform(pil_trans(X_pert[i])).cuda(device) + noise[i]

                    ##save_img(x_AA,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_AA)

                elif index == 21:
                    x_neural = img_n.clone()
                    ##save_img(x_neural,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(x_neural + noise)
                elif index == 22:
                    X_EE = Edge_Enhance(X_pert + noise, device)
                    ##save_img(X_EE,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(X_EE)
                elif index == 23:
                    X_CO = mycutout(X_pert + noise, p=possibility, ratio=(1, 1), value=(0, 1), device=device)
                    ##save_img(X_CO,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(X_CO)

                elif index == 24:

                    transform = iaa.JpegCompression(compression=(70, 99))
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_JC = transform(images=x_nd)
                    X_JC = torch.tensor(np.transpose(X_JC, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_JC, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testJC")

                    augmented.append(X_JC + noise)
                    # transform=K.enhance.ZCAWhitening()
                    # X_ZCA =transform(X_pert + noise,include_fit=True)
                    # #save_img(X_ZCA, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")
                    # augmented.append(X_ZCA)
                elif index == 25:

                    # rescale to 0-255 and transfer to np.ndarray
                    transform = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_DO = transform(images=x_nd)
                    X_DO = torch.tensor(np.transpose(X_DO, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_DO, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(X_DO + noise)

                elif index == 26:
                    transform = iaa.color.KMeansColorQuantization(n_colors=(8, 16), max_size=299)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_KCQ = transform(images=x_nd)
                    X_KCQ = torch.tensor(np.transpose(X_KCQ, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_KCQ, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                    augmented.append(X_KCQ + noise)
                elif index == 27:
                    transform = iaa.segmentation.RelativeRegularGridVoronoi(max_size=299)  # to be determined

                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_RRGV = transform(images=x_nd)
                    X_RRGV = torch.tensor(np.transpose(X_RRGV, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_RRGV, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./RRGV")

                    augmented.append(X_RRGV + noise)
                    # pass
                elif index == 28:
                    transform = iaa.Superpixels(max_size=299)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_SP = transform(images=x_nd)
                    X_SP = torch.tensor(np.transpose(X_SP, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_SP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./SP")

                    augmented.append(X_SP + noise)
                elif index == 29:
                    transform = iaa.AverageBlur(k=(2, 7))  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_AB = transform(images=x_nd)
                    X_AB = torch.tensor(np.transpose(X_AB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_AB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./SP")

                    augmented.append(X_AB + noise)
                elif index == 30:
                    transform = iaa.MedianBlur(k=(3, 7))  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MB = transform(images=x_nd)
                    X_MB = torch.tensor(np.transpose(X_MB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./SP")

                    augmented.append(X_MB + noise)
                elif index == 31:
                    transform = iaa.BilateralBlur(d=(2, 7))  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_BB = transform(images=x_nd)
                    X_BB = torch.tensor(np.transpose(X_BB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_BB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./BB")

                    augmented.append(X_BB + noise)
                elif index == 32:
                    transform = iaa.MotionBlur()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MB2 = transform(images=x_nd)
                    X_MB2 = torch.tensor(np.transpose(X_MB2, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MB2, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MB2")

                    augmented.append(X_MB2 + noise)
                elif index == 33:
                    transform = iaa.MeanShiftBlur()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MSB = transform(images=x_nd)
                    X_MSB = torch.tensor(np.transpose(X_MSB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MSB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MSB")

                    augmented.append(X_MSB + noise)
                elif index == 34:
                    transform = iaa.Emboss()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_EB = transform(images=x_nd)
                    X_EB = torch.tensor(np.transpose(X_EB, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_EB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./EB")

                    augmented.append(X_EB + noise)
                elif index == 35:
                    transform = iaa.EdgeDetect()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_ED = transform(images=x_nd)
                    X_ED = torch.tensor(np.transpose(X_ED, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_ED, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./ED")

                    augmented.append(X_ED + noise)
                elif index == 36:
                    transform = iaa.Canny()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_C = transform(images=x_nd)
                    X_C = torch.tensor(np.transpose(X_C, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_C, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./canny")

                    augmented.append(X_C + noise)
                elif index == 37:
                    transform = iaa.AveragePooling(keep_size=True)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_AP = transform(images=x_nd)
                    X_AP = torch.tensor(np.transpose(X_AP, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_AP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./AP")

                    augmented.append(X_AP + noise)
                elif index == 38:
                    transform = iaa.MaxPooling(keep_size=True)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MP = transform(images=x_nd)
                    X_MP = torch.tensor(np.transpose(X_MP, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MP")

                    augmented.append(X_MP + noise)
                elif index == 39:
                    transform = iaa.MinPooling(keep_size=True)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MP2 = transform(images=x_nd)
                    X_MP2 = torch.tensor(np.transpose(X_MP2, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MP2, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MP2")

                    augmented.append(X_MP2 + noise)
                elif index == 40:
                    transform = iaa.MedianPooling(keep_size=True)  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_MP3 = transform(images=x_nd)
                    X_MP3 = torch.tensor(np.transpose(X_MP3, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_MP3, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MP3")

                    augmented.append(X_MP3 + noise)
                elif index == 41:
                    X_PS = patchShuffle(X_pert + noise, device=device)
                    # transform = iaa.Rain() # to be determined
                    # x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    # X_R= transform(images=x_nd)
                    # X_R = torch.tensor(np.transpose(X_R, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_PS, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./X_PS")
                    augmented.append(X_PS)

                elif index == 42:
                    transform = iaa.Clouds()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_CD = transform(images=x_nd)
                    X_CD = torch.tensor(np.transpose(X_CD, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_CD, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./CD")

                    augmented.append(X_CD + noise)
                elif index == 43:
                    transform = iaa.Fog()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_FG = transform(images=x_nd)
                    X_FG = torch.tensor(np.transpose(X_FG, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_FG, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./FG")

                    augmented.append(X_FG + noise)

                elif index == 44:
                    transform = iaa.imgcorruptlike.Frost()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_FT = transform(images=x_nd)
                    X_FT = torch.tensor(np.transpose(X_FT, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_FT, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./FT")

                    augmented.append(X_FT + noise)
                elif index == 45:
                    transform = iaa.imgcorruptlike.Snow()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_S = transform(images=x_nd)
                    X_S = torch.tensor(np.transpose(X_S, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_S, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./snow")

                    augmented.append(X_S + noise)
                elif index == 46:
                    transform = iaa.Rain()  # to be determined
                    x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                    X_R = transform(images=x_nd)
                    X_R = torch.tensor(np.transpose(X_R, (0, 3, 1, 2)) / 255, device=device)
                    # save_img(X_R, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./rain")

                    augmented.append(X_R + noise)
                elif index == 47:
                    transform = AddGaussianNoise(std=15 / 255, device=device)  # to be determined
                    X_AGN = transform(X_pert + noise)
                    # save_img(X_AGN, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./X_AGN")

                    augmented.append(X_AGN)

        label_2 = label  # torch.cat(tuple([label] * len(augmented)))
        # augmented = torch.cat(tuple(augmented))

        if order[0] == '1':
            for item in augmented:
                augmented_SI.append((item))
                augmented_SI.append((item / 2))
                augmented_SI.append((item / 4))
                augmented_SI.append((item / 8))
                augmented_SI.append((item / 16))
            # augmented_SI = torch.cat(tuple(augmented_SI))
            label_3 = label_2  # torch.cat(tuple([label_2] * 5))
            augmented_images_si_di = []

            for augmented_image in augmented_SI:
                augmented_image_si_di = input_diversity(augmented_image)
                augmented_images_si_di.append(augmented_image_si_di)
            augmented_SI = augmented_images_si_di  # torch.cat(tuple(augmented_images_si_di))

        else:
            augmented_SI = augmented
            # augmented_SI = torch.cat(tuple(augmented_SI))
            label_3 = label_2  # torch.cat(tuple([label_2] * 1))
        grad = 0

        for i, each_x in enumerate(augmented_SI):
            logit = 0
            aux_logit = 0
            aux_logit_count = 0
            for model_name in modelnames:
                model = models[model_name]
                output = model(each_x.float())
                logit += output[0]

                if not 'resnet' in model_name:
                    aux_logit_count += 1
                    aux_logit += output[1]

            # output = model(each_x.float())
            print(f"size of augmented images: {len(each_x)}")
            if len(output[0].size()) == 1:
                for image_i in range(len(output)):
                    if opt.parallel:
                        output[image_i] = output[image_i].reshape(-1, 1001)
                    else:
                        output[image_i] = output[image_i].unsqueeze(0)

            loss = F.cross_entropy(logit, label_3, reduction='sum')  # logit

            if aux_logit_count > 0:
                loss = loss + F.cross_entropy(aux_logit, label_3, reduction='sum')  # aux_logit
            loss.backward(retain_graph=True)
            grad = grad + noise.grad.data
            zero_gradients(noise)

        if order[0] == '1':
            # translation invariant
            grad = grad.reshape([1, batch * channel, H, W])
            grad = nn.functional.conv2d(grad, stack_kernel, padding='same', groups=channel * batch)
            grad = grad.reshape([batch, channel, H, W])

        # momentum
        grad = grad / torch.abs(grad).mean([1, 2, 3], keepdim=True)
        grad = momentum * old_grad + grad
        old_grad = grad

        noise = noise + alpha * torch.sign(grad)
        # Avoid out of bound
        noise = torch.clamp(noise, -eps, eps)
        x = img + noise
        x = torch.clamp(x, 0.0, 1.0)
        noise = x - img

        noise = V(noise, requires_grad=True)

    adv = img + noise.detach()
    # TODO:assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0A6000
    # TODO:assert(np.all(np.logical_and(adv-img>-eps, adv-img<eps)))
    assert rounddown(adv.max().item()) <= 1 and rounddown(adv.min().item()) >= 0
    assert rounddown((adv - img).min().item()) >= -eps and rounddown((adv - img).max()) <= eps

    return adv






def CIFARAdvancedUltimateAdmixFGSM(model, img, img_n, label, using_aux_logit, order, opt, device):
    #SAFE BUT slow!!!
     logging.basicConfig(filename='log.txt', level=logging.INFO)
     print("order:")
     print(order)

     order = str(bin(order))[2:]
     extrazero = 48 - len(order)
     order = extrazero * '0' + order

     eps = opt.max_epsilon

     num_iter = opt.num_iter
     if opt.lr:
         alpha = opt.lr
     else:
         alpha = eps / num_iter



     momentum = 1  # set in the original paper
     grad = 0
     X_pert = img.clone()
     noise = torch.zeros_like(img, requires_grad=True)
     old_grad = torch.zeros_like(img)
     # label = torch.cat(tuple([label] * 3))
     batch, channel, H, W = X_pert.shape
     kernel = gkern(3, 3).astype(np.float32)
     stack_kernel = np.stack([kernel, kernel, kernel])
     stack_kernel = np.expand_dims(stack_kernel, 0)
     stack_kernel = torch.tensor(stack_kernel).cuda(device)
     stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
     iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
     stack_kernel = stack_kernel.transpose(0, 1)
     stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])
     gradBetweenIters=[]

     for i in range(num_iter):
         zero_gradients(noise)

         augmented = []
         augmented_SI = []
         augmented.append(X_pert + noise)
         ##save_img(X_pert + noise, ["0"+ "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

         # set_trace()
         for index, whethertouse in enumerate(order):
             possibility = 1
             if whethertouse == '1':
                 if index == 1:
                     # pass
                     # TODO
                     x_cm = cutmix(X_pert + noise, p=possibility, device=device)
                     augmented.append(x_cm)
                     ##save_img(x_cm, [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")
                 elif index == 2:
                     cropSize = 25
                     targetSize = 32

                     # 随机选择填充大小的范围，可根据实际情况设置
                     w_pad = targetSize - cropSize
                     h_pad = targetSize - cropSize

                     # 随机填充
                     left = torch.randint(0, w_pad + 1, (1,))
                     top = torch.randint(0, h_pad + 1, (1,))

                     transform = RandomCrop(size=cropSize)
                     x_RC = transform(X_pert + noise)

                     X_p = F.pad(x_RC, (left, w_pad - left, top, h_pad - top))
                     ##save_img(X_p ,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")
                     augmented.append(X_p)
                     # augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
                 elif index == 3:
                     degree = random.randint(1, 359)
                     transform = RandomRotation(degrees=(-1 * degree, degree))
                     x_RR = transform(X_pert + noise)
                     ##save_img(x_RR ,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RR)
                     # x_neural = img_n.clone()
                     # augmented.append(x_neural + noise)
                 elif index == 4:
                     transform = transforms.Grayscale(num_output_channels=3)
                     x_GS = transform(X_pert + noise)
                     ##save_img(x_GS,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_GS)
                     # augmented.append(Edge_Enhance(X_pert + noise))
                 elif index == 5:
                     transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                     x_CJ = transform(X_pert + noise)
                     ##save_img(x_CJ,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_CJ)
                 elif index == 6:
                     transform = transforms.GaussianBlur(kernel_size=(3, 3))
                     x_GB = transform(X_pert + noise)
                     ##save_img(x_GB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_GB)
                 elif index == 7:

                     x_GB = channel_shuffle(X_pert + noise)
                     ##save_img(x_GB,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_GB)
                 elif index == 8:

                     transform = transforms.RandomAffine(degrees=(-30, 30), translate=(0.2, 0.2),
                                                         shear=(-30, 30))  # given by gpt
                     x_RA = transform(X_pert + noise)
                     ##save_img(x_RA,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RA)
                 elif index == 9:

                     transform = transforms.RandomPerspective(p=possibility)
                     x_RP = transform(X_pert + noise)
                     ##save_img(x_RP,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RP)
                 elif index == 10:
                     x1, x2, x3 = group_pca_color_augmention(X_pert)
                     ##save_img(x1,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x1)
                 elif index == 11:
                     # set_trace()
                     # pass
                     transform = T.ElasticTransform(alpha=200.0)
                     x_ET = transform(X_pert + noise)
                     ##save_img(x_ET,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_ET)
                 elif index == 12:
                     transform = transforms.RandomHorizontalFlip(p=possibility)
                     x_RHP = transform(X_pert + noise)
                     ##save_img(x_RHP,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RHP)
                 elif index == 13:
                     transform = transforms.RandomVerticalFlip(p=possibility)

                     x_RVP = transform(X_pert + noise)
                     ##save_img(x_RVP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RVP)
                 elif index == 14:

                     augmented.append(admix(X_pert + noise, 1))
                 elif index == 15:
                     transform = transforms.RandomInvert(p=possibility)
                     x_RI = transform(X_pert + noise)
                     ##save_img(x_RI,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RI)
                 elif index == 16:

                     transform = transforms.RandomPosterize(p=possibility, bits=6)

                     x_RP2 = transform((255 * (X_pert + noise)).to(torch.uint8)) / 255
                     ##save_img(x_RP2,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RP2)
                 elif index == 17:
                     transform = transforms.RandomSolarize(p=possibility, threshold=0.5)
                     x_RS = transform(X_pert + noise)
                     ##save_img(x_RS,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RS)
                 elif index == 18:
                     transform = transforms.RandomEqualize(p=possibility)
                     x_RE = transform((255 * (X_pert + noise)).to(torch.uint8)) / 255
                     ##save_img(x_RE,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RE)
                 elif index == 19:
                     transform = transforms.RandomErasing(p=possibility, scale=(0.02, 0.2), ratio=(0.3, 3.3),
                                                          value='random')
                     x_RE2 = transform(X_pert + noise)
                     ##save_img(x_RE2,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_RE2)
                 elif index == 20:
                     transform = transforms.Compose([
                         CIFAR10Policy,
                         transforms.ToTensor()
                     ])
                     pil_trans = ToPILImage()
                     x_AA = torch.zeros_like(img)
                     for i in range(img.shape[0]):
                         x_AA[i] = transform(pil_trans(X_pert[i])).cuda(device) + noise[i]

                     ##save_img(x_AA,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(x_AA)

                 elif index == 21:
                     continue
                     # neural transfer is not feasible for cifar-10
                     # x_neural = img_n.clone()
                     ##save_img(x_neural,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     # augmented.append(x_neural + noise)
                 elif index == 22:
                     X_EE = Edge_Enhance(X_pert + noise, device)
                     ##save_img(X_EE,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(X_EE)
                 elif index == 23:
                     X_CO = mycutout(X_pert + noise, p=possibility, ratio=(1, 1), value=(0, 1), device=device)
                     ##save_img(X_CO,  [str(index)+"_"+str(i)+".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(X_CO)

                 elif index == 24:

                     transform = iaa.JpegCompression(compression=(70, 99))
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_JC = transform(images=x_nd)
                     X_JC = torch.tensor(np.transpose(X_JC, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_JC, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testJC")

                     augmented.append(X_JC + noise)
                     # transform=K.enhance.ZCAWhitening()
                     # X_ZCA =transform(X_pert + noise,include_fit=True)
                     # #save_img(X_ZCA, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")
                     # augmented.append(X_ZCA)
                 elif index == 25:

                     # rescale to 0-255 and transfer to np.ndarray
                     transform = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_DO = transform(images=x_nd)
                     X_DO = torch.tensor(np.transpose(X_DO, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_DO, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(X_DO + noise)

                 elif index == 26:
                     transform = iaa.color.KMeansColorQuantization(n_colors=(8, 16), max_size=32)  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_KCQ = transform(images=x_nd)
                     X_KCQ = torch.tensor(np.transpose(X_KCQ, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_KCQ, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./testGM")

                     augmented.append(X_KCQ + noise)
                 elif index == 27:
                     transform = iaa.segmentation.RelativeRegularGridVoronoi(max_size=32)  # to be determined

                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_RRGV = transform(images=x_nd)
                     X_RRGV = torch.tensor(np.transpose(X_RRGV, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_RRGV, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./RRGV")

                     augmented.append(X_RRGV + noise)
                     # pass
                 elif index == 28:
                     transform = iaa.Superpixels(max_size=32)  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_SP = transform(images=x_nd)
                     X_SP = torch.tensor(np.transpose(X_SP, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_SP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./SP")

                     augmented.append(X_SP + noise)
                 elif index == 29:
                     transform = iaa.AverageBlur(k=(2, 3))  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_AB = transform(images=x_nd)
                     X_AB = torch.tensor(np.transpose(X_AB, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_AB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./SP")

                     augmented.append(X_AB + noise)
                 elif index == 30:
                     transform = iaa.MedianBlur(k=(3, 3))  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_MB = transform(images=x_nd)
                     X_MB = torch.tensor(np.transpose(X_MB, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_MB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./SP")

                     augmented.append(X_MB + noise)
                 elif index == 31:
                     transform = iaa.BilateralBlur(d=(2, 3))  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_BB = transform(images=x_nd)
                     X_BB = torch.tensor(np.transpose(X_BB, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_BB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./BB")

                     augmented.append(X_BB + noise)
                 elif index == 32:
                     transform = iaa.MotionBlur()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_MB2 = transform(images=x_nd)
                     X_MB2 = torch.tensor(np.transpose(X_MB2, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_MB2, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MB2")

                     augmented.append(X_MB2 + noise)
                 elif index == 33:
                     transform = iaa.MeanShiftBlur()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_MSB = transform(images=x_nd)
                     X_MSB = torch.tensor(np.transpose(X_MSB, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_MSB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MSB")

                     augmented.append(X_MSB + noise)
                 elif index == 34:
                     transform = iaa.Emboss()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_EB = transform(images=x_nd)
                     X_EB = torch.tensor(np.transpose(X_EB, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_EB, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./EB")

                     augmented.append(X_EB + noise)
                 elif index == 35:
                     transform = iaa.EdgeDetect()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_ED = transform(images=x_nd)
                     X_ED = torch.tensor(np.transpose(X_ED, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_ED, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./ED")

                     augmented.append(X_ED + noise)
                 elif index == 36:
                     transform = iaa.Canny()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_C = transform(images=x_nd)
                     X_C = torch.tensor(np.transpose(X_C, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_C, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./canny")

                     augmented.append(X_C + noise)
                 elif index == 37:
                     transform = iaa.AveragePooling(keep_size=True)  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_AP = transform(images=x_nd)
                     X_AP = torch.tensor(np.transpose(X_AP, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_AP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./AP")

                     augmented.append(X_AP + noise)
                 elif index == 38:
                     transform = iaa.MaxPooling(keep_size=True)  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_MP = transform(images=x_nd)
                     X_MP = torch.tensor(np.transpose(X_MP, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_MP, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MP")

                     augmented.append(X_MP + noise)
                 elif index == 39:
                     transform = iaa.MinPooling(keep_size=True)  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_MP2 = transform(images=x_nd)
                     X_MP2 = torch.tensor(np.transpose(X_MP2, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_MP2, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MP2")

                     augmented.append(X_MP2 + noise)
                 elif index == 40:
                     transform = iaa.MedianPooling(keep_size=True)  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_MP3 = transform(images=x_nd)
                     X_MP3 = torch.tensor(np.transpose(X_MP3, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_MP3, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./MP3")

                     augmented.append(X_MP3 + noise)
                 elif index == 41:
                     X_PS = patchShuffle(X_pert + noise, device=device,k=2)
                     # transform = iaa.Rain() # to be determined
                     # x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     # X_R= transform(images=x_nd)
                     # X_R = torch.tensor(np.transpose(X_R, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_PS, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./X_PS")
                     augmented.append(X_PS)

                 elif index == 42:
                     transform = iaa.Clouds()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_CD = transform(images=x_nd)
                     X_CD = torch.tensor(np.transpose(X_CD, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_CD, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./CD")

                     augmented.append(X_CD + noise)
                 elif index == 43:
                     transform = iaa.Fog()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_FG = transform(images=x_nd)
                     X_FG = torch.tensor(np.transpose(X_FG, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_FG, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./FG")

                     augmented.append(X_FG + noise)

                 elif index == 44:
                     transform = iaa.imgcorruptlike.Frost()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_FT = transform(images=x_nd)
                     X_FT = torch.tensor(np.transpose(X_FT, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_FT, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./FT")

                     augmented.append(X_FT + noise)
                 elif index == 45:
                     transform = iaa.imgcorruptlike.Snow()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_S = transform(images=x_nd)
                     X_S = torch.tensor(np.transpose(X_S, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_S, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./snow")

                     augmented.append(X_S + noise)
                 elif index == 46:
                     transform = iaa.Rain()  # to be determined
                     x_nd = X_pert.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                     X_R = transform(images=x_nd)
                     X_R = torch.tensor(np.transpose(X_R, (0, 3, 1, 2)) / 255, device=device)
                     # save_img(X_R, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./rain")

                     augmented.append(X_R + noise)
                 elif index == 47:
                     transform = AddGaussianNoise(std=15 / 255, device=device)  # to be determined
                     X_AGN = transform(X_pert + noise)
                     # save_img(X_AGN, [str(index) + "_" + str(i) + ".jpeg" for i in range(X_pert.shape[0])], "./X_AGN")

                     augmented.append(X_AGN)




         label_2 = label  # torch.cat(tuple([label] * len(augmented)))
         # augmented = torch.cat(tuple(augmented))

         if order[0] == '1':
             for item in augmented:
                 augmented_SI.append((item))
                 augmented_SI.append((item / 2))
                 augmented_SI.append((item / 4))
                 augmented_SI.append((item / 8))
                 augmented_SI.append((item / 16))
             # augmented_SI = torch.cat(tuple(augmented_SI))
             label_3 = label_2  # torch.cat(tuple([label_2] * 5))
             augmented_images_si_di = []

             for augmented_image in augmented_SI:
                 augmented_image_si_di = input_diversity(augmented_image,image_width=32, image_resize=35)
                 augmented_images_si_di.append(augmented_image_si_di)
             augmented_SI = augmented_images_si_di  # torch.cat(tuple(augmented_images_si_di))

         else:
             augmented_SI = augmented
             # augmented_SI = torch.cat(tuple(augmented_SI))
             label_3 = label_2  # torch.cat(tuple([label_2] * 1))
         grad=0
         mean = (0.4914, 0.4822, 0.4465)
         std = (0.2471, 0.2435, 0.2616)
         normalization = T.Compose(
             [

                 T.Normalize(mean, std),
             ]
         )
         for i, each_x in enumerate(augmented_SI):
             # noise.detach_()
             # noise=noise.data
             # noise= V(noise.data, requires_grad=True)
             # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
             output = model(normalization(each_x.float()))
             print(f"size of augmented images: {len(each_x)}")
             # if len(output[0].size()) == 1:
             #     for image_i in range(len(output)):
             #         if opt.parallel:
             #             output[image_i] = output[image_i].reshape(-1, 1001)
             #         else:
             #             output[image_i] = output[image_i].unsqueeze(0)
             # set_trace()
             loss = F.cross_entropy(output, label_3, reduction='sum')  # logit


             loss.backward(retain_graph=True)
             grad =grad+ noise.grad.data
             zero_gradients(noise)
         if opt.cosine_dir:
             gradBetweenIters.append(grad)

         if order[0] == '1':
             # translation invariant
             grad = grad.reshape([1, batch * channel, H, W])
             grad = nn.functional.conv2d(grad, stack_kernel, padding='same', groups=channel * batch)
             grad = grad.reshape([batch, channel, H, W])

         # momentum
         grad = grad / torch.abs(grad).mean([1, 2, 3], keepdim=True)
         grad = momentum * old_grad + grad
         old_grad = grad

         noise = noise + alpha * torch.sign(grad)
         # Avoid out of bound
         noise = torch.clamp(noise, -eps, eps)
         x = img + noise
         x = torch.clamp(x, 0.0, 1.0)
         noise = x - img

         noise = V(noise, requires_grad=True)


     if opt.cosine_dir:
         result=np.zeros(batch)
         for i in range(len(gradBetweenIters)-1):
            result=result+torch.abs(F.cosine_similarity(gradBetweenIters[i].view(batch, -1), gradBetweenIters[i+1].view(batch, -1))).cpu().numpy()
         result=result/9
         result=result.tolist()
         # set_trace()
         with open(f"{opt.method}.txt", 'a') as file:
             for row in result:
                 file.write(str(row) + '\n')




     adv = img + noise.detach()
     # TODO:assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0A6000
     # TODO:assert(np.all(np.logical_and(adv-img>-eps, adv-img<eps)))
     assert rounddown(adv.max().item()) <= 1 and rounddown(adv.min().item()) >= 0
     assert rounddown((adv - img).min().item()) >= -eps and rounddown((adv - img).max()) <= eps

     return adv
