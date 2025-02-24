# encoding:utf-8
"""Implementation of sample attack."""
import logging
from torchtoolbox.transform import Cutout, ImageNetPolicy
from torchvision.transforms import ToPILImage
from scorecam_implement import generate_cam
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
from misc_functions import get_example_params, save_class_activation_images
from PIL import Image, ImageFilter, ImageGrab
from torchvision import transforms
from utils import regularizer, rand_bbox
from colourspace import group_pca_color_augmention
from newbaseline import ens_NewBackend

global randomseed
from FinalultimateCombo import ens_AdvancedUltimateAdmixFGSM

randomseed = 0


def zero_gradients(x):
    if x.grad is not None:
        x.grad.zero_()

    return x


from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
)

# torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='zebin', help='the attack method used')

parser.add_argument('--gpu', type=str, default='0', help='The ID of GPU to use.')
parser.add_argument('--input_csv', type=str, default='dataset/dev_dataset.csv', help='Input csv with images.')
parser.add_argument('--input_dir', type=str, default='dataset/images/', help='Input images.')
parser.add_argument('--output_dir', type=str, default='adv_img_torch/', help='Output directory with adv images.')
parser.add_argument('--model_dir', type=str, default='torch_nets_weight/', help='Model weight directory.')

parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
parser.add_argument("--batch_size", type=int, default=5, help="How many images process at one time.")

parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--lr", type=float, default=None, help="learning rate")
parser.add_argument('--parallel', type=bool, default=False, help='The ID of GPU to use.')
parser.add_argument('--csv_dir', type=str, default="geneticTempleResule", help='dir stored results.')
parser.add_argument('--log', type=str, default="log", help='log results.')
parser.add_argument('--dataset', type=str, default="imagenet", help='dataset.')
opt = parser.parse_args()
logging.basicConfig(filename=f'{opt.log}.txt', level=logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
# device=
device = torch.device("cuda:" + opt.gpu)


def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def mkdir(path):
    """Check if the folder exists, if it does not exist, create it"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


class Normalize(nn.Module):

    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class ImageNet(data.Dataset):
    """load data from img and csv"""

    def __init__(self, dir, csv_path, transforms=None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel']
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)


def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')
    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    device_list = [int(i) for i in cuda_visible_devices.split(',')]
    device_list = [i for i in range(len(cuda_visible_devices.split(',')))]
    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        Normalize('tensorflow'),
        net.KitModel(model_path).eval(), )
    if opt.parallel:
        torch.cuda.set_device(device_list[0])
        global device
        device = 'cuda'
        model = nn.DataParallel(model, device_list).cuda()

    else:
        model = model.cuda(device)
    return model


def rounddown(number):
    return int(number * 10000) / 10000


def get_models(list_nets, model_dir):
    """load models with dict"""
    nets = {}
    for net in list_nets:
        nets[net] = get_model(net, model_dir)
    return nets


def save_img(images, filenames, output_dir):
    """save high quality jpeg"""
    mkdir(output_dir)

    for i, filename in enumerate(filenames):
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(os.path.join(output_dir, filename), quality=100)


def input_diversity(X, p=0.5, image_width=299, image_resize=330):
    # AW optimize: change random to start instead of end
    if torch.rand(()) >= p:
        return X
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


def ens_ADTMFGSM(modelnames, img, label):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    size = 3
    old_grad = torch.zeros_like(img)
    label = torch.cat(tuple([label] * 3))

    batch, channel, H, W = X_pert.shape
    kernel = gkern(9, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])

    for i in range(num_iter):
        zero_gradients(noise)

        x = admix(X_pert + noise, size)

        x_nes_2 = 1 / 2 * x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        logit = 0
        aux_logit = 0
        loss = 0
        aux_logit_count = 0
        for i, each_x in enumerate([x, x_nes_2, x_nes_4, x_nes_8, x_nes_16]):

            for model_name in modelnames:
                model = models[model_name]
                output = model(input_diversity(each_x))
                logit += output[0]

                if not 'resnet' in model_name:
                    aux_logit_count += 1
                    aux_logit += output[1]

            logit = logit / 4
            aux_logit = aux_logit / aux_logit_count

            loss = loss + F.cross_entropy(logit, label)  # logit
            if aux_logit_count > 0:
                loss = loss + F.cross_entropy(aux_logit, label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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


def ens_SDTMIFGSM(modelnames, img, label):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    old_grad = 0.0
    X_pert = img.clone()

    batch, channel, H, W = X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])

    noise = torch.zeros_like(img, requires_grad=True)

    for i in range(num_iter):
        zero_gradients(noise)
        x = X_pert + noise
        x_nes_2 = 1 / 2 * x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        temp_grad = 0
        logit = 0
        aux_logit = 0
        loss = 0
        aux_logit_count = 0
        for i, each_x in enumerate([x, x_nes_2, x_nes_4, x_nes_8, x_nes_16]):
            for model_name in modelnames:
                model = models[model_name]
                output = model(input_diversity(each_x))
                logit += output[0]

                if not 'resnet' in model_name:
                    aux_logit_count += 1
                    aux_logit += output[1]

            logit = logit / 4
            aux_logit = aux_logit / aux_logit_count

            loss = loss + F.cross_entropy(logit, label)  # logit
            if aux_logit_count > 0:
                loss = loss + F.cross_entropy(aux_logit, label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

        # translation invariant
        grad = grad.reshape([1, batch * channel, H, W])
        grad = nn.functional.conv2d(grad, stack_kernel, padding='same', groups=channel * batch)
        grad = grad.reshape([batch, channel, H, W])

        # MI-FGSM
        grad = grad / torch.abs(grad).sum([1, 2, 3], keepdim=True)
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
    assert rounddown(adv.max().item()) <= 1 and rounddown(adv.min().item()) >= 0
    assert rounddown((adv - img).min().item()) >= -eps and rounddown((adv - img).max()) <= eps


    return adv


def mycutout(img, p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False, inplace=False):
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
        # save_img(img, [str(i) + "cutout.jpeg" for i in range(img.shape[0])], opt.output_dir)
        return img
    else:
        return img


def Edge_Enhance(x):
    kernel = torch.unsqueeze(torch.tensor([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]]), 0)
    kernel = torch.unsqueeze(kernel, dim=0).cuda(device)
    kernel = torch.repeat_interleave(kernel, 3, dim=0)

    return F.conv2d(x, kernel, padding='same', groups=3)


def ens_greyschale_FGSM(modelnames, img, label):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
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
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        transform = transforms.Grayscale(num_output_channels=3)
        x_grey = transform(X_pert + noise)
        x_origin = X_pert + noise
        x_nes_2 = 1 / 2 * x_origin
        x_nes_4 = 1 / 4 * x_origin
        x_nes_8 = 1 / 8 * x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2 = 1 / 2 * x_grey
        x_nes_cs_4 = 1 / 4 * x_grey
        x_nes_cs_8 = 1 / 8 * x_grey
        x_nes_cs_16 = 1 / 16 * x_grey
        logit = 0
        aux_logit = 0
        loss = 0
        aux_logit_count = 0
        for i, each_x in enumerate(
                [x_origin, x_nes_2, x_nes_4, x_nes_8, x_nes_16, x_grey, x_nes_cs_2, x_nes_cs_4, x_nes_cs_8,
                 x_nes_cs_16]):
            for model_name in modelnames:
                model = models[model_name]
                output = model(input_diversity(each_x))
                logit += output[0]

                if not 'resnet' in model_name:
                    aux_logit_count += 1
                    aux_logit += output[1]

            logit = logit / 4
            aux_logit = aux_logit / aux_logit_count

            loss = loss + F.cross_entropy(logit, label)  # logit
            if aux_logit_count > 0:
                loss = loss + F.cross_entropy(aux_logit, label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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


def ens_bestcom_admix_FGSM(modelnames, img, label):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
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

        transform = transforms.Grayscale(num_output_channels=3)
        x_grey = transform(X_pert + noise)
        augmented.append(x_grey)

        augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
        augmented.append(Edge_Enhance(X_pert + noise))

        transform = transforms.Compose([
            ImageNetPolicy,
            transforms.ToTensor()
        ])

        pil_trans = ToPILImage()
        x_RE1 = torch.zeros_like(img)
        for i in range(img.shape[0]):
            x_RE1[i] = transform(pil_trans(X_pert[i])).cuda(device) + noise[i]
        augmented.append(x_RE1)

        augmented.append(admix(X_pert + noise, 1))
        for item in augmented:
            augmented_SI.append(item / 2)
            augmented_SI.append(item / 4)
            augmented_SI.append(item / 8)
            augmented_SI.append(item / 16)

        logit = 0
        aux_logit = 0
        loss = 0
        aux_logit_count = 0
        for i, each_x in enumerate(augmented + augmented_SI):
            for model_name in modelnames:
                model = models[model_name]
                output = model(input_diversity(each_x))
                logit += output[0]

                if not 'resnet' in model_name:
                    aux_logit_count += 1
                    aux_logit += output[1]

            logit = logit / 4
            aux_logit = aux_logit / aux_logit_count

            loss = loss + F.cross_entropy(logit, label)  # logit
            if aux_logit_count > 0:
                loss = loss + F.cross_entropy(aux_logit, label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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


def ens_CS_TI_FGSM(modelnames, img, label):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
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
        # set_trace()
        x_cs = channel_shuffle((X_pert + noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        x_origin = X_pert + noise
        x_nes_2 = 1 / 2 * x_origin
        x_nes_4 = 1 / 4 * x_origin
        x_nes_8 = 1 / 8 * x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2 = 1 / 2 * x_cs
        x_nes_cs_4 = 1 / 4 * x_cs
        x_nes_cs_8 = 1 / 8 * x_cs
        x_nes_cs_16 = 1 / 16 * x_cs
        logit = 0
        aux_logit = 0
        loss = 0
        aux_logit_count = 0
        for i, each_x in enumerate(
                [x_origin, x_nes_2, x_nes_4, x_nes_8, x_nes_16, x_cs, x_nes_cs_2, x_nes_cs_4, x_nes_cs_8, x_nes_cs_16]):
            for model_name in modelnames:
                model = models[model_name]
                output = model(input_diversity(each_x))
                logit += output[0]

                if not 'resnet' in model_name:
                    aux_logit_count += 1
                    aux_logit += output[1]

            logit = logit / 4
            aux_logit = aux_logit / aux_logit_count

            loss = loss + F.cross_entropy(logit, label)  # logit
            if aux_logit_count > 0:
                loss = loss + F.cross_entropy(aux_logit, label)  # aux_logit
        # set_trace()
        loss.backward()
        grad = noise.grad.data

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


def ens_BCSH_TI_FGSM(modelnames, img, label):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
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
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        x_bcsh = transform(X_pert + noise)
        x_origin = X_pert + noise
        x_nes_2 = 1 / 2 * x_origin
        x_nes_4 = 1 / 4 * x_origin
        x_nes_8 = 1 / 8 * x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2 = 1 / 2 * x_bcsh
        x_nes_cs_4 = 1 / 4 * x_bcsh
        x_nes_cs_8 = 1 / 8 * x_bcsh
        x_nes_cs_16 = 1 / 16 * x_bcsh
        logit = 0
        aux_logit = 0
        loss = 0
        aux_logit_count = 0
        for i, each_x in enumerate(
                [x_origin, x_nes_2, x_nes_4, x_nes_8, x_nes_16, x_bcsh, x_nes_cs_2, x_nes_cs_4, x_nes_cs_8,
                 x_nes_cs_16]):
            for model_name in modelnames:
                model = models[model_name]
                output = model(input_diversity(each_x))
                logit += output[0]

                if not 'resnet' in model_name:
                    aux_logit_count += 1
                    aux_logit += output[1]

            logit = logit / 4
            aux_logit = aux_logit / aux_logit_count

            loss = loss + F.cross_entropy(logit, label)  # logit
            if aux_logit_count > 0:
                loss = loss + F.cross_entropy(aux_logit, label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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


def ens_cj_TI_FGSM(modelnames, img, label):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
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
        # set_trace()

        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        x1, x2, x3 = group_pca_color_augmention((X_pert + noise))

        x_origin = X_pert + noise
        x_nes_2 = 1 / 2 * x_origin
        x_nes_4 = 1 / 4 * x_origin
        x_nes_8 = 1 / 8 * x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2 = 1 / 2 * x1
        x_nes_cs_4 = 1 / 4 * x1
        x_nes_cs_8 = 1 / 8 * x1
        x_nes_cs_16 = 1 / 16 * x1
        x_nes_pca_2 = 1 / 2 * x2
        x_nes_pca_4 = 1 / 4 * x2
        x_nes_pca_8 = 1 / 8 * x2
        x_nes_pca_16 = 1 / 16 * x2
        logit = 0
        aux_logit = 0
        loss = 0
        aux_logit_count = 0
        for i, each_x in enumerate(
                [x_origin, x1, x2, x_nes_2, x_nes_4, x_nes_8, x_nes_16, x1, x2, x_nes_cs_2, x_nes_cs_4, x_nes_cs_8,
                 x_nes_cs_16, x_nes_pca_2, x_nes_pca_4, x_nes_pca_8, x_nes_pca_16]):
            for model_name in modelnames:
                model = models[model_name]
                output = model(input_diversity(each_x))
                logit += output[0]

                if not 'resnet' in model_name:
                    aux_logit_count += 1
                    aux_logit += output[1]

            logit = logit / 4
            aux_logit = aux_logit / aux_logit_count

            loss = loss + F.cross_entropy(logit, label)  # logit
            if aux_logit_count > 0:
                loss = loss + F.cross_entropy(aux_logit, label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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


def ens_VMI_DI_TI_SI_FGSM(modelnames, images, labels):
    r"""
    Overridden.
    """
    beta = 1.5
    # if opt.dataset == 'imagenet':
    eps = opt.max_epsilon / 255.0
    # else:
    #     eps = opt.max_epsilon

    alpha = eps / opt.num_iter

    images = images.clone().detach()
    labels = labels.clone().detach()
    batch, channel, H, W = images.shape
    # if opt.dataset=='cifar-10':
    #     kernel_size = 3
    # elif opt.dataset=='imagenet':
    kernel_size = 7
    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])
    momentum = torch.zeros_like(images).detach()

    v = torch.zeros_like(images).detach()

    loss = nn.CrossEntropyLoss()

    adv_images = images.clone().detach()
    labels = torch.cat(tuple([labels] * 5))
    logit = 0
    aux_logit = 0
    loss_value = 0
    aux_logit_count = 0
    for _ in range(opt.num_iter):
        adv_images.requires_grad = True
        si_adv_images = torch.cat(tuple([adv_images, adv_images / 2, adv_images / 4, adv_images / 8, adv_images / 16]),
                                  axis=0)

        for model_name in modelnames:
            model = models[model_name]
            output = model(input_diversity(si_adv_images))
            logit += output[0]

            if not 'resnet' in model_name:
                aux_logit_count += 1
                aux_logit += output[1]

        logit = logit / 4
        aux_logit = aux_logit / aux_logit_count

        cost = F.cross_entropy(logit, labels)  # logit
        if aux_logit_count > 0:
            cost = cost + F.cross_entropy(aux_logit, labels)  # aux_logit

        adv_grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
        intimate_grad = adv_grad + v

        current_grad = intimate_grad.reshape([1, batch * channel, H, W])
        current_grad = nn.functional.conv2d(current_grad, stack_kernel, padding='same', groups=channel * batch)
        current_grad = current_grad.reshape([batch, channel, H, W])

        grad = (current_grad) / torch.mean(torch.abs(current_grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * opt.momentum
        momentum = grad

        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(images).detach()

        for _ in range(20):
            neighbor_images = adv_images.detach() + \
                              torch.randn_like(images).uniform_(-1 * eps * beta, eps * beta)
            neighbor_images.requires_grad = True
            input = torch.cat(tuple(
                [neighbor_images, neighbor_images / 2, neighbor_images / 4, neighbor_images / 8, neighbor_images / 16]),
                              axis=0)
            for model_name in modelnames:
                model = models[model_name]
                output = model(input_diversity(input))
                logit += output[0]
                if not 'resnet' in model_name:
                    aux_logit_count += 1
                    aux_logit += output[1]

            logit = logit / 4
            aux_logit = aux_logit / aux_logit_count
            # outputs =model(input_diversity(input))

            cost = loss(logit, labels)
            # if using_aux_logit:
            cost += loss(aux_logit, labels)

            GV_grad += torch.autograd.grad(cost, neighbor_images,
                                           retain_graph=False, create_graph=False)[0]
        # obtaining the gradient variance

        v = GV_grad / 20 - adv_grad

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    assert rounddown(adv_images.max().item()) <= 1 and rounddown(adv_images.min().item()) >= 0
    assert rounddown((adv_images - images).min().item()) >= -eps and rounddown((adv_images - images).max()) <= eps
    return adv_images


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def attack(combination_name, img, img_n, label, models, method='zebin'):
    """generate adversarial images"""
    if method == 'ens_SDTMIFGSM':
        return ens_SDTMIFGSM(combination_name, img, label)
    elif method == 'ens_greyschale_FGSM':
        return ens_greyschale_FGSM(combination_name, img, label)
    elif method == 'ens_CS_TI_FGSM':
        return ens_CS_TI_FGSM(combination_name, img, label)
    elif method == 'ens_cj_TI_FGSM':
        return ens_cj_TI_FGSM(combination_name, img, label)
    elif method == 'ens_BCSH_TI_FGSM':
        return ens_BCSH_TI_FGSM(combination_name, img, label)
    elif method == 'ens_ADTMFGSM':
        return ens_ADTMFGSM(combination_name, img, label)
    elif method == 'ens_UC_base':
        return ens_bestcom_admix_FGSM(combination_name, img, label)
    elif method == "ens_VMI_DI_TI_SI_FGSM":
        return ens_VMI_DI_TI_SI_FGSM(combination_name, img, label)
    elif method == 'ens_UNDP':
        PI = ens_NewBackend()
        return PI(opt, img, label, models, combination_name, device)
    elif method.startswith("ens_UC_gen"):
        order = int(method[10:])
        return ens_AdvancedUltimateAdmixFGSM(combination_name, img, img_n, label, order, opt, device, models=models)


# Create models
list_nets = [
    'tf_inception_v4',
    'tf_resnet_v2_50',
    'tf_resnet_v2_101',
    'tf_resnet_v2_152',
    'tf_inc_res_v2',
    'tf_adv_inception_v3',
    'tf_ens3_adv_inc_v3',
    'tf_ens4_adv_inc_v3',
    'tf_ens_adv_inc_res_v2',
    'tf_inception_v3',
]
models = get_models(list_nets, opt.model_dir)


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


def main():
    index = 0

    transforms = T.Compose([T.ToTensor()])
    df_res = pd.DataFrame(
        columns=['dst_benign_success', 'name'])
    # Load inputs
    inputs = ImageNet(opt.input_dir, opt.input_csv, transforms)
    transforms2 = T.Compose([T.Resize(299), T.ToTensor()])
    neural = ImageNet('/home/sharifm/students/zebinyun/neural-style-pt/output', opt.input_csv, transforms2)
    data_loader = DataLoader(inputs, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    data_loader_neural = DataLoader(neural, batch_size=opt.batch_size, shuffle=False, pin_memory=True,
                                    num_workers=0)
    input_num = len(inputs)

    # Start iteration
    method = opt.method
    print(method)

    if method == 'all':
        methods = ['ens_SDTMIFGSM', 'ens_greyschale_FGSM']
    elif method == 'newall':
        methods = ['ens_CS_TI_FGSM', 'ens_cj_TI_FGSM', 'ens_BCSH_TI_FGSM']
    elif method == 'method_0810':
        methods = ['zebin', 'DI2FGSM', 'TIwithDI', 'PIFGSM', 'zebin', 'TI_DI_AITM']
    elif method == 'final':
        methods = ['ens_bestcom_admix_FGSM', 'ens_SDTMIFGSM', 'ens_ADTMFGSM', 'ens_greyschale_FGSM', 'ens_CS_TI_FGSM',
                   'ens_cj_TI_FGSM', 'ens_BCSH_TI_FGSM']
    else:
        methods = [method]
    global use_model

    nameWithCombo = {"4_50_101_152": ['tf_inception_v4',
                                      'tf_resnet_v2_50',
                                      'tf_resnet_v2_101',
                                      'tf_resnet_v2_152', ]}
    name = '4_50_101_152'
    ensemble_combination = [nameWithCombo[name]]


    for index2, surrogate in enumerate(ensemble_combination):
        use_model = surrogate
        global df
        # df = pd.DataFrame(columns=include_methods, dtype=float)
        global image_loss_record
        image_loss_record = 0
        for method in methods:
            # Initialization parameters
            correct_num = {}
            logits = {}
            for net in list_nets:
                correct_num[net] = 0

            global imageindex
            imageindex = 0

            for (images, filename, label), (images_n, filename_n, label_n) in zip(data_loader, data_loader_neural):
                import os


                imageindex += 1
                logging.info(f"iternation:{imageindex}")
                label = label.cuda(device)
                images = images.cuda(device)
                images_n = images_n.cuda(device)
                # demo=group_pca_color_augmention(images)
                # save_img(demo, filename, opt.output_dir)

                # Start Attack
                adv_img = attack(surrogate, images, images_n, label, method=method, models=models)

                # Save adversarial examples
                # save_img(adv_img, filename, opt.output_dir)
                save_img(adv_img, filename, f"{method}_{name}")
                # Prediction

                with torch.no_grad():
                    for net in list_nets:
                        logits[net] = models[net](adv_img)
                        correct_num[net] += (torch.argmax(logits[net][0], axis=1) != label).detach().sum().cpu()
                        logging.info(f"net: {net}, trans:{correct_num[net]}")
            # Print attack success rate

            for net in list_nets:
                df_res.loc[index, 'name'] = f'transferability_{name}to{net}_{method}'
                df_res.loc[index, 'success_rate'] = round((correct_num[net] / input_num), 3)
                index += 1
                print('{} attack {} success rate: {:.2%} by {}'.format(name, net, correct_num[net] / input_num,
                                                                       method))


            df_res.to_csv(f"./transferability_result/{opt.method}_{name}.csv")


if __name__ == '__main__':
    seed_torch(0)
    main()

