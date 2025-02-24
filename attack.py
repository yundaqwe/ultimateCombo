# encoding:utf-8
"""Implementation of sample attack."""

from torchtoolbox.transform import Cutout,ImageNetPolicy,CIFAR10Policy
from torchvision.datasets import CIFAR10
from torchvision.transforms import  ToPILImage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from torch.autograd import Variable as V
import math
from torchvision.datasets import CIFAR10
from models.cifar_resnet import resnet as resnet_cifar
from torch.utils import data
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from ipdb import  set_trace
import logging
from PIL import Image, ImageFilter, ImageGrab
from torchvision import transforms
from utils import regularizer,rand_bbox
from colourspace import  group_pca_color_augmention
from UCA import AdvancedUltimateAdmixFGSM
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


list_nets = [
    'tf_inception_v3',
    'tf_inception_v4',
    'tf_resnet_v2_50',

    'tf_resnet_v2_152',
    'tf_inc_res_v2',
    'tf_resnet_v2_101',
    'tf_adv_inception_v3',
    'tf_ens3_adv_inc_v3',
    'tf_ens4_adv_inc_v3',
    'tf_ens_adv_inc_res_v2',

    ]
#torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='imagenet', help='the dataset used')
parser.add_argument('--method', type=str, default='MIFGSM', help='the attack method used')

parser.add_argument('--gpu', type=str, default='0', help='The ID of GPU to use.')
parser.add_argument('--input_csv', type=str, default='dataset/dev_dataset.csv', help='Input csv with images.')
parser.add_argument('--input_dir', type=str, default='dataset/images/', help='Input images.')
parser.add_argument('--output_dir', type=str, default='adv_img_torch/', help='Output directory with adv images.')
parser.add_argument('--model_dir', type=str, default='torch_nets_weight/', help='Model weight directory.')

parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--num_workers", type=int, default=0, help="How many  workers process at one time.")
parser.add_argument("--momentum", type=float, default=1, help="Momentum")
parser.add_argument("--surrogate", type=str, default=None, help="which used to craft the adversarial example")
parser.add_argument("--lr", type=float, default=None, help="learning rate")
parser.add_argument("--v", type=int, default=20, help="Variance")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument('--parallel', type=bool, default=False, help='The ID of GPU to use.')
parser.add_argument('--csv_dir', type=str, default="geneticTempleResule", help='dir stored results.')
parser.add_argument('--log', type=str, default="log", help='log results.')
parser.add_argument('--cosine', type=bool, default=False, help='check cosine similarity results.')



opt = parser.parse_args()
logging.basicConfig(filename=f'{opt.log}.txt', level=logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)
normalization = T.Compose(
 [
     # T.ToTensor(),
     T.Normalize(mean, std),
 ]
)
os.environ['TORCH_HOME']='./pretrain_weight'
device = torch.device("cuda:"+opt.gpu)

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
    # set_trace()
    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    device_list =[int(i) for i in  cuda_visible_devices.split(',')]
    device_list=[i for i in range(len(cuda_visible_devices.split(',')))]
    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        Normalize('tensorflow'),
        net.KitModel(model_path).eval(),)
    if opt.parallel:
        torch.cuda.set_device(device_list[0])
        global device
        device='cuda'
        model=nn.DataParallel(model,device_list).cuda()

    else:
        device = torch.device("cuda:"+opt.gpu )
        model=model.cuda(device)
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

def IFGSM(model, img, label,using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = opt.momentum
    batch, channel, H, W = img.shape
    noise = torch.zeros_like(img, requires_grad=True)

    old_grad = 0.0
    for i in range(num_iter):
        zero_gradients(noise)
        x = img + noise
        output = model(x)

        loss = F.cross_entropy(output[0], label)  # logit
        if using_aux_logit:
            loss += F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data
        # MI-FGSM
        # grad = grad / torch.abs(grad).mean([1,2,3], keepdim=True)
        # grad = momentum * old_grad + grad
        # old_grad = grad

        noise = noise + alpha * torch.sign(grad)
        # Avoid out of bound
        noise = torch.clamp(noise, -eps, eps)
        x = img + noise
        x = torch.clamp(x, 0.0, 1.0)
        noise = x - img
        noise = V(noise, requires_grad=True)
    adv = img + noise.detach()
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps



    if opt.cosine:
        grad =  grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None
def MIFGSM(model, img, label,using_aux_logit=True):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = opt.momentum

    noise = torch.zeros_like(img, requires_grad=True)

    old_grad = 0.0

    for i in range(num_iter):
        zero_gradients(noise)
        x = img + noise
        output = model(x)

        loss = F.cross_entropy(output[0], label)  # logit
        if using_aux_logit:
            loss += F.cross_entropy(output[1], label)  # aux_logit

        loss.backward()
        grad = noise.grad.data
        # MI-FGSM
        grad = grad / torch.abs(grad).sum([1,2,3], keepdim=True)
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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps
    #visualize
    global imageindex

    batch, channel, H, W = img.shape
    if opt.cosine:
        grad =  grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None








def input_diversity(X, p=0.5, image_width=299, image_resize=330):
    #AW optimize: change random to start instead of end
    if torch.rand(()) >= p:
        return X
    if opt.dataset=='cifar-10':
        image_width =32
        image_resize = 35
    rnd = torch.randint(image_width, image_resize, ())
    rescaled = nn.functional.interpolate(X, [rnd, rnd])
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem,())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [image_width, image_width])
    #return padded if torch.rand(()) < p else X

    return padded


def Edge_Enhance(x):

    kernel=torch.unsqueeze(torch.tensor([[-0.5,-0.5,-0.5],[-0.5,5,-0.5],[-0.5,-0.5,-0.5]]),0)
    kernel=torch.unsqueeze(kernel,dim=0).cuda(device)
    kernel=torch.repeat_interleave(kernel,3,dim=0)

    return F.conv2d(x,kernel,padding='same',groups=3)
def CUTOUT_MIFGSM(model, img, label,using_aux_logit):
    eps = opt.max_epsilon / 255.0

    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)


        x_origin=X_pert + noise
        x_RE1 = mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1))

        for i, each_x in enumerate([x_origin, x_RE1]):
            output = model(each_x)
            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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


def SI_NI_TI_DI_FGSM(model, img, label,using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)

    for i in range(num_iter):
        zero_gradients(noise)
        x=X_pert+ noise+ momentum * alpha * grad
        x_nes_2=1/2*x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        temp_grad=0
        for i,each_x in enumerate([x,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16]):
            zero_gradients(noise)
            output = model(input_diversity(each_x))
            loss = F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
            loss.backward()
            temp_grad+=noise.grad.data
            # if i ==0:
            #     loss = F.cross_entropy(output[0], label)  # logit
            # else:
            #     loss=loss+ F.cross_entropy(output[0], label)  # logit
            # if using_aux_logit:
            #     loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        # loss.backward()
        grad = temp_grad
        # MI-FGSM
        grad = grad / torch.abs(grad).sum([1,2,3], keepdim=True)
        grad = momentum * old_grad + grad
        old_grad = grad
        # grad = grad / torch.abs(grad).mean([1, 2, 3], keepdim=True)

        noise = noise + alpha * torch.sign(grad)
        # Avoid out of bound
        noise = torch.clamp(noise, -eps, eps)
        x = img + noise
        x = torch.clamp(x, 0.0, 1.0)
        noise = x - img
        noise = V(noise, requires_grad=True)

    adv = img + noise.detach()
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps
    batch, channel, H, W = img.shape
    if opt.cosine:
        grad = grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None
def neural_TDSMFGSM(model, img,img_n, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    x_neural=img_n.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    size = 3
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

        # x = admix(X_pert + noise, size)
        x=X_pert + noise
        x_n=x_neural+noise

        x_nes_2 = 1 / 2 * x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        x_n_2=1 / 2 * img_n
        x_n_4 = 1 / 2 *x_n
        x_n_8 = 1 / 2 * x_n
        x_n_16 = 1 / 2 * x_n
        for i, each_x in enumerate([x,img_n, x_nes_2, x_nes_4, x_nes_8, x_nes_16,x_n_2,x_n_4,x_n_8,x_n_16]):
            output = model(input_diversity(each_x))

            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
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

    batch, channel, H, W = img.shape
    if opt.cosine:
        grad = grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None
def neuraltransfer_MIFGSM(model, img,img_n, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    x_neural=img_n.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)


    for i in range(num_iter):
        zero_gradients(noise)

        # x = admix(X_pert + noise, size)
        x=X_pert + noise
        x_n=x_neural+noise


        for i, each_x in enumerate([x,x_n]):
            output = model(each_x)

            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data



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

    assert rounddown(adv.max().item()) <= 1 and rounddown(adv.min().item()) >= 0
    assert rounddown((adv - img).min().item()) >= -eps and rounddown((adv - img).max()) <= eps

    batch, channel, H, W = img.shape
    if opt.cosine:
        grad = grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None
def channel_shuffle_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    if opt.lr:
        alpha = opt.lr
    else:
        alpha =   eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_cs
        x_nes_cs_4 = 1 / 4 * x_cs
        x_nes_cs_8 = 1 / 8 * x_cs
        x_nes_cs_16 = 1 / 16 * x_cs
        for i,each_x in enumerate([x_origin,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_cs,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps






    return adv
def cutout_TI_FGSM(model, img, label, using_aux_logit):

    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    if opt.lr:
        alpha = opt.lr
    else:
        alpha =   eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)

        global  randomseed
        randomseed += 1
        # set_trace()
        x_RE1=torch.zeros_like(img)
        randomseed += 1
        seed_torch(randomseed)
        x_RE1=mycutout(X_pert+noise,p=0.5,ratio=(1, 1 ),value=(0, 1))
        seed_torch(0)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_RE1
        x_nes_cs_4 = 1 / 4 * x_RE1
        x_nes_cs_8 = 1 / 8 * x_RE1
        x_nes_cs_16 = 1 / 16 * x_RE1



        for i,each_x in enumerate([x_origin,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps






    return adv



def CS_DST(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    if opt.lr:
        alpha = opt.lr
    else:
        alpha =   eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_cs
        x_nes_cs_4 = 1 / 4 * x_cs
        x_nes_cs_8 = 1 / 8 * x_cs
        x_nes_cs_16 = 1 / 16 * x_cs
        for i,each_x in enumerate([x_origin,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_cs,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps






    return adv

def batch_grad(model, img, label,using_aux_logit,noise,grad):
    for iter  in range(20):
        neighbor = torch.cuda.FloatTensor(img.size())
        torch.randn(img.size(), out=neighbor)
        img2 =img + neighbor *1.5
        x_neighbor = img2.clone() +noise
        x_neighbor_2 = 1/2. * x_neighbor
        x_neighbor_4 = 1/4. * x_neighbor
        x_neighbor_8 = 1/8. * x_neighbor
        x_neighbor_16 = 1/16. * x_neighbor

        for i, each_x in enumerate([x_neighbor, x_neighbor_2, x_neighbor_4, x_neighbor_8,x_neighbor_16]):
            zero_gradients(noise)
            output = model(each_x)
            loss = F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss += F.cross_entropy(output[1], label)  # aux_logit
            loss.backward()

            grad +=noise.grad.data*(1/2)**i

    return  grad


def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel
def TranslationInvariantAttack(model, img, label,using_aux_logit,use_diversity=True):
    X_pert = img.clone()
    batch,channel,H,W=X_pert.shape
    kernel = gkern(15, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])

    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = opt.momentum

    noise = torch.zeros_like(img, requires_grad=True)
    X_pert = img.clone()
    # X_pert.requires_grad = True
    old_grad = 0.0
    for i in range(num_iter):
        zero_gradients(noise)

        if use_diversity:
            x =input_diversity(X_pert+noise, p=0.5, image_width=299, image_resize=330)
        else:
            x=X_pert+ noise
        output = model(x)
        loss = F.cross_entropy(output[0], label)  # logit
        if using_aux_logit:
            loss += F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data
        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])
        # momentum
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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps

    batch, channel, H, W = img.shape
    if opt.cosine:
        grad = grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None

import numpy as np

import scipy.stats as st


def project_noise(x, stack_kern, kern_size,channel,batch,H,W):
    x = torch.nn.functional.pad(x, (kern_size,kern_size,kern_size,kern_size,0,0,0,0), "constant", 0)
    x = x.reshape([1, batch * channel, H+2*kern_size, W+2*kern_size])


    x = nn.functional.conv2d(x,stack_kern,padding='valid',groups=channel*batch)
    x=x.reshape([batch, channel, H, W])
    return x
def project_kern(kern_size,batch=10):

    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    channel=3
    stack_kern = np.expand_dims(stack_kern, 0)
    stack_kern = torch.tensor(stack_kern).cuda(device)
    stack_kern=stack_kern.repeat(batch,1,1,1)
    iteration, __, H_Kernel, W_Kernel = stack_kern.shape
    stack_kern= stack_kern.transpose(0,1)
    stack_kern=stack_kern.reshape([batch*channel,1,H_Kernel,W_Kernel])
    return stack_kern, kern_size // 2
def admix(x,size=3):
    portion=0.2
    # size=3 #mixup
    return  torch.cat(tuple([(x + portion * x[torch.randperm(x.size(0))]) for _ in range(size)]), axis=0)/(1+portion*size)




def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()

    x2=x
    x3=x
    x4=x
    for i in range(x.shape[0]):
        tem=x2[i][1]
        x2[i][1]=x2[i][2]
        x2[i][2]=tem
    return x2



def rgb_to_grayscale(img, num_output_channels: int = 1,r_c=0.2989,g_c=0.587,b_c=0.114) :
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (r_c * r + g_c* g + b_c* b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img

def greyschale_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        transform = transforms.Grayscale(num_output_channels=3)
        x_grey=transform(X_pert+ noise)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_grey
        x_nes_cs_4 = 1 / 4 * x_grey
        x_nes_cs_8 = 1 / 8 * x_grey
        x_nes_cs_16 = 1 / 16 * x_grey
        for i,each_x in enumerate([x_origin,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_grey,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps


    batch, channel, H, W = img.shape
    if opt.cosine:
        grad =  grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None

def CJ_DST(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)

        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        x_bcsh=transform(X_pert+ noise)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_bcsh
        x_nes_cs_4 = 1 / 4 * x_bcsh
        x_nes_cs_8 = 1 / 8 * x_bcsh
        x_nes_cs_16 = 1 / 16 * x_bcsh
        for i,each_x in enumerate([x_origin,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_bcsh,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16]):
            output = model(input_diversity(each_x))

            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps





    batch, channel, H, W = img.shape
    if opt.cosine:
        grad =  grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None

def mycutout(img,p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False, inplace=False):
    if random.random() < p:
    # if True:

        batch, img_c,img_h, img_w = img.shape
        s = random.uniform(*scale)
        s = s * img_h * img_w
        r = random.uniform(*ratio)
        w = int(math.sqrt(s / r))
        h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)
        c = torch.tensor(0).to(device)

        for i in range(batch):
            img[i,:,left:left + w,top:top + h]=c
        # save_img(img, [str(i) + "cutout.jpeg" for i in range(img.shape[0])], opt.output_dir)
        return  img
    else:
        return  img


def grid_search(model, img,img_n, label, using_aux_logit,order):

    order=str(bin(order))[2:]
    extrazero=6-len(order)
    order=extrazero*'0'+order
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


        augmented=[]
        augmented_SI=[]
        augmented.append(X_pert + noise)
        for index,whethertouse in enumerate(order):
            if whethertouse=='1':
                if index==0:
                    transform = transforms.Grayscale(num_output_channels=3)
                    x_grey = transform(X_pert + noise)
                    augmented.append(x_grey)

                elif index==1:
                    # rand_index = torch.randperm(img.shape[0]).cuda(device)
                    # bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
                    # x = X_pert + noise
                    #
                    # x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
                elif index==2:
                    x_neural = img_n.clone()
                    augmented.append(x_neural+noise)
                elif index==3:
                    augmented.append(Edge_Enhance(X_pert + noise))
                elif index==4:
                    transform = transforms.Compose([
                        ImageNetPolicy,
                        transforms.ToTensor()
                    ])

                    pil_trans = ToPILImage()
                    x_RE1 = torch.zeros_like(img)
                    for i in range(img.shape[0]):
                        x_RE1[i] = transform(pil_trans(X_pert[i])).cuda(device) + noise[i]
                    augmented.append(x_RE1)

        augmented.append(admix(X_pert + noise,1))
        if order[5] == '1':
            for item in augmented:
                augmented_SI.append(item/2)
                augmented_SI.append(item / 4)
                augmented_SI.append(item / 8)
                augmented_SI.append(item /16)






        for i, each_x in enumerate(augmented_SI+augmented):
            # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
            if order[5]=='1':

                output = model(input_diversity(each_x))
            else:
                output = model(each_x)
            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data
        if order[5] == '1':
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
def FPCA_DST(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    if opt.lr:
        alpha = opt.lr
    else:
        alpha =   eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()

        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        x1,x2,x3=group_pca_color_augmention((X_pert+ noise))

        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x1
        x_nes_cs_4 = 1 / 4 * x1
        x_nes_cs_8 = 1 / 8 * x1
        x_nes_cs_16 = 1 / 16 * x1
        x_nes_pca_2=1/2*x2
        x_nes_pca_4 = 1 / 4 * x2
        x_nes_pca_8 = 1 / 8 * x2
        x_nes_pca_16 = 1 / 16 * x2
        for i,each_x in enumerate([x_origin,x1,x2,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16,x_nes_pca_2,x_nes_pca_4,x_nes_pca_8,x_nes_pca_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps






    return adv
def bestcombo_admixFGSMkernel(model, img,label, using_aux_logit,kernel_size):

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
    kernel = gkern(kernel_size, 3).astype(np.float32)
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


        augmented=[]
        augmented_SI=[]
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
        augmented.append(admix(X_pert + noise,1))

        for item in augmented:
            augmented_SI.append(item/2)
            augmented_SI.append(item / 4)
            augmented_SI.append(item / 8)
            augmented_SI.append(item /16)






        for i, each_x in enumerate(augmented_SI+augmented):
            output = model(input_diversity(each_x))

            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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

def grad_search(model, img, img_n, label, using_aux_logit, order):

    order=str(bin(order))[2:]
    extrazero=6-len(order)
    order=extrazero*'0'+order
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


        augmented=[]
        augmented_SI=[]
        augmented.append(X_pert + noise)
        for index,whethertouse in enumerate(order):
            if whethertouse=='1':
                if index==0:
                    transform = transforms.Grayscale(num_output_channels=3)
                    x_grey = transform(X_pert + noise)
                    augmented.append(x_grey)

                elif index==1:
                    # rand_index = torch.randperm(img.shape[0]).cuda(device)
                    # bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
                    # x = X_pert + noise
                    #
                    # x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
                elif index==2:
                    x_neural = img_n.clone()
                    augmented.append(x_neural+noise)
                elif index==3:
                    augmented.append(Edge_Enhance(X_pert + noise))
                elif index==4:
                    transform = transforms.Compose([
                        ImageNetPolicy,
                        transforms.ToTensor()
                    ])

                    pil_trans = ToPILImage()
                    x_RE1 = torch.zeros_like(img)
                    for i in range(img.shape[0]):
                        x_RE1[i] = transform(pil_trans(X_pert[i])).cuda(device) + noise[i]
                    augmented.append(x_RE1)

        augmented.append(admix(X_pert + noise,1))
        if order[5] == '1':
            for item in augmented:
                augmented_SI.append(item/2)
                augmented_SI.append(item / 4)
                augmented_SI.append(item / 8)
                augmented_SI.append(item /16)






        for i, each_x in enumerate(augmented_SI+augmented):
            # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
            if order[5]=='1':

                output = model(input_diversity(each_x))
            else:
                output = model(each_x)
            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data
        if order[5] == '1':
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
def SI_DI_TI_MIFGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    old_grad=0.0
    X_pert  = img.clone()

    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    noise = torch.zeros_like(img, requires_grad=True)

    for i in range(num_iter):
        zero_gradients(noise)
        x=X_pert+ noise
        x_nes_2=1/2*x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        temp_grad=0
        for i,each_x in enumerate([x,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16]):
            output = model(input_diversity(each_x))

            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

        # translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

        # MI-FGSM
        grad = grad / torch.abs(grad).sum([1,2,3], keepdim=True)
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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps





    batch, channel, H, W = img.shape
    if opt.cosine:
        grad =  grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None


def fpca_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    if opt.lr:
        alpha = opt.lr
    else:
        alpha =   eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()

        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        x1,x2,x3=group_pca_color_augmention((X_pert+ noise))

        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x1
        x_nes_cs_4 = 1 / 4 * x1
        x_nes_cs_8 = 1 / 8 * x1
        x_nes_cs_16 = 1 / 16 * x1
        x_nes_pca_2=1/2*x2
        x_nes_pca_4 = 1 / 4 * x2
        x_nes_pca_8 = 1 / 8 * x2
        x_nes_pca_16 = 1 / 16 * x2
        for i,each_x in enumerate([x_origin,x1,x2,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16,x_nes_pca_2,x_nes_pca_4,x_nes_pca_8,x_nes_pca_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps






    return adv
def admix_TI_FGSM(model, img, label, using_aux_logit):
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
        for i, each_x in enumerate([x, x_nes_2, x_nes_4, x_nes_8, x_nes_16]):
            output = model(input_diversity(each_x))

            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
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

    batch, channel, H, W = img.shape
    if opt.cosine:
        grad = grad.reshape([batch, channel, H, W])

        return adv, grad
    return adv, None

def utimate_MIFGSM(model, img,img_n, label, using_aux_logit,order):

    order=str(bin(order))[2:]
    extrazero=6-len(order)
    order=extrazero*'0'+order
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

        global randomseed
        randomseed += 1
        augmented=[]
        augmented_SI=[]
        augmented.append(X_pert + noise)
        seed_torch(randomseed)
        for index,whethertouse in enumerate(order):
            if whethertouse=='1':
                if index==0:
                    transform = transforms.Grayscale(num_output_channels=3)
                    x_grey = transform(X_pert + noise)
                    augmented.append(x_grey)

                elif index==1:
                    # rand_index = torch.randperm(img.shape[0]).cuda(device)
                    # bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
                    # x = X_pert + noise
                    #
                    # x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
                elif index==2:
                    x_neural = img_n.clone()
                    augmented.append(x_neural+noise)
                elif index==3:
                    augmented.append(Edge_Enhance(X_pert + noise))
                elif index==4:
                    transform = transforms.Compose([
                        ImageNetPolicy,
                        transforms.ToTensor()
                    ])

                    pil_trans = ToPILImage()
                    x_RE1 = torch.zeros_like(img)
                    for i in range(img.shape[0]):
                        x_RE1[i] = transform(pil_trans(X_pert[i])).cuda(device) + noise[i]
                    augmented.append(x_RE1)

        if order[5] == '1':
            for item in augmented:
                augmented_SI.append(item/2)
                augmented_SI.append(item / 4)
                augmented_SI.append(item / 8)
                augmented_SI.append(item /16)



        seed_torch(0)


        for i, each_x in enumerate(augmented_SI+augmented):
            # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
            if order[5]=='1':

                output = model(input_diversity(each_x))
            else:
                output = model(each_x)
            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data
        if order[5] == '1':
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
def RE_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    if opt.lr:
        alpha = opt.lr
    else:
        alpha =   eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)

        transform =transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value= 'random')
        x_RE1=transform(X_pert+ noise)

        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_RE1
        x_nes_cs_4 = 1 / 4 * x_RE1
        x_nes_cs_8 = 1 / 8 * x_RE1
        x_nes_cs_16 = 1 / 16 * x_RE1



        for i,each_x in enumerate([x_origin,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_RE1,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16]):
            output = model(input_diversity(each_x))

            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps






    return adv
def cutmix_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    if opt.lr:
        alpha = opt.lr
    else:
        alpha =   eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    size = 3
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

        rand_index = torch.randperm(img.shape[0]).cuda(device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
        x = X_pert + noise

        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        x_ori=X_pert + noise
        x_ori_2 = 1 / 2 * x_ori
        x_ori_4 = 1 / 4 * x_ori
        x_ori_8 = 1 / 8 * x_ori
        x_ori_16 = 1 / 16 * x_ori
        x_nes_2 = 1 / 2 * x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        for i, each_x in enumerate([x_ori,x_ori_2,x_ori_4,x_ori_8,x_ori_16,x, x_nes_2, x_nes_4, x_nes_8, x_nes_16]):
            output = model(input_diversity(each_x))

            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
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
def autoaugment_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    if opt.lr:
        alpha = opt.lr
    else:
        alpha =   eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        if opt.dataset=='imagenet':
            transform = transforms.Compose([
                ImageNetPolicy,
                transforms.ToTensor()
            ])
        elif opt.dataset=='cifar-10':
            transform = transforms.Compose([
                CIFAR10Policy,
                transforms.ToTensor()
            ])

        pil_trans = ToPILImage()
        global  randomseed
        randomseed += 1

        x_RE1=torch.zeros_like(img)
        for i in range(img.shape[0]):
            x_RE1[i]=transform(pil_trans(X_pert[i])).cuda(device)+noise[i]
        # save_img(x_RE1,[str(i)+"autoA.jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        randomseed += 1
        # seed_torch(randomseed)
        # x_RE2=torch.zeros_like(img)
        # for i in range(img.shape[0]):
        #     x_RE2[i]=(transform(((X_pert.cpu().detach().numpy())[i]).swapaxes(0,2))).permute(0,1,2).cuda(device)+noise[i]
        # randomseed += 1
        # seed_torch(randomseed)
        # x_RE3=torch.zeros_like(img)
        # for i in range(img.shape[0]):
        #     x_RE3[i]=(transform(((X_pert.cpu().detach().numpy())[i]).swapaxes(0,2))).permute(0,1,2).cuda(device)+noise[i]
        seed_torch(0)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_RE1
        x_nes_cs_4 = 1 / 4 * x_RE1
        x_nes_cs_8 = 1 / 8 * x_RE1
        x_nes_cs_16 = 1 / 16 * x_RE1
        #
        # x_nes_RE2_2=1/2*x_RE2
        # x_nes_RE2_4 = 1 / 4 * x_RE2
        # x_nes_RE2_8 = 1 / 8 * x_RE2
        # x_nes_RE2_16 = 1 / 16 * x_RE2
        #
        # x_nes_RE3_2=1/2*x_RE3
        # x_nes_RE3_4 = 1 / 4 * x_RE3
        # x_nes_RE3_8 = 1 / 8 * x_RE3
        # x_nes_RE3_16 = 1 / 16 * x_RE3


        for i,each_x in enumerate([x_origin,x_RE1,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16,]):
                                   # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps




    return adv
def sharp_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    if opt.lr:
        alpha = opt.lr
    else:
        alpha =   eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda(device)
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)

        global  randomseed
        randomseed += 1

        x_RE1=Edge_Enhance(X_pert+ noise )


        seed_torch(0)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_RE1
        x_nes_cs_4 = 1 / 4 * x_RE1
        x_nes_cs_8 = 1 / 8 * x_RE1
        x_nes_cs_16 = 1 / 16 * x_RE1
        #
        # x_nes_RE2_2=1/2*x_RE2
        # x_nes_RE2_4 = 1 / 4 * x_RE2
        # x_nes_RE2_8 = 1 / 8 * x_RE2
        # x_nes_RE2_16 = 1 / 16 * x_RE2
        #
        # x_nes_RE3_2=1/2*x_RE3
        # x_nes_RE3_4 = 1 / 4 * x_RE3
        # x_nes_RE3_8 = 1 / 8 * x_RE3
        # x_nes_RE3_16 = 1 / 16 * x_RE3


        for i,each_x in enumerate([x_origin,x_RE1,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16,]):
                                   # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps






    return adv
global image_loss_record

global df

image_loss_record=0

def attack(model, img, label,model_name,benign=False,method='benign',images_n=None,opt=None):
    """generate adversarial images"""

    using_aux_logit = not 'resnet' in model_name
    if benign:
        return img
    elif method=='SI_DI_TI_MIFGSM':
        return SI_DI_TI_MIFGSM(model, img, label, using_aux_logit)
    elif method=='CS-DST':
        return CS_DST(model, img, label, using_aux_logit)
    elif method=='GS-DST':
        return greyschale_TI_FGSM(model, img, label,using_aux_logit)
    elif method=='ultimate_combo' :
        return grid_search(model, img, images_n, label, using_aux_logit, 55)
    elif method=='admix_DI_TI_FGSM':
        return admix_TI_FGSM(model, img, label, using_aux_logit)
    elif method=='CJ-DST':
        return CJ_DST(model, img, label, using_aux_logit)
    elif method=='SI_DI_TI_MIFGSM':
        return  SI_DI_TI_MIFGSM(model, img, label, using_aux_logit)
    elif method=='FPCA_DST':
        return FPCA_DST(model, img, label, using_aux_logit)
    elif method.startswith('grid_search'):
        order = int(method[11:])
        return grid_search(model, img, images_n, label, using_aux_logit, order)
    elif method.startswith('ultimateMIFGSM'):
        order=int(method[14:])
        return utimate_MIFGSM(model, img,images_n, label,using_aux_logit,order)

    elif method=='fPCA-DST':
        return fpca_TI_FGSM(model, img, label, using_aux_logit)
    elif method=='RE-DST':
        return RE_TI_FGSM(model, img, label, using_aux_logit)
    elif method=='cutmix-DST':
        return cutmix_TI_FGSM(model, img, label, using_aux_logit)
    elif method=='CutOut-DST':
        return cutout_TI_FGSM(model, img, label, using_aux_logit)
    elif method=='AutoAugment-DST':
        return autoaugment_TI_FGSM(model, img, label,using_aux_logit)
    elif method=='Sharpness-DST':
        return sharp_TI_FGSM(model, img, label,using_aux_logit)



    elif method.startswith("ultimate-combo-gen"):
        order = 271559200529755 # int(method[25:])
        if opt.dataset=="cifar-10":
            return CIFARAdvancedUltimateAdmixFGSM(model, img, images_n, label, using_aux_logit, order,opt,device=device)

        
        return AdvancedUltimateAdmixFGSM(model, img, images_n, label, using_aux_logit, order,opt,device=device)


    elif method=='benign':
        return img, None






global module_name
global features_blobs
module_name = []
features_blobs=[]

def hook_feature(module, input, output):
    global module_name
    global features_blobs
    features_blobs.append(output[0].data.cpu().numpy())
    module_name.append(module)

def main():
    index=0
    transforms = T.Compose([T.ToTensor()])
    transforms2 = T.Compose([T.Resize(299), T.ToTensor()])

    df_res = pd.DataFrame(
        columns=['dst_benign_success',  'name'])
    # Load inputs
    inputs = ImageNet(opt.input_dir, opt.input_csv, transforms)
    neural = ImageNet('./neural-style-pt/output', opt.input_csv, transforms2)
    data_loader = DataLoader(inputs, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # transforms.Resize(224)
    data_loader_neural = DataLoader(neural, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    input_num = len(inputs)

    # Create models
    models = get_models(list_nets, opt.model_dir)



    # Start iteration
    method=opt.method

    surrogate_models = list_nets

    methods=[method]

    global use_model
    accumulatedTime = 0

    for skipnumber,surrogate in enumerate(surrogate_models):
        use_model=surrogate
        global df

        global image_loss_record
        image_loss_record = 0
        for method in methods:
            # Initialization parameters
            correct_num = {}
            cos_similarity = {}
            logits = {}
            for net in list_nets:
                correct_num[net] = 0
                cos_similarity[net] = 0


            global  imageindex
            imageindex=0
            if opt.dataset=="imagenet":

                for (images, filename, label),(images_n, filename_n, label_n) in tqdm(zip(data_loader,data_loader_neural)):
                    print(imageindex)
                    imageindex+=1
                    label = label.cuda(device)
                    images = images.cuda(device)
                    images_n =images_n.cuda(device)
                    currenttime = time.time()


                    if method=="NeuTrans-DST":
                        adv_img, grad_surrogate=neural_TDSMFGSM(models[surrogate], images, images_n,label,not 'resnet' in surrogate)
                    elif method=="neural_transfer_benign":
                        adv_img, grad_surrogate=images_n,None
                    else:

                        adv_img, grad_surrogate = attack(models[surrogate], images, label, surrogate, method=method,images_n=images_n,opt=opt)
                    accumulatedTime += (time.time() - currenttime)

                    # with torch.no_grad():

                    for net in list_nets:
                        models[net].eval()
                        if opt.cosine:
                            adv_img.requires_grad = True
                            zero_gradients(adv_img)
                            using_aux_logit = not 'resnet' in net

                            logits[net] = models[net](adv_img)
                            loss = F.cross_entropy(logits[net][0], label)
                            if using_aux_logit:
                                loss += F.cross_entropy(logits[net][1], label)  # aux_logit
                            grad_target = autograd.grad(loss, adv_img, create_graph=True)[0].detach()
                            cos_similarity[net] += Cosine(grad_surrogate, grad_target).detach().cpu()
                            zero_gradients(adv_img)
                            torch.cuda.empty_cache()

                        else:
                            logits[net] = models[net](adv_img)



                        if (opt.batch_size == 1 or opt.parallel) and not (net in normal_net):
                            logits[net][0] = logits[net][0].reshape(-1, 1001)
                        correct_num[net] += (torch.argmax(logits[net][0], axis=1) != label).detach().sum().cpu()
                        # logging.info(f"net: {net}, trans:{correct_num[net]}")
            elif opt.dataset=="cifar-10":
                for images, label in tqdm(data_loader):
                    label = label.cuda(device)
                    images = images.cuda(device)

                    # Start Attack

                    currenttime = time.time()
                    adv_img = attack(models[surrogate].cuda(device), images, label, surrogate, method=method,opt=opt)
                    accumulatedTime += (time.time() - currenttime)
                    with torch.no_grad():
                        for net in list_nets:

                            if opt.dataset == 'imagenet':
                                logits[net] = models[net].cuda(device)((adv_img))
                                correct_num[net] += (torch.argmax(logits[net][0], axis=1) != label).detach().sum().cpu()
                            elif opt.dataset == 'cifar-10':

                                if  net=='PreActResNet':
                                    logits[net] = models[net].cuda(device)(adv_img)
                                else:
                                    logits[net] = models[net].cuda(device)(normalization(adv_img))

                                correct_num[net] += (torch.argmax(logits[net], axis=1) != label).detach().sum().cpu()
                            # print('{} attack {} using {}success rate: {:.2%}'.format(surrogate, net, method,
                            logging.info(f"net: {net}, trans:{correct_num[net]},time:{accumulatedTime}")
            # Print attack success rate

            for net in list_nets:
                df_res.loc[index,'name']=f'transferability_{surrogate}to{net}_{method}'
                df_res.loc[index, 'success_rate'] = round((correct_num[net]/input_num).item(),3)
                df_res.loc[index, 'time'] = round(accumulatedTime/input_num,3)
                if opt.cosine:
                    df_res.loc[index, 'cos_similarity'] = round((cos_similarity[net]/input_num).item(),3)
                index+=1
                # print('{} attack {} using {}success rate: {:.2%}'.format(surrogate,net,method, correct_num[net]/input_num))




            df_res.to_csv(f"./{opt.method}.csv")

if __name__ == '__main__':

    global randomseed
    randomseed = 0
    seed_torch(randomseed)
    # set_trace()
    main()
