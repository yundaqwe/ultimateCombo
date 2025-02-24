import torch
import copy
import torch.nn as nn
# from torch.autograd.gradcheck import zero_gradients

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.distributions import uniform
def zero_gradients(x):
    if x.grad is not None:
        x.grad.zero_()


    return x
from ipdb import set_trace
def regularizer(inputs, targets, net, h=3.,device=0,criterion=nn.CrossEntropyLoss()):
    '''
    Regularizer term in CURE
    '''
    # if len(targets.shape)==1:
    #     targets=targets[0]

    z, norm_grad = _find_z(inputs, targets, h,net, device,criterion)

    inputs.requires_grad_()
    outputs_pos = net.eval()(inputs + z)
    outputs_pos[0]=torch.unsqueeze(outputs_pos[0],0 ) if len(outputs_pos[0].shape)==1 else outputs_pos[0]
    outputs_orig =net.eval()(inputs)
    outputs_orig[0]=torch.unsqueeze(outputs_orig[0],0 ) if len(outputs_orig[0].shape)==1 else outputs_orig[0]
    loss_pos = criterion(outputs_pos[0], targets)
    loss_orig =criterion(outputs_orig[0], targets)
    grad_diff = \
    torch.autograd.grad((loss_pos - loss_orig), inputs,
                        create_graph=True)[0]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    net.zero_grad()

    #abs is not used in original repo
    return torch.sum(torch.abs( reg)) / float(inputs.size(0)), norm_grad
def _find_z(inputs, targets, h,net,device=0,criterion=nn.CrossEntropyLoss()):
    '''
    Finding the direction in the regularizer
    '''
    inputs.requires_grad_()

    outputs = net.eval()(inputs)
    outputs[0]  = torch.unsqueeze(outputs[0], 0) if len(outputs[0].shape) == 1 else outputs[0]
    loss_z = criterion(outputs[0], targets)

    loss_z.backward()
    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.
    z = 1. * (h) * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)
    zero_gradients(inputs)
    net.zero_grad()

    return z, norm_grad
def rand_bbox(size, lam=2):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2