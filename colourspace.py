from numpy import linalg
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import  torch
from torch import tensor
#
import os

def group_pca_color_augmention(image_array: tensor):
    jitter1=torch.zeros_like(image_array)
    jitter2 = torch.zeros_like(image_array)
    jitter3 = torch.zeros_like(image_array)
    for i,each_image in enumerate(image_array):
        jitter1[i],jitter2[i],jitter3[i]=pca_color_augmention(each_image)
    newimage1=image_array+jitter1
    newimage2 = image_array + jitter2
    newimage3 = image_array + jitter3
    torch.clamp(newimage1, 0.0, 1.0)
    torch.clamp(newimage2, 0.0, 1.0)
    torch.clamp(newimage3, 0.0, 1.0)
    return  newimage1,newimage2,newimage3

def pca_color_augmention(image_array: tensor):
    '''
    image augmention: PCA jitter
    :param image_array: tensor
    :return img2: PCA-jitter enhanced noise
    '''
    img1=image_array.clone().cpu().detach().numpy()
    mean = img1.mean(axis = 1).mean(axis = 1)
    std = img1.reshape((3, -1)).std(1)  

    img1 = (img1 - np.reshape(mean,(3,1,1))) / (np.reshape(std,(3,1,1)))


    img1 = img1.reshape((-1, 3))


    cov = np.cov(img1, rowvar=False)

    eigValue, eigVector = linalg.eig(cov)


    rand1 = np.array([random.normalvariate(0, 0.2) for i in range(3)])
    seed_torch(1)
    rand2 = np.array([random.normalvariate(0, 0.2) for i in range(3)])
    seed_torch(2)
    rand3 = np.array([random.normalvariate(0, 0.2) for i in range(3)])
    seed_torch(3)
    jitter = np.dot(eigVector, eigValue * rand1)
    jitter =  np.reshape(jitter,(3,1,1))
    jitter=torch.from_numpy(jitter).to(image_array.device)
    jitter2 = np.dot(eigVector, eigValue * rand2)
    jitter2 =  np.reshape(jitter2,(3,1,1))
    jitter2=torch.from_numpy(jitter2).to(image_array.device)

    jitter3 = np.dot(eigVector, eigValue * rand3)
    jitter3 =  np.reshape(jitter3,(3,1,1))
    jitter3=torch.from_numpy(jitter3).to(image_array.device)


    return jitter,jitter2,jitter3


def show_image(image_array):
    for _ in range(8):
        ax = plt.subplot(241 + _)
        ax.imshow(pca_color_augmention(image_array))
        ax.axis('off')
    plt.show()
    plt.savefig("jitter.jpg")
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

