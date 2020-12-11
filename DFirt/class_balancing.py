'''
Balance the classes Expression Swap, Entire Face Synthesis and Attribute Manipulation.
'''

import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision
from torchvision import transforms
import torch.utils.data as data
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

data = ['face_synthesis', 'expression_swap', 'attribute_manipulation']

def augment(d, im_path):
    im = Image.open(join('train/' + d + '/', im_path))
    im = Image.fromarray(np.uint8(im))
    
    t = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])
    save_image(TF.to_tensor(t(im)), 'balanced/train/' + d + '/' + im_path.split('.')[0] + '_' + 'hflip' + '.jpg')
    t = transforms.Compose([transforms.RandomRotation(degrees=[30, 30])])
    save_image(TF.to_tensor(t(im)), 'balanced/train/' + d + '/' + im_path.split('.')[0] + '_' + 'r+30' + '.jpg')
    t = transforms.Compose([transforms.RandomRotation(degrees=[-30, -30])])
    save_image(TF.to_tensor(t(im)), 'balanced/train/' + d + '/' + im_path.split('.')[0] + '_' + 'r-30' + '.jpg')

for d in data:
    os.makedirs('balanced/train/' + d + '/')
    cnt = 0
    print('Augment class ' + d)
    for im_path in os.listdir('train/' + d + '/'):
        if im_path.split('.')[1] == 'jpg':
            augment(d, im_path)
            cnt += 1
            print('Number of images processed: ', cnt, end='\r')
        else:
            break
