'''
Augment dataset to create blends between reals and fakes.
'''

import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torchvision
from torchvision import transforms
import torch.utils.data as data
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

def gaussian_blur(img):
    ksize = np.random.choice([7, 9])
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image, (ksize, ksize), 10)
    return Image.fromarray(np.uint8(image_blur))

data = ['identity_swap', 'face_synthesis', 'expression_swap', 'attribute_manipulation', 'real']

def augment(d, im_path):
    im = Image.open(join('train/' + d + '/', im_path))
    im = Image.fromarray(np.uint8(im))
    
    t = [transforms.ColorJitter(brightness=[0.5, 2], contrast=[0.5, 2], saturation=[0.1, 0.5], hue=[-0.1, 0.1]),
         transforms.Grayscale(num_output_channels=3),
         transforms.RandomAffine(degrees=[-30, 30], translate=[0.1, 0.2], scale=[1, 1.2]),
         transforms.RandomPerspective(distortion_scale=0.2, p=1, interpolation=3),
         transforms.Compose([transforms.ToTensor(),
                             transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=True),
                             transforms.ToPILImage()]),
         transforms.Lambda(gaussian_blur)]
    
    num_t = np.random.randint(1, 6+1)
    for i in range(num_t):
        t_current = transforms.RandomChoice(t)
        im = t_current(im)
    
    save_image(TF.to_tensor(im), 'augmented/train/' + d + '/' + im_path.split('.')[0] + '_' + 'augmented' + '.jpg')

for d in data:
    #os.makedirs('augmented/train/' + d + '/')
    cnt = 0
    print('Augment class ' + d)
    for im_path in os.listdir('train/' + d + '/'):
        if im_path.split('.')[-1] == 'jpg':
            augment(d, im_path)
            cnt += 1
            print('Number of images processed: ', cnt, end='\r')
        else:
            break
