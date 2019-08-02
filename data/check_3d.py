import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch
from models.xyz_to_anglelimb import absolute_angles
import debugger

def rescale3d(pred_3d, bias):
    scale = 256
    res = pred_3d *scale
    res[:,:2] = res[:,:2] + bias
    return res

dataroot = './fashion_data'
phase = 'train'
dir_P = os.path.join(dataroot, phase) #person images
dir_K = os.path.join(dataroot, phase + 'K') #keypoints
dir_3d = os.path.join(dataroot, phase + '_3d_full')

P1_name = 'fashionMENTees_Tanksid0000404701_1front.jpg'

P1_path = os.path.join(dir_P, P1_name) # person 1
BP1_path = os.path.join(dir_K, P1_name + '.npy') # bone of person 1

# person 1 & 2 3d pose(x,y,z)
P13d_path = os.path.join(dir_3d, P1_name.replace('.jpg', '.npy')) # person 1

BP1_3d_dict = np.load(P13d_path, allow_pickle=True).item()
BP1_3d, bias1 = BP1_3d_dict['pred_3d'], BP1_3d_dict['bias']
BP1_3d = rescale3d(BP1_3d, bias1)


BP1_img = np.load(BP1_path) # h, w, c
P1_img = Image.open(P1_path).convert('RGB')


debugger1 = debugger.Debugger()
debugger1.add_img(Image.open(P1_path))
# debugger1.add_img(Image.open(BP1_img))
# debugger1.add_point_2d(pred, (255, 0, 0))
debugger1.add_point_3d(BP1_3d, 'b')
debugger1.show_3d()

# BP1_3d = rescale3d(BP1_3d, bias1)

# BP1 = torch.from_numpy(BP1_img).float() #h, w, c
# BP1 = BP1.transpose(2, 0) #c,w,h
# BP1 = BP1.transpose(2, 1) #c,h,w 
