from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch
from models.xyz_to_anglelimb import absolute_angles
import debugger
import pdb
import os

import matplotlib.pyplot as plt

def anglelimbtoxyz(offset, absolute_angles, limbs):
    res_3d = np.zeros([1, 16, 3])
    
    norm_direction = np.cos(absolute_angles)
    limbs = np.tile(limbs, (1,1,3))
    direction = limbs * norm_direction

    res_3d[:, 0] = offset
    res_3d[:, 1] = res_3d[:, 0] - direction[:, 0]
    res_3d[:, 2] = res_3d[:, 1] - direction[:, 1]
    res_3d[:, 6] = res_3d[:, 2] - direction[:, 2]
    res_3d[:, 3] = res_3d[:, 6] - direction[:, 3]
    res_3d[:, 4] = res_3d[:, 3] - direction[:, 4]
    res_3d[:, 5] = res_3d[:, 4] - direction[:, 5]
    res_3d[:, 8] = res_3d[:, 6] - direction[:, 12]
    res_3d[:, 9] = res_3d[:, 8] - direction[:, 13]
    res_3d[:, 13] = res_3d[:, 8] - direction[:, 9]
    res_3d[:, 14] = res_3d[:, 13] - direction[:, 10]
    res_3d[:, 15] = res_3d[:, 14] - direction[:, 11]
    res_3d[:, 12] = res_3d[:, 8] + direction[:, 8]
    res_3d[:, 11] = res_3d[:, 12] + direction[:, 7]
    res_3d[:, 10] = res_3d[:, 11] + direction[:, 6]

    return res_3d

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

P1_name = 'fashionMENTees_Tanksid0000404701_7additional.jpg'

P1_path = os.path.join(dir_P, P1_name) # person 1
BP1_path = os.path.join(dir_K, P1_name + '.npy') # bone of person 1

# person 1 & 2 3d pose(x,y,z)
P13d_path = os.path.join(dir_3d, P1_name.replace('.jpg', '.npy')) # person 1

BP1_3d_dict = np.load(P13d_path, allow_pickle=True).item()
BP1_3d, bias1 = BP1_3d_dict['pred_3d'], BP1_3d_dict['bias']
BP1_3d = rescale3d(BP1_3d, bias1)


BP1_img = np.load(BP1_path) # h, w, c
P1_img = Image.open(P1_path).convert('RGB')

absolute_angles1, limbs1, offset = absolute_angles(BP1_3d)

BP_aug_xyz = anglelimbtoxyz(offset, absolute_angles1, limbs1)
pdb.set_trace()