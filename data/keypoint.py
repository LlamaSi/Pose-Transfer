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
from . import debugger

class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #keypoints
        self.dir_3d = os.path.join(opt.dataroot, opt.phase + '_3d_full')

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def rescale3d(self, pred_3d, bias):
        scale = 256
        res = pred_3d *scale
        res[:,:2] = res[:,:2] + bias
        return res

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name) # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # bone of person 1
        
        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name) # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2

        # person 1 & 2 3d pose(x,y,z)
        P13d_path = os.path.join(self.dir_3d, P1_name.replace('.jpg', '.npy')) # person 1
        P23d_path = os.path.join(self.dir_3d, P2_name.replace('.jpg', '.npy')) # person 2

        BP1_3d_dict, BP2_3d_dict = np.load(P13d_path, allow_pickle=True).item(), np.load(P23d_path, allow_pickle=True).item()
        BP1_3d, bias1 = BP1_3d_dict['pred_3d'], BP1_3d_dict['bias']
        BP2_3d, bias2 = BP2_3d_dict['pred_3d'], BP2_3d_dict['bias']

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_3d, BP2_3d = self.rescale3d(BP1_3d, bias1), self.rescale3d(BP2_3d, bias2)

        absolute_angles1, limbs1, offset = absolute_angles(BP1_3d)
        absolute_angles2, _, _ = absolute_angles(BP2_3d)

        BP1_img = np.load(BP1_path) # h, w, c
        BP2_img = np.load(BP2_path) 


        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0,1)
            
            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = np.array(BP1_img[:, ::-1, :]) # flip
                BP2_img = np.array(BP2_img[:, ::-1, :]) # flip

            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        else:
            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name, 'BP1_3d': BP1_3d, 'BP2_3d': BP2_3d, 'BP1_3d_angle': absolute_angles1, 'BP2_3d_angle': absolute_angles2,
                'limbs': limbs1, 'offset':offset}
                

    def __len__(self):
        if self.opt.phase == 'train':
            return 4002
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'
