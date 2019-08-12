from .vae_networks import AutoEncoder3x
from .base_model import BaseModel
from torch.autograd import Variable

import torch.nn as nn
import torch
import os
import numpy as np
import pdb

mot_en_channels = [30, 64, 96, 128]
body_en_channels = [28, 32, 48, 64, 16]
view_en_channels = [28, 32, 48, 64, 8]
de_channels = [152, 128, 64, 30]

def get_meanpose():
    meanpose_path = './deepfashion_meanpose_centered.npy'
    stdpose_path = './/deepfashion_stdpose_centered.npy'
    meanfashion_path = './deepfashion_meanpose.npy'

    meanpose = np.load(meanpose_path)[:15]*2
    stdpose = np.load(stdpose_path)[:15]*2
    meanpose_fashion = np.load(meanfashion_path)[:15]

    return torch.Tensor(meanpose).cuda(), torch.Tensor(stdpose).cuda(), torch.Tensor(meanpose_fashion).cuda()

def normalize_motion_inv(motion, mean_pose, std_pose):
    if len(motion.shape) == 2:
        motion = motion.reshape(-1, 2, motion.shape[-1])
    return motion * std_pose.unsqueeze(2) + mean_pose.unsqueeze(2)

def trans_motion_inv(motion, sx, sy,velocity=None):
    # remove mid hip at the end
    motion_inv = motion[:,:-1]
    
    motion_inv = motion_inv[...,0]
    for i in range(4):
        for j in range(14):
            # if torch.abs(motion_inv[i,j,0]) > 0.0001 or torch.abs(motion_inv[i,j,1]) > 0.0001:
                # pdb.set_trace()
            motion_inv[i,j,0] += sx[i,0]
            motion_inv[i,j,1] += sy[i,0]
            # else:
            #     motion_inv[i,j] += 2*mean_pose_fashion[j]
    # restore centre position
    # for i in range(motion.shape[0]):
    #     # pdb.set_trace()
    #     # print(motion_inv[i])
    #     for j in range(8):
    #         if torch.abs(motion_inv[i,j,0]) < 0.001:
    #             motion_inv[i,j] = -2
    #         else:
    #             motion_inv[i,j,0] = motion_inv[i,j,0] + sx[i]
    #             motion_inv[i,j,1] = motion_inv[i,j,1] + sy[i]
    #     # print(motion_inv[i])
    #     # pdb.set_trace()
    # this is only for top training
    motion_inv[:,9:11] = -2
    motion_inv[:,12:14] = -2
    return motion_inv

class Vae_Skeleton_Model(BaseModel):
    def name(self):
        return 'Vae_Skeleton_Model'

    def __init__(self, opt):
        super(Vae_Skeleton_Model, self).__init__()
        BaseModel.initialize(self, opt)
        self.vae_net = AutoEncoder3x(mot_en_channels, body_en_channels, 
            view_en_channels, de_channels)  

        self.alpha_m = Variable(torch.randn(1, device="cuda"), requires_grad=True)
        self.alpha_v = Variable(torch.randn(1, device="cuda"), requires_grad=True)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.vae_net, 'netSK', which_epoch)
            self.save_path = os.path.join(self.save_dir, 'alphas.npy')
            if os.path.exists(self.save_path):
                self.alpha_m.data = torch.Tensor(np.load(self.save_path)).cuda().data
                print(self.save_path)

        self.vae_net.eval()
        for param in self.vae_net.parameters():
            param.requires_grad = False

        self.mean_pose, self.std_pose, self.mean_pose_fashion = get_meanpose()

        self.w, self.h, self.scale = 352, 512, 2

    def forward(self, input1, input2, center):
        m1 = self.vae_net.mot_encoder(input1)
        m2 = self.vae_net.mot_encoder(input2)
        b1 = self.vae_net.body_encoder(input1[:, :-2, :])
        v1 = self.vae_net.view_encoder(input1[:, :-2, :])
        v2 = self.vae_net.view_encoder(input2[:, :-2, :])

        # may differ if we want more sequence
        m_mix = (1 - (0.5+self.alpha_m)) * m1 + (0.5+self.alpha_m) * m2
        v_mix = (1 - (0.5+self.alpha_m)) * v1 + (0.5+self.alpha_m) * v2
        # m_mix = 0.5*m1 + 0.5*m2
        # v_mix = 0.5*v1 + 0.5*v2
        # now get only one interpolation
        dec_input = torch.cat([m_mix, b1.repeat(1, 1, m1.shape[-1]), v_mix.repeat(1, 1, m1.shape[-1])], dim=1)
        
        out = self.vae_net.decoder(dec_input)
        out = out.view((out.shape[0], 15, 2, -1))
        # check if still need mapping
        # post process
        
        out = trans_motion_inv(normalize_motion_inv(out, self.mean_pose, 
            self.std_pose), sx=center[:,0:1,0], sy=center[:,1:2,0])
        # print(out.shape) # (b, 14, 2)

        return out / 2 # rescale

    def save(self, label):
        if False:
            self.save_network(self.vae_net,  'net_vae')
        np.save(self.save_path, self.alpha_m.item())