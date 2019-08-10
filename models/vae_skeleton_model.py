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
    meanpose_path = './mixamo_data/meanpose_with_view.npy'
    stdpose_path = './mixamo_data/stdpose_with_view.npy'

    meanpose = np.load(meanpose_path)
    stdpose = np.load(stdpose_path)

    return torch.Tensor(meanpose).cuda(), torch.Tensor(stdpose).cuda()

def normalize_motion_inv(motion, mean_pose, std_pose):
    if len(motion.shape) == 2:
        motion = motion.reshape(-1, 2, motion.shape[-1])
    return motion * std_pose.unsqueeze(2) + mean_pose.unsqueeze(2)

def trans_motion_inv(motion, sx, sy, velocity=None):
    # seems like hip is the center, still need to check
    # motion_inv = torch.cat([motion[:,:8], torch.zeros((motion.shape[0], 1, 2, motion.shape[-1])).cuda(), motion[:,8:-1]], 1)
    motion_inv = motion[:,:-1]
    
    motion_inv = motion_inv[...,0]
    # restore centre position
    for i in range(motion.shape[0]):
        # pdb.set_trace()
        # print(motion_inv[i])
        for j in range(14):
            if motion_inv[i,j,0] < -1000:
                motion_inv[i,j] = -2
            else:
                motion_inv[i,j,0] = motion_inv[i,j,0] + sx[i]
                motion_inv[i,j,1] = motion_inv[i,j,1] + sy[i]
        # print(motion_inv[i])
        # pdb.set_trace()
    return motion_inv

class Vae_Skeleton_Model(BaseModel):
    def name(self):
        return 'Vae_Skeleton_Model'

    def __init__(self, opt):
        super(Vae_Skeleton_Model, self).__init__()
        BaseModel.initialize(self, opt)
        self.vae_net = AutoEncoder3x(mot_en_channels, body_en_channels, 
            view_en_channels, de_channels)  

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.vae_net, 'netSK', which_epoch)
        self.vae_net.eval()
        for param in self.vae_net.parameters():
            param.requires_grad = False
        # still need interpolation variables
        # self.interpm = nn.Conv2d(2, 1, 1) 
        # self.interpv = nn.Conv2d(2, 1, 1)
        self.alpha_m = Variable(torch.randn(1, device="cuda"), requires_grad=True)
        self.alpha_v = Variable(torch.randn(1, device="cuda"), requires_grad=True)

        # self.alpha_m.cuda()
        # self.alpha_v.cuda()
        self.mean_pose, self.std_pose = get_meanpose()

        self.w, self.h, self.scale = 352, 512, 2

    def preprocess_motion2d(motion):
        motion_trans = normalize_motion(trans_motion2d(motion), self.mean_pose, self.std_pose)
        motion_trans = motion_trans.reshape((-1, motion_trans.shape[-1]))
        return torch.Tensor(motion_trans).unsqueeze(0)

    def forward(self, input1, input2, center):
        m1 = self.vae_net.mot_encoder(input1)
        m2 = self.vae_net.mot_encoder(input2)
        b1 = self.vae_net.body_encoder(input1[:, :-2, :])
        v1 = self.vae_net.view_encoder(input1[:, :-2, :])
        v2 = self.vae_net.view_encoder(input2[:, :-2, :])

        # may differ if we want more sequence
        m_mix = (1 - (0.5+self.alpha_m)) * m1 + (0.5+self.alpha_m) * m2
        v_mix = (1 - (0.5+self.alpha_m)) * v1 + (0.5+self.alpha_m) * v2
        # m_mix = m2
        # v_mix = v2
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