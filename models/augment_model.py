import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from .models import create_model
from . import networks
import util.util as util
from collections import OrderedDict
import itertools
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss
import numpy as np

from . import xyz_to_anglelimb
import torch.nn.functional as F
from .vae_skeleton_model import Vae_Skeleton_Model, trans_motion_inv, normalize_motion_inv
import matplotlib.pyplot as plt
import pdb

def cords_to_map(cords, img_size, sigma=6):
    MISSING_VALUE = -1
    result = torch.zeros([cords.size(0), 18, 256, 176])
    for i, points in enumerate(cords):
        for j in range(18):
            point = points[j]
            if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
                continue
            xx, yy = torch.meshgrid(torch.arange(img_size[0], dtype=torch.int32).cuda(), torch.arange(img_size[1],dtype=torch.int32).cuda())
            xx = xx.float()
            yy = yy.float()
            res = torch.exp(-((yy - point[1]) ** 2 + (xx - point[0]) ** 2) / (2 * sigma ** 2))
            result[i, j] = res

    return result

class AugmentModel(BaseModel):
    def name(self):
        return 'AugmentModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # create and initialize network
        opt.model = 'PATN'
        self.opt = opt
        self.main_model = create_model(opt)
        self.skeleton_net = Vae_Skeleton_Model(opt).cuda()

        print('---------- Networks initialized -------------')
        networks.print_network(self.skeleton_net)
        print('-----------------------------------------------')

        if self.isTrain:
            self.skeleton_lr = opt.lr2

            if opt.L1_type_sk == 'origin':
                self.criterionL1 = torch.nn.L1Loss()

                self.optimizers = []
                self.schedulers = []

                # initialize optimizers
                self.optimizer_SK = torch.optim.Adam([self.skeleton_net.alpha_m, self.skeleton_net.alpha_v], lr=self.skeleton_lr , betas=(opt.beta2, 0.999))

                # need to check whether parameter contains abundant ones
                self.optimizers.append(self.optimizer_SK)
                self.optimizers = self.optimizers + self.main_model.optimizers

                for optimizer in self.optimizers:
                    self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def forward_aug(self, input):
        # with mid hip at 15
        input1, input2 = input['K1'].squeeze(1), input['K2'].squeeze(1)
        F2 = input['F2'].cuda().float()
        # still need to preprocess, ie mean std
        input1 = input1.repeat((1,1,16)).cuda().float()
        input2 = input2.repeat((1,1,16)).cuda().float()
        # check output range
        center = input['C2'].cuda().float()
        # 14, no mid hip
        BP_aug_kpts = self.skeleton_net(input1, input2, center)
        neck = F2[:,1:2] #4,2
        aug_neck = BP_aug_kpts[:,1:2] # 4,2
        eye_ears = F2[:,2:] # 4, 4, 2
        # print(neck, aug_neck, eye_ears)

        aug_eye_ears = eye_ears - neck + aug_neck
        # print(aug_eye_ears)
        
        BP_aug_kpts_full = torch.cat([BP_aug_kpts, aug_eye_ears], 1)
        
        self.input_BP_aug = cords_to_map(BP_aug_kpts_full, (256, 176), sigma=3)
        self.input_BP_res = cords_to_map(BP_aug_kpts_full, (256, 176), sigma=3*8).cuda()

        for i in range(self.opt.batchSize):
            for j in range(4):
                if eye_ears[i,j,0] == -1 or eye_ears[i,j,1] == -1:
                    self.input_BP_aug[i, j+14] = 0
            if F2[i, 0, 0] == -1:
                self.input_BP_aug[i,0] = 0
        # pdb.set_trace()
        main_input = input.copy()
        main_input['BP2'] = self.input_BP_aug

        self.main_model.set_input(main_input)
        # get fake_b 
        self.main_model.forward(self.input_BP_res)
        # should add skeleton loss inside main_model

        self.main_model.opt.with_D_PP = 1
        self.main_model.opt.with_D_PB = 0
        self.main_model.opt.L1_type = 'None'

        # update main model
        self.main_model.optimize_parameters()

        self.fake_aug = self.main_model.fake_p2[0].cpu().detach().numpy().transpose(1,2,0).copy()

        return self.main_model.fake_p2

    def forward_target(self, input):
        # augment skeleton model
        self.main_model.set_input(input)
        # get fake_b 
        self.main_model.test()

        fake_b = self.main_model.fake_p2

        self.main_model.opt.with_D_PP = 1
        self.main_model.opt.with_D_PB = 1
        self.main_model.opt.L1_type = 'l1_plus_perL1'

        self.skeleton_net.train()

        pair_loss = self.main_model.backward_G(infer=True)
        pair_loss.backward()

        self.optimizer_SK.step()
        self.optimizer_SK.zero_grad()

        return fake_b

    def get_current_errors(self):
        return self.main_model.get_acc_error()

    def get_current_visuals(self):
        height, width = self.main_model.input_P1.size(2), self.main_model.input_P1.size(3)
        aug_pose = util.draw_pose_from_map(self.input_BP_aug.data)[0]
        part_vis = self.main_model.get_current_visuals()['vis']
        vis = np.zeros((height, width*8, 3)).astype(np.uint8) #h, w, c

        vis[:,:width*5,:] = part_vis
        vis[:,width*5:width*6,:] = aug_pose
        vis[:,width*6:width*7,:] = ((self.fake_aug + 1) / 2.0 * 255).astype(np.uint8)

        heatmap = F.upsample(self.main_model.heat6, scale_factor=8)
        heatmap = torch.cat([torch.zeros([self.opt.batchSize, 2, 256, 176]).cuda(), heatmap], 1)
        heatmap = heatmap.data
        vis[:,width*7:width*8,:] = util.draw_pose_from_map(heatmap.data)[0]
        
        # for i in range(12):
        #     plt.imshow(heatmap[i])
        #     plt.show()
        ret_visuals = OrderedDict([('vis', vis)])
        return ret_visuals

    def save(self, label):
        self.skeleton_net.save(label)
        self.main_model.save(label)


# input22 = input2.view((input2.shape[0], 15, 2, -1))
# # check if still need mapping
# # post process

# BP_aug_kpts = trans_motion_inv(normalize_motion_inv(input22, self.skeleton_net.mean_pose, 
#     self.skeleton_net.std_pose), sx=center[:,0:1,0], sy=center[:,1:2,0]) / 2
# # print(out.shape) # (b, 14, 2)
        