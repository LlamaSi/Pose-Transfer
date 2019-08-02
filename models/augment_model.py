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
import matplotlib.pyplot as plt
import pdb

class AugmentModel(BaseModel):
    def name(self):
        return 'AugmentModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # create and initialize network
        opt.model = 'PATN'
        self.main_model = create_model(opt)
        from .inter_skeleton_model import InterSkeleton_Model
        self.skeleton_net = InterSkeleton_Model(opt).cuda()

        print('---------- Networks initialized -------------')
        networks.print_network(self.skeleton_net)
        print('-----------------------------------------------')

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            # self.load_network(self.skeleton_net, 'netSK', which_epoch)

        if self.isTrain:
            self.skeleton_lr = opt.lr2

            if opt.L1_type_sk == 'origin':
                self.criterionL1 = torch.nn.L1Loss()

                self.optimizers = []
                self.schedulers = []

                # initialize optimizers
                self.optimizer_SK = torch.optim.Adam(self.skeleton_net.parameters(), lr=self.skeleton_lr , betas=(opt.beta2, 0.999))

                self.optimizers.append(self.optimizer_SK)
                self.optimizers = self.optimizers + self.main_model.optimizers

                for optimizer in self.optimizers:
                    self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def forward_aug(self, input):
        # update main model
        # the order of joints for a 3d is angles
        aug_input = torch.stack((input['BP1_3d_angle'].float().cuda(), input['BP2_3d_angle'].float().cuda()), 1)
        offset, limbs = input['offset'].cuda(), input['limbs'].cuda()

        BP_aug_3d = torch.squeeze(self.skeleton_net(aug_input))
        # BP_aug_3d = aug_input[:,0]
        BP_aug_xyz = xyz_to_anglelimb.anglelimbtoxyz2(offset, BP_aug_3d, limbs)
        BP_aug_hm = torch.zeros([BP_aug_xyz.size(0), BP_aug_xyz.size(1), 2]).cuda()
        BP_aug_hm[:,:,0] = torch.clamp(BP_aug_xyz[:,:,0], min=0, max=176)
        BP_aug_hm[:,:,1] = torch.clamp(BP_aug_xyz[:,:,1], min=0, max=256)

        self.input_BP_aug = xyz_to_anglelimb.cords_to_map(BP_aug_hm, [256, 176])

        # for exp on top
        self.input_BP_aug[:,9:11] = input['BP2'][:,9:11] # = 0
        self.input_BP_aug[:,12:14] = input['BP2'][:,12:14] # = 0

        # missing 0, 14:17
        self.input_BP_aug[:,0] = input['BP2'][:,0]
        self.input_BP_aug[:,14:] = input['BP2'][:,14:]

        main_input = input.copy()
        main_input['BP2'] = self.input_BP_aug

        self.main_model.set_input(main_input)
        # get fake_b 
        self.main_model.forward()
        # should add skeleton loss inside main_model

        self.main_model.opt.with_D_PP = 1
        self.main_model.opt.with_D_PB = 0
        self.main_model.opt.L1_type = 'None'

        self.main_model.optimize_parameters(False)
        # pdb.set_trace()
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

        # print(self.skeleton_lr )
        # for w in list(self.skeleton_net.parameters()):
        #     print('grad')
        #     print(w.grad)

        # for w in list(self.skeleton_net.parameters()):
        #     print('weight')
        #     print(w)
        self.optimizer_SK.step()
        # self.optimizer_SK.zero_grad()

        return fake_b

    def get_current_errors(self):
        return self.main_model.get_current_errors()

    def get_current_visuals(self):
        height, width = self.main_model.input_P1.size(2), self.main_model.input_P1.size(3)
        aug_pose = util.draw_pose_from_map(self.input_BP_aug.data)[0]
        part_vis = self.main_model.get_current_visuals()['vis']
        vis = np.zeros((height, width*7, 3)).astype(np.uint8) #h, w, c
        vis[:,:width*5,:] = part_vis
        vis[:,width*5:width*6,:] = aug_pose
        vis[:,width*6:width*7,:] = self.fake_aug*256

        ret_visuals = OrderedDict([('vis', vis)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.skeleton_net, 'netSK', label, self.gpu_ids)
        self.main_model.save(label)