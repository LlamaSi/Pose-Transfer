import numpy as np
import torch
import os
from collections import OrderedDict
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss

import sys
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

from torch.autograd import Variable
from . import cpm_model
from . import cpm_test

import pdb

class TransferModel(BaseModel):
    def name(self):
        return 'TransferModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize

        self.opt = opt
        self.centermap = cpm_test.get_centermap(nb, 256, 176)
        cpm_model_path = '/home/wenwens/Documents/HumanPose/Pose-Transfer/checkpoints/cpm/cpm_latest.pth.tar'
        self.cpm_model = cpm_test.construct_model(cpm_model_path)

        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc]

        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_downsampling=opt.G_n_downsampling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc+opt.BP_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            if opt.pose_loss:
                self.pose_loss = torch.nn.MSELoss()
            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            else:
                raise Excption('Unsurportted type of L1!')
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
        print('-----------------------------------------------')

    def set_input(self, input):
        self.input_P1, self.input_BP1 = input['P1'], input['BP1']
        self.input_P2, self.input_BP2 = input['P2'], input['BP2']
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]

        if len(self.gpu_ids) > 0:
            self.input_P1 = self.input_P1.cuda()
            self.input_BP1 = self.input_BP1.cuda()
            self.input_P2 = self.input_P2.cuda()
            self.input_BP2 = self.input_BP2.cuda()
            
    def forward(self, input_BP_res):
        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.fake_p2 = self.netG(G_input)
        
        self.cpm_model.eval()
        # self.ds_BP2 = F.upsample(self.input_BP2, scale_factor=0.125)
        self.ds_BP2 = torch.nn.MaxPool2d(8,8)(input_BP_res)
        
        _, _, _, _, _, heat6 = self.cpm_model(self.fake_p2, self.centermap)
        heat6 = heat6[:,1:] # 0 - 14
        # pdb.set_trace()
        cores = [10,9,8,11,12,13,4,3,2,5,6,7,1]
        
        self.heat6 = torch.zeros(heat6.shape).cuda()
        # if range(13), no include neck
        for i in range(12):
            self.heat6[:,cores[i]] = heat6[:,i]

        self.heat6 = self.heat6[:,2:]

    def test(self):
        with torch.no_grad():
            G_input = [self.input_P1,
                       torch.cat((self.input_BP1, self.input_BP2), 1)]
            self.fake_p2 = self.netG(G_input)
        # self.cpm_model.eval()
        # # self.ds_BP2 = F.upsample(self.input_BP2, scale_factor=0.125)
        # self.ds_BP2 = torch.nn.MaxPool2d(8,8)(self.input_BP2)
        # self.ds_BP2 = self.ds_BP2.flatten(start_dim=2)
        # # pdb.set_trace()
        # _, _, _, _, _, heat6 = self.cpm_model(self.fake_p2, self.centermap)
        # heat6 = heat6[:,1:]
        # cores = [10,9,8,11,12,13,4,3,2,5,6,7,1]
        
        # self.heat6 = torch.zeros(heat6.shape).cuda()
        # for i in range(13):
        #     self.heat6[:,cores[i]] = heat6[:,i]

        # self.heat6 = self.heat6.flatten(start_dim=2)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)
        self.loss_D_PB = loss_D_PB.item()

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake_p2, self.input_P1), 1).data )
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)
        self.loss_D_PP = loss_D_PP.item()

    def backward_G(self, infer=False):
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        # L1 loss
        if self.opt.L1_type != 'None':
            # if we have target results, then we can add L1 loss
            # either peceptual of normal L1
            if self.opt.L1_type == 'l1_plus_perL1' :
                # print("self.opt.L1_type={}".format(self.opt.L1_type))
                losses = self.criterionL1(self.fake_p2, self.input_P2)
                self.loss_G_L1 = losses[0]
                self.loss_originL1 = losses[1].item()
                self.loss_perceptual = losses[2].item()
            else:
                self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2) * self.opt.lambda_A
        else:
            # otherwise L1 loss is zero
            self.loss_G_L1 = Variable(torch.cuda.FloatTensor([0]))

        pair_L1loss = self.loss_G_L1
        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN

        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss = pair_L1loss + pair_GANloss
        else:
            pair_loss = pair_L1loss

        if self.opt.pose_loss:
            # if compute pose loss
            t = Variable(self.ds_BP2[:, 2:14], requires_grad=False)
            # lambda_pose
            # heat_weight = 46 * 46 / 1.0
            pl = self.pose_loss(torch.clamp(self.heat6, min=0, max=1), t)*self.opt.lambda_pose
            self.pl = pl
            # print(self.pl)

        # just to assign the result for print error
        self.pair_L1loss = pair_L1loss.item()

        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.item()

        if not infer:
            # if want to update G
            # add pose loss
            # pair loss is the total loss
            pair_loss += pl
            # pair_loss = pl
            pair_loss.backward()
        else:
            # currently no pose loss for target
            # pair_loss += pl
            return pair_loss

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.backward_G(infer=False)
        self.optimizer_G.step()
        
        # choose not to update discriminator for now
        # D_P
        # if self.opt.with_D_PP:
        #     for i in range(self.opt.DG_ratio):
        #         self.optimizer_D_PP.zero_grad()
        #         self.backward_D_PP()
        #         self.optimizer_D_PP.step()

        # # D_BP
        # if self.opt.with_D_PB:
        #     for i in range(self.opt.DG_ratio):
        #         self.optimizer_D_PB.zero_grad()
        #         self.backward_D_PB()
        #         self.optimizer_D_PB.step()


    def get_current_errors(self):
        ret_errors = OrderedDict([ ('pair_L1loss', self.pair_L1loss)])
        # if self.opt.with_D_PP:
        #     ret_errors['D_PP'] = self.loss_D_PP
        # if self.opt.with_D_PB:
        #     ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_PB or self.opt.with_D_PP:
            ret_errors['pair_GANloss'] = self.pair_GANloss

        if self.opt.L1_type == 'l1_plus_perL1':
            ret_errors['origin_L1'] = self.loss_originL1
            ret_errors['perceptual'] = self.loss_perceptual
        ret_errors['pose_loss'] = self.pl

        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        fake_p2 = util.tensor2im(self.fake_p2.data)

        vis = np.zeros((height, width*5, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_BP1
        vis[:, width*2:width*3, :] = input_P2
        vis[:, width*3:width*4, :] = input_BP2
        vis[:, width*4:, :] = fake_p2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)

