import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks

import torch.nn as nn

class Skeleton_Model(BaseModel):
    def name(self):
        return 'Skeleton_Model'

    def __init__(self, opt):
        super(Skeleton_Model, self).__init__()
        BaseModel.initialize(self, opt)
        self.main = nn.Sequential(
            nn.Conv2d(36, 18, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
    	# input angle and joints
        # print(input.shape)
        return self.main(input)