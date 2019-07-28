import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks

import torch.nn as nn

class Skeleton_Model(BaseModel):
    def name(self):
        return 'Skeleton_Model'

        # input shape (2, 28, 3)

    def __init__(self, opt):
        super(Skeleton_Model, self).__init__()
        BaseModel.initialize(self, opt)
        self.main = nn.Sequential(
            nn.Linear(84, 42)
            # nn.Sigmoid()
        )

    def forward(self, input):
    	# input angle and joints
        input = input.view(4, -1)
        out = self.main(input)
        return out.view(-1, 14, 3)