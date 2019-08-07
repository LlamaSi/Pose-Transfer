import torch
from .base_model import BaseModel
from . import networks

import torch.nn as nn

class InterSkeleton_Model(BaseModel):
    def name(self):
        return 'InterSkeleton_Model'

        # input shape (b, 2, 14, 3)

    def __init__(self, opt):
        super(InterSkeleton_Model, self).__init__()
        BaseModel.initialize(self, opt)
        self.main = nn.Sequential(
            nn.Conv2d(2, 1, 1)
        )

    def forward(self, input):
        out = self.main(input)
        return out