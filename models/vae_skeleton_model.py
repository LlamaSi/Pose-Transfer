from .vae_networks import AutoEncoder3x
import torch.nn as nn
import torch

mot_en_channels = [30, 64, 96, 128]
body_en_channels = [28, 32, 48, 64, 16]
view_en_channels = [28, 32, 48, 64, 8]
de_channels = [152, 128, 64, 30]

def get_meanpose():
	pass

def normalize_motion_inv(motion, mean_pose, std_pose):
    if len(motion.shape) == 2:
        motion = motion.reshape(-1, 2, motion.shape[-1])
    return motion * std_pose[:, :, np.newaxis] + mean_pose[:, :, np.newaxis]

def trans_motion_inv(motion, sx=256, sy=256, velocity=None):
    if velocity is None:
        velocity = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:-1]]

    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    for i in range(motion.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum
    centers += np.array([[sx], [sy]])

    return motion_inv + centers.reshape((1, 2, -1))

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
            self.load_network(self.skeleton_net, 'netSK', which_epoch)
        	net.eval()

        # still need interpolation variables
        self.interpm = nn.Conv2d(2, 1, 1) 
        self.interpv = nn.Conv2d(2, 1, 1)
        self.mean_pose, std_pose = get_meanpose()

    def forward(self, input):
        m1 = self.vae_net.mot_encoder(input1)
	    m2 = self.vae_net.mot_encoder(input2)
	    b1 = self.vae_net.body_encoder(input1[:, :-2, :])
	    v1 = self.vae_net.view_encoder(input1[:, :-2, :])
	    v2 = self.vae_net.view_encoder(input2[:, :-2, :])

	    m_mix = self.interpm(m1, m2) 
	    v_mix = self.interpv(v1, v2)
	    # the encoder is matrix, should check whether conv2d works fine
	    dec_input = torch.cat([m_mix, b1, v_mix])
	    out = self.vae_net.decoder(dec_input)
	    # check if still need mapping
	    out = trans_motion_inv(normalize_motion_inv(out, mean_pose, 
	    	std_pose), sx=256, sy=256)


	def postprocess_motion2d(self, motion, mean_pose, std_pose, sx=256, sy=256):
	    motion = motion.reshape(-1, 2, motion.shape[-1])
	    motion = trans_motion_inv(normalize_motion_inv(motion, self.mean_pose, 
	    	self.std_pose), sx, sy)
	    return motion