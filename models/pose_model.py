import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
from .openpose.rtpose_vgg import get_model
from .openpose.post import decode_pose
import cv2

from .openpose import im_transform

# weight_name = './checkpoints/openpose/pose_model.pth'

# model = get_model('vgg19')     
# model.load_state_dict(torch.load(weight_name))
# model = torch.nn.DataParallel(model).cuda()
# model.float()
# model.eval()

def rtpose_preprocess(image):
    image = image.astype(np.float32)
    image = image / 256. - 0.5
    image = image.transpose((2, 0, 1)).astype(np.float32)

    return image

def handle_paf_and_heat(normal_heat, flipped_heat, normal_paf, flipped_paf):
    """Compute the average of normal and flipped heatmap and paf
    :param normal_heat: numpy array, the normal heatmap
    :param normal_paf: numpy array, the normal paf
    :param flipped_heat: numpy array, the flipped heatmap
    :param flipped_paf: numpy array, the flipped  paf
    :returns: numpy arrays, the averaged paf and heatmap
    """

    # The order to swap left and right of heatmap
    swap_heat = np.array((0, 1, 5, 6, 7, 2, 3, 4, 11, 12,
                          13, 8, 9, 10, 15, 14, 17, 16, 18))

    # paf's order
    # 0,1 2,3 4,5
    # neck to right_hip, right_hip to right_knee, right_knee to right_ankle

    # 6,7 8,9, 10,11
    # neck to left_hip, left_hip to left_knee, left_knee to left_ankle

    # 12,13 14,15, 16,17, 18, 19
    # neck to right_shoulder, right_shoulder to right_elbow, right_elbow to
    # right_wrist, right_shoulder to right_ear

    # 20,21 22,23, 24,25 26,27
    # neck to left_shoulder, left_shoulder to left_elbow, left_elbow to
    # left_wrist, left_shoulder to left_ear

    # 28,29, 30,31, 32,33, 34,35 36,37
    # neck to nose, nose to right_eye, nose to left_eye, right_eye to
    # right_ear, left_eye to left_ear So the swap of paf should be:
    swap_paf = np.array((6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 20, 21, 22, 23,
                         24, 25, 26, 27, 12, 13, 14, 15, 16, 17, 18, 19, 28,
                         29, 32, 33, 30, 31, 36, 37, 34, 35))

    flipped_paf = torch.flip(flipped_paf, (1,2))

    # The pafs are unit vectors, The x will change direction after flipped.
    # not easy to understand, you may try visualize it.
    flipped_paf[:, :, :,swap_paf[1::2]] = flipped_paf[:, :, :,swap_paf[1::2]]
    flipped_paf[:, :, :,swap_paf[::2]] = -flipped_paf[:, :, :,swap_paf[::2]]

    averaged_paf = (normal_paf + flipped_paf[:, :, :,swap_paf]) / 2.
    averaged_heatmap = (
        normal_heat + torch.flip(flipped_heat, (1,2))[:, :, :,swap_heat]) / 2.

    return averaged_paf, averaged_heatmap

def get_multiplier(img):
    """Computes the sizes of image at different scales
    :param img: numpy array, the current image
    :returns : list of float. The computed scales
    """
    scale_search = [0.5, 1., 1.5, 2, 2.5]
    return [x * 368. / float(img.shape[0]) for x in scale_search]

def get_outputs(multiplier, batch_var, model, preprocess):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """

    heatmap_avg = torch.zeros((batch_var.shape[0], batch_var.shape[2], batch_var.shape[3], 19))
    paf_avg = torch.zeros((batch_var.shape[0], batch_var.shape[2], batch_var.shape[3], 38))
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]

    # heatmaps = output2.transpose(1, 2).transpose(2,3)
    # pafs = output1.transpose(1, 2).transpose(2,3)

    heatmaps = F.upsample(output2, size=(batch_var.shape[2], batch_var.shape[3]), mode='bilinear').transpose(1, 2).transpose(2,3)
    pafs = F.upsample(output1, size=(batch_var.shape[2], batch_var.shape[3]), mode='bilinear').transpose(1, 2).transpose(2,3)

    heatmap_avg = heatmap_avg.cuda() + heatmaps
    paf_avg = paf_avg.cuda() + pafs

    return paf_avg, heatmap_avg

# test_image = './readme/ski.jpg'
# oriImg = cv2.imread(test_image) # B,G,R order

def get_pose(oriImg, model):
	shape_dst = np.min(oriImg.shape[0:2])

	# Get results of original image
	multiplier = get_multiplier(oriImg)

	with torch.no_grad():
	    orig_paf, orig_heat = get_outputs(
	        multiplier, oriImg, model,  'rtpose')
	          
	    # Get results of flipped image
	    swapped_img = torch.flip(oriImg, (2, 3))
	    # import pdb
	    # pdb.set_trace()
	    flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
	                                            model, 'rtpose')

	    # compute averaged heatmap and paf
	    paf, heatmap = handle_paf_and_heat(
	        orig_heat, flipped_heat, orig_paf, flipped_paf)
	            
	param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
	for i in range(2):

		# import pdb
		# pdb.set_trace()
		canvas, to_plot, candidate, subset = decode_pose(
		    oriImg.cpu().detach().numpy()[i].transpose(1,2,0), param, heatmap.cpu().numpy()[i], paf.cpu().numpy()[i])

	return to_plot