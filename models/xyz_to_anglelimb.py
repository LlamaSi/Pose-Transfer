import numpy as np
import torch

mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
				[10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
              [6, 8], [8, 9]]

joints = [[0,1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9],
			[9, 10], [10, 11], [12, 13], [2, 12]]
sample_path = '3d_sample.npy'

# should be 16 * 3 dim xyz
prediction_3d = np.load(sample_path)
limbs = [0] * len(mpii_edges)
# apex for computing the angle
relative_angles = []
absolute_angles = np.zeros([16, 3])
offset = prediction_3d[0]

def joint_angles(prediction_3d):
	jangles = torch.zeros([12, 3])
	for i, joint in enumerate(joints):
		e1, e2 = joint
		i1, i2 = mpii_edges[e1]
		i3, i4 = mpii_edges[e2]
		v1 = prediction_3d[i1] - prediction_3d[i2]
		v2 = prediction_3d[i4] - prediction_3d[i3]

		jangles[i] = torch.acos(torch.dot(v1, v2) / torch.norm(v1) / torch.norm(v2))
	return jangles

def absolute_angles(prediction_3d):
	absolute_angles = torch.zeros([14, 3])
	limbs = torch.zeros([14, 3])
	for i in range(len(mpii_edges)):
		i1, i2 = mpii_edges[i]
		e1 = prediction_3d[i1] - prediction_3d[i2]
		l = torch.norm(e1)
		limbs.append(l)
		ax, bx, cx = e1 / l
		aa, ba, ca = torch.acos(ax), torch.acos(bx), torch.acos(cx)

		absolute_angles[i] = torch.Tensor([aa, ba, ca])

	return absolute_angles, limbs

def anglelimbtoxyz(offset, absolute_angles, limbs):
	res_3d = torch.zeros([16, 3])
	
	norm_direction = torch.cos(absolute_angles)
	direction = torch.mm(limbs, norm_direction)

	res_3d[0] = offset
	res_3d[1] = res_3d[0] + direction[0]
	res_3d[2] = res_3d[1] + direction[1]
	res_3d[6] = res_3d[2] + direction[2]
	res_3d[3] = res_3d[6] + direction[3]
	res_3d[4] = res_3d[3] + direction[4]
	res_3d[5] = res_3d[4] + direction[5]
	res_3d[8] = res_3d[6] + direction[12]
	res_3d[9] = res_3d[8] + direction[13]
	res_3d[13] = res_3d[8] + direction[9]
	res_3d[14] = res_3d[13] + direction[10]
	res_3d[15] = res_3d[14] + direction[11]
	res_3d[12] = res_3d[8] - direction[8]
	res_3d[11] = res_3d[12] - direction[7]
	res_3d[10] = res_3d[11] + direction[6]

	return res_3d

import pdb
pdb.set_trace()

