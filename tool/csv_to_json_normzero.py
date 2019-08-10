import numpy as np
import pandas as pd 
import json
import os 
import pdb
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

MISSING_VALUE = -1
# fix PATH
img_dir  = '../fashion_data' #raw image path
annotations_file = '../fashion_data/fasion-resize-annotation-train.csv' #pose annotation path
save_path = '../fashion_data/train_kjson_normzero' #path to store pose maps

def pad_to_16x(x):
    if x % 16 > 0:
        return x - x % 16 + 16
    return x

def pad_to_height(tar_height, img_height, img_width):
    scale = tar_height / img_height
    h = pad_to_16x(tar_height)
    w = pad_to_16x(int(img_width * scale))
    return h, w, scale

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def trans_motion2d(motion2d):
    # subtract centers to local coordinates
    # pdb.set_trace()
    centers = motion2d[8, :, :]
    motion_proj = motion2d - centers

    # adding velocity
    velocity = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)
    motion_proj = np.r_[motion_proj[:8], motion_proj[9:], velocity]

    return motion_proj, centers

def normalize_motion(motion, mean_pose, std_pose):
    """
    :param motion: (J, 2, T)
    :param mean_pose: (J, 2)
    :param std_pose: (J, 2)
    :return:
    """
    return (motion - mean_pose[:, :, np.newaxis]) / std_pose[:, :, np.newaxis]


def preprocess_motion2d(motion, mean_pose, std_pose):
    motion, centers = trans_motion2d(motion)
    motion_trans = normalize_motion(motion, mean_pose, std_pose)
    motion[9:11] = 0
    motion[12:14] = 0
    motion_trans = motion_trans.reshape((-1, motion_trans.shape[-1]))
    return np.expand_dims(motion_trans, axis=0), centers

def get_meanpose():
    meanpose_path = '../mixamo_data/meanpose_with_view.npy'
    stdpose_path = '../mixamo_data/stdpose_with_view.npy'
    meanpose = np.load(meanpose_path)
    stdpose = np.load(stdpose_path)
    return meanpose, stdpose

def normalize_motion_inv(motion, mean_pose, std_pose):
    if len(motion.shape) == 2:
        motion = motion.reshape(-1, 2, motion.shape[-1])
    return motion * std_pose[:,:,np.newaxis] + mean_pose[:,:,np.newaxis]

def trans_motion_inv(motion, sx, sy, velocity=None):
    # seems like hip is the center, still need to check
    # motion_inv = torch.cat([motion[:,:8], torch.zeros((motion.shape[0], 1, 2, motion.shape[-1])).cuda(), motion[:,8:-1]], 1)
    motion_inv = motion[:,:-1]

    # restore centre position
    motion_inv[:,:,0] += sx
    motion_inv[:,:,1] += sy
    return motion_inv[...,0]

def compute_pose(annotations_file, savePath, sigma=6):
    annotations_file = pd.read_csv(annotations_file, sep=':')
    annotations_file = annotations_file.set_index('name')
    image_size = (256, 176)
    cnt = len(annotations_file)
    for i in tqdm(range(cnt)):
        row = annotations_file.iloc[i]
        name = row.name
        file_name = os.path.join(savePath, name + '.npy')
        motion = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
        # pdb.set_trace()
        hip_motion = np.zeros([19,2])
        mid_hip = (motion[8] + motion[11]) / 2
        hip_motion[8] = mid_hip
        hip_motion[:8] = motion[:8]
        hip_motion[9:] = motion[8:]
        
        face = np.zeros([5,2])
        face[0] = motion[0]
        face[1:] = motion[14:]
        h1, w1, scale = pad_to_height(512, 256, 176)
        # pdb.set_trace()
        mean_pose, std_pose = get_meanpose()
        # motion = gaussian_filter1d(motion[:15], sigma=2, axis=-1)
        # still need to check xy or yx
        motion = hip_motion[:15] * scale
        motion = np.expand_dims(motion, axis=2)
        input1, centers = preprocess_motion2d(motion, mean_pose, std_pose)
        # trans -> norm
        d1 = {'input1':input1, 'centers': centers, 'face': face}
        # input2 = np.reshape(input1, (input1.shape[0], 15, 2, -1))
        # # norm 
        # # trans
        # # scale
        # input2 = normalize_motion_inv(input2, mean_pose, std_pose)
        # input2 = trans_motion_inv(input2, sx=254, sy=146)
        # pdb.set_trace()
        np.save(file_name, d1)


compute_pose(annotations_file, save_path)
