import numpy as np
import pandas as pd 
import json
import os 
import pdb
import matplotlib.pyplot as plt

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='uint8')
    plt.scatter(cords[:,0], cords[:,1])
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        plt.annotate(str(i), (point[0], point[1]))

    plt.show()
    return result


MISSING_VALUE = -1
# fix PATH
img_dir  = 'fashion_data' #raw image path
annotations_file = 'fashion_data/fasion-resize-annotation-train.csv' #pose annotation path
save_path = 'fashion_data/trainK' #path to store pose maps

annotations_file = pd.read_csv(annotations_file, sep=':')
annotations_file = annotations_file.set_index('name')
image_size = (256, 176)

row = annotations_file.iloc[0]
name = row.name

kp_array = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
pose = cords_to_map(kp_array, image_size)
