from tqdm import tqdm
import pandas as pd 
import numpy as np
import json

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)



def get_meanpose():
    meanpose_path = 'deepfashion_meanpose.npy'
    stdpose_path = 'deepfashion_stdpose.py'
    meanpose, stdpose, meanpose_centered, stdpose_centered = gen_meanpose()
    np.save(meanpose_path, meanpose)
    np.save(stdpose_path, stdpose)
    print(meanpose, stdpose, meanpose_centered, stdpose_centered)
    print("meanpose saved at {}".format(meanpose_path))
    print("stdpose saved at {}".format(stdpose_path))

    meanpose_path = 'deepfashion_meanpose_centered.npy'
    stdpose_path = 'deepfashion_stdpose_centered.py'

    np.save(meanpose_path, meanpose_centered)
    np.save(stdpose_path, stdpose_centered)

    # return meanpose, stdpose

def gen_meanpose():
    annotations_file = './fashion_data/fasion-resize-annotation-train.csv'
    annotations_file = pd.read_csv(annotations_file, sep=':')
    annotations_file = annotations_file.set_index('name')

    meanpose, stdpose = np.zeros([19, 2]), np.zeros([19, 2])
    cnt = len(annotations_file)
    # why not use matrix mean ??? 
    # anyway, I'm too exhausted

    all_jointx =[]
    all_jointy = []
    j1, j2 = [], []
    for i in tqdm(range(cnt)):
        row = annotations_file.iloc[i]
        kp_array = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
        point = (kp_array[8] + kp_array[11] )/2
        j1.append(point[0])
        j2.append(point[1])
        if -1 in list(kp_array[8]) + list(kp_array[11]):
            all_jointx.append(point[0])
            all_jointy.append(point[1])

    meanpose[14,0], meanpose[14,1] = np.mean(all_jointx), np.mean(all_jointy)
    stdpose[14,0], stdpose[14,1] = np.std(all_jointx),np. std(all_jointy)
    hip_array = np.stack([np.array(j1), np.array(j2)], 1)

    meanposec, stdposec = np.zeros([19, 2]), np.zeros([19, 2])
    for j in range(18):
        all_jointx, all_jointxc = [], []
        all_jointy, all_jointyc = [], []
        for i in tqdm(range(cnt)):
            row = annotations_file.iloc[i]
            kp_array = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
            point = kp_array[j]
            if point[0] == -1 or point[1] == -1:
                continue
            all_jointx.append(point[0])
            all_jointy.append(point[1])
            all_jointxc.append(point[0]-hip_array[i,0])
            all_jointyc.append(point[1]-hip_array[i,1])
        if j >= 14:
            meanpose[j+1,0], meanpose[j+1,1] = np.mean(all_jointx), np.mean(all_jointy)
            stdpose[j+1,0], stdpose[j+1,1] = np.std(all_jointx),np. std(all_jointy)
            meanposec[j+1,0], meanposec[j+1,1] = np.mean(all_jointxc), np.mean(all_jointyc)
            stdposec[j+1,0], stdposec[j+1,1] = np.std(all_jointxc),np. std(all_jointyc)
        else:
            meanpose[j,0], meanpose[j,1] = np.mean(all_jointx), np.mean(all_jointy)
            stdpose[j,0], stdpose[j,1] = np.std(all_jointx),np. std(all_jointy)
            meanposec[j,0], meanposec[j,1] = np.mean(all_jointxc), np.mean(all_jointyc)
            stdposec[j,0], stdposec[j,1] = np.std(all_jointxc),np. std(all_jointyc)



        
    stdpose[np.where(stdpose == 0)] = 1e-9
    stdposec[np.where(stdposec == 0)] = 1e-9
    # meanpose_centered = meanpose - np.array([256/2, 176/2])
    # stdpose_centered = stdpose
    
    return meanpose, stdpose, meanposec, stdposec

if __name__ == '__main__':
    get_meanpose()