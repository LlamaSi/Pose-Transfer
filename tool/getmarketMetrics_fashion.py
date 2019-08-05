import os
# from inception_score import get_inception_score

from skimage.io import imread, imsave
from skimage.measure import compare_ssim
from skimage.draw import circle, line_aa, polygon

import numpy as np
import pandas as pd

from tqdm import tqdm
import re

MISSING_VALUE = -2

def produce_ma_mask(kp_array, img_size, point_radius=4):
    from skimage.morphology import dilation, erosion, square
    mask = np.zeros(shape=img_size, dtype=bool)
    limbs = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
              [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
               [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    limbs = np.array(limbs) - 1
    for f, t in limbs:
        from_missing = kp_array[f][0] == MISSING_VALUE or kp_array[f][1] == MISSING_VALUE
        to_missing = kp_array[t][0] == MISSING_VALUE or kp_array[t][1] == MISSING_VALUE
        if from_missing or to_missing:
            continue

        norm_vec = kp_array[f] - kp_array[t]
        norm_vec = np.array([-norm_vec[1], norm_vec[0]])
        norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)


        vetexes = np.array([
            kp_array[f] + norm_vec,
            kp_array[f] - norm_vec,
            kp_array[t] - norm_vec,
            kp_array[t] + norm_vec
        ])
        yy, xx = polygon(vetexes[:, 0], vetexes[:, 1], shape=img_size)
        mask[yy, xx] = True

    for i, joint in enumerate(kp_array):
        if kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=point_radius, shape=img_size)
        mask[yy, xx] = True

    mask = dilation(mask, square(5))
    mask = erosion(mask, square(5))
    return mask

def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)


def save_images(input_images, target_images, generated_images, names, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for images in zip(input_images, target_images, generated_images, names):
        res_name = str('_'.join(images[-1])) + '.png'
        imsave(os.path.join(output_folder, res_name), np.concatenate(images[:-1], axis=1))


def create_masked_image(names, images, annotation_file):
    from pose_utils import load_pose_cords_from_strings
    masked_images = []
    df = pd.read_csv(annotation_file, sep=':')
    for name, image in zip(names, images):
        to = name[1]
        ano_to = df[df['name'] == to].iloc[0]

        kp_to = load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

        mask = produce_ma_mask(2*kp_to, image.shape[:2])
        masked_images.append(image * mask[..., np.newaxis])

    return masked_images


def load_generated_images(images_folder):
    input_images = []
    target_images = []
    generated_images = []

    names = []
    for img_name in os.listdir(images_folder):
        img = imread(os.path.join(images_folder, img_name))
        w = int(img.shape[1] / 5) #h, w ,c
        input_images.append(img[:, :w])
        target_images.append(img[:, 2*w:3*w])
        generated_images.append(img[:, 4*w:5*w])

        # assert img_name.endswith('_vis.png'), 'unexpected img name: should end with _vis.png'
        assert img_name.endswith('_vis.png') or img_name.endswith('_vis.jpg'), 'unexpected img name: should end with _vis.png'

        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        fr = img_name[0]
        to = img_name[1]

        # m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)', img_name)
        # m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)_vis.png', img_name)
        # fr = m.groups()[0]
        # to = m.groups()[1]
        names.append([fr, to])

    return input_images, target_images, generated_images, names



def test(generated_images_dir, annotations_file_test):
    print(generated_images_dir, annotations_file_test)
    print ("Loading images...")
    input_images, target_images, generated_images, names = load_generated_images(generated_images_dir)

    # print ("Compute inception score...")
    # inception_score = get_inception_score(generated_images)
    # print ("Inception score %s" % inception_score[0])


    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    print ("SSIM score %s" % structured_score)

    # print ("Compute l1 score...")
    # norm_score = l1_score(generated_images, target_images)
    # print ("L1 score %s" % norm_score)

    # print ("Compute masked inception score...")
    generated_images_masked = create_masked_image(names, generated_images, annotations_file_test)
    reference_images_masked = create_masked_image(names, target_images, annotations_file_test)
    # inception_score_masked = get_inception_score(generated_images_masked)
    # print ("Inception score masked %s" % inception_score_masked[0])

    print ("Compute masked SSIM...")
    structured_score_masked = ssim_score(generated_images_masked, reference_images_masked)
    print ("SSIM score masked %s" % structured_score_masked)

    print ("Inception score = %s, masked = %s; SSIM score = %s, masked = %s; l1 score = %s" %
           (inception_score, inception_score_masked, structured_score, structured_score_masked, norm_score))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    generated_images_dir = args.dir
    # generated_images_dir = '/home/wenwens/Documents/HumanPose/Preprocess/Pose-Transfer/results/201908021506/test_latest/images'
    annotations_file_test = 'market_data/market-annotation-test.csv'

    test(generated_images_dir, annotations_file_test)






