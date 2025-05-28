import os
import sys
import cv2
import yaml
import torch
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

lightglue_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Baselines/LightGlue")
sys.path.append(lightglue_path)

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from constants import constants
import utils.utils as hf

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],

    "text.latex.preamble": r"\usepackage{amsmath}",
    })

sns.set_context("paper", font_scale=constants['font_scale'])


def load_image_paths(txt_path):
    df = pd.read_csv(txt_path, sep=' ', header=None)
    return df[1].tolist()

def main():
    
    inlier_threshold = 10
    
    # load arguments from yaml file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml_file', type=str, default='../arguments/args_demo.yaml',
                        help='yaml file containing arguments')
    arguments = parser.parse_args()
    
    with open (arguments.yaml_file, 'r') as f:
        args = yaml.safe_load(f)
            
    benchmark = hf.parse_yaml(arguments.yaml_file)[0]    
    dataset = args['dataset'].upper()
    log_path = os.path.abspath(os.path.join(root, 'logs', args['log_path'], dataset.lower()))
    visualize = args['visualize']
    
    query_folder = str(args['query_folder'])
    database_folder = str(args['dbs_folder'])
    
    K = constants['K']
    visualize_top_n = constants['visualize_top_n']
    
    # check that all required directories and files exist
    rgb_queries = os.path.join(benchmark, dataset, query_folder)
    rgb_dbs = os.path.join(benchmark, dataset, database_folder)
    rgb_txt_query = os.path.join(benchmark, dataset, query_folder, 'rgb.txt')
    rgb_txt_db = os.path.join(benchmark, dataset, database_folder, 'rgb.txt')
    
    required_files = [os.path.join(benchmark, dataset), rgb_queries, rgb_dbs, rgb_txt_query, rgb_txt_db]
    hf.check_files(required_files)
    
    query_paths = load_image_paths(rgb_txt_query)
    db_paths = load_image_paths(rgb_txt_db)
    
    # setup device and lightglue models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    
    method = 'superpoint-hierarchical'
    method_log_path = os.path.join(log_path, method)
    topK_hierarchical_file = os.path.join(method_log_path, f'{method}_similarity_matrix_K{K}.npy')
    
    if os.path.exists(topK_hierarchical_file):
        topK_matrix = np.load(topK_hierarchical_file)
        print(f'Loaded: {topK_hierarchical_file}')
    else:
        print(f'    ERROR! Top K file not found: {topK_hierarchical_file}')
        print(f'Please run {method} first to generate the topK file.')
        exit(0)
    
    figures_dir = os.path.join(log_path, f'figures')
    query_imgs_dir = os.path.join(figures_dir, f'{method}_top-{visualize_top_n}_query-images')
    database_imgs_dir = os.path.join(figures_dir, f'{method}_top-{visualize_top_n}_database-images')
    superpoint_dir = os.path.join(figures_dir, f'{method}_top-{visualize_top_n}_superpoint-correspondences')
    mre_dir = os.path.join(figures_dir, f'{method}_top-{visualize_top_n}_mre')
    
    directories = [
        figures_dir,
        query_imgs_dir,
        database_imgs_dir,
        superpoint_dir,
        mre_dir
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # create a list of tuples (query_idx, db_idx, inliers) for the best result for each query
    best_pairs = []
    for query_idx in range(topK_matrix.shape[1]):
            best_dbs_idx = np.argsort(topK_matrix[:, query_idx])[-1:][0]
            inliers = topK_matrix[best_dbs_idx, query_idx]
            
            best_pairs.append((query_idx, best_dbs_idx, inliers))
    
    # get the top N query-database pairs based on the number of inliers
    best_pairs = sorted(best_pairs, key=lambda x: x[2], reverse=True)[:visualize_top_n]

    for query_idx, best_dbs_idx, inliers in tqdm(best_pairs, desc=f'TOP {visualize_top_n} RESULTS'):
        db_idx = best_dbs_idx

        inliers = topK_matrix[db_idx, query_idx]
        if inliers == -1:
            continue
        
        query_path = os.path.join(rgb_queries, query_paths[query_idx])
        db_path = os.path.join(rgb_dbs, db_paths[db_idx])
        
        # save the query and database images
        rgb_query = Image.open(query_path)
        rgb_db = Image.open(db_path)

        query_fig_path = os.path.join(query_imgs_dir, f'{dataset.lower()}_query_{query_idx}.png')
        db_fig_path = os.path.join(database_imgs_dir, f'{dataset.lower()}_db_{db_idx}.png')
        rgb_query.save(query_fig_path)
        rgb_db.save(db_fig_path)
        plt.close('all')
        
        # load the images and extract features
        tensor_q = load_image(query_path).cuda()
        feats_q = extractor.extract(tensor_q)
        
        tensor_db = load_image(db_path).cuda()
        feats_db = extractor.extract(tensor_db)
        
        matches01 = matcher({"image0": feats_q, "image1": feats_db})
        feats_q_i, feats_db_i, matches01_i = [
            rbd(x) for x in [feats_q, feats_db, matches01]
        ]
        
        # get matched keypoints
        kpts_q, kpts_db, matches = feats_q_i["keypoints"], feats_db_i["keypoints"], matches01_i["matches"]
        m_kpts0, m_kpts1 = kpts_q[matches[..., 0]], kpts_db[matches[..., 1]]
        
        num_matches = len(matches)

        # plot matching keypoints
        tensor_q = load_image(query_path).cuda()
        tensor_db = load_image(db_path).cuda()
        
        axes = viz2d.plot_images([tensor_q, tensor_db])
        viz2d.plot_matches(m_kpts0, m_kpts1, color=None, lw=0.8)

        # save figure with keypoint correspondences
        outfile = os.path.abspath(os.path.join(superpoint_dir, f'superpoint_{num_matches}_{dataset.lower()}_{method}_{query_idx}-{db_idx}.png'))
        plt.savefig(outfile, format='png', bbox_inches='tight')
        
        # continue if there will not be enough keypoint correspondences to compute homography
        if len(matches) < inlier_threshold:
            continue
        
        # convert keypoints to numpy and compute homography
        m_kpts0 = np.float32(m_kpts0.cpu())
        m_kpts1 = np.float32(m_kpts1.cpu())
            
        H, _ = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 5.0)
        
        # use homography to compute reprojection error
        mean_reprojection_error = -1
        if H is not None:
            
            # project keypoints from the query image using H
            projected_kpts = cv2.perspectiveTransform(m_kpts0.reshape(-1, 1, 2), H).reshape(-1, 2)
            
            # compute the distance between the projected keypoints and the matched keypoints in the database image
            distances = np.linalg.norm(m_kpts1 - projected_kpts, axis=1)
            mean_reprojection_error = np.mean(distances)

            image0 = cv2.imread(query_path, cv2.IMREAD_COLOR)
            image1 = cv2.imread(db_path, cv2.IMREAD_COLOR)
            
            # warp the query image using H
            h, w, _ = image0.shape
            warped_image = cv2.warpPerspective(image0, H, (w, h))
            
            alpha = 0.25

            # project a blue tint onto the database image
            tint_filter = np.full_like(image1, constants['blue_bgr'], dtype=np.uint8)
            tinted_img = cv2.addWeighted(image1, 1 - alpha, tint_filter, alpha, 0)
            
            # create a mask for the warped query image and the database image
            fg_mask = (warped_image != 0).any(axis=-1)
            fg_mask_uint8 = fg_mask.astype(np.uint8) * 255
            fg_contours, _ = cv2.findContours(fg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mask_bg = (warped_image == 0).all(axis=-1)
            mask_uint8 = mask_bg.astype(np.uint8) * 255
            bg_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # remove the overlap from the background mask to keep coloring clear
            composite = warped_image.copy()
            composite[mask_bg] = tinted_img[mask_bg]
            
            cv2.drawContours(composite, bg_contours, -1, constants['blue_bgr'], thickness=4)   # BGR
            cv2.drawContours(composite, fg_contours, -1, constants['orange_bgr'], thickness=3) # BGR
            
            # plot actual keypoints from the database image and projected keypoints from the query image
            plt.figure(figsize=(14, 14))
            plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
            plt.scatter(m_kpts1[:, 0], m_kpts1[:, 1], c='blue', marker='o', s=100, label='Actual Keypoints')
            plt.scatter(projected_kpts[:, 0], projected_kpts[:, 1], c='orange', s=150, marker='x', label='Projected Keypoints')
            plt.axis('off')
            
            # plot the reprojection error
            for (x1, y1), (x2, y2) in zip(m_kpts1, projected_kpts):
                plt.plot([x1, x2], [y1, y2], 'y--', linewidth=1.5)
            
            outfile = os.path.abspath(os.path.join(mre_dir, f'mre_{mean_reprojection_error}_{dataset.lower()}_{method}_{query_idx}-{db_idx}.png'))
            plt.savefig(outfile, format='png', bbox_inches='tight')
            
            if visualize:
                plt.show()
            plt.close('all')

if __name__ == '__main__':
    main()