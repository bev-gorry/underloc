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

from tqdm import tqdm
from datetime import datetime

lightglue_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Baselines/LightGlue")
sys.path.append(lightglue_path)

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

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


def get_topK(log_path, method, K, topK_file, gt_positives):
    """Get the top K matches for each query image based on the similarity matrix."""

    similarity_matrix_path = os.path.join(log_path, method, f'{method}_similarity_matrix.npy')
    
    if not os.path.exists(similarity_matrix_path):
        print(f"    ERROR! Similarity matrix not found: {similarity_matrix_path}")
        exit(0)
    similarity_matrix = np.load(similarity_matrix_path)
    
    # initialize topK matrix with zeros
    topK = np.zeros_like(similarity_matrix)
    
    # get the top K indices for each column (query)
    topK_indices = np.argsort(-similarity_matrix, axis=0)[:K, :]
    
    # populate the topK matrix with 1s at the top K indices
    np.put_along_axis(topK, topK_indices, 1, axis=0)
    
    # remove columns where all values are 0
    for col in range(gt_positives.shape[1]):
        if np.all(gt_positives[:, col] == 0):
            topK[:, col] = 0
    
    np.save(topK_file, topK)

    return topK

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
    
    query_folder = str(args['query_folder'])
    database_folder = str(args['dbs_folder'])
    
    rgb_queries = os.path.join(benchmark, dataset, query_folder)
    rgb_dbs = os.path.join(benchmark, dataset, database_folder)
    rgb_txt_query = os.path.join(benchmark, dataset, query_folder, 'rgb.txt')
    rgb_txt_db = os.path.join(benchmark, dataset, database_folder, 'rgb.txt')
    
    loc_rad = args['vpr_loc_radius']
    K = constants['K']
    
    vpr_method = 'megaloc'                                    ####################################### ONLY DO HIERARCHICAL SUPERPOINT ON THE BEST VPR
    method_log_path = os.path.join(log_path, vpr_method)
    if not os.path.exists(method_log_path):
        print(f"    ERROR! VPR method log path not found: {method_log_path}")
        print(f'Please set vpr_method: megaloc in {arguments.yaml_file} and run << pixi run -e vpr execute --yaml_file {arguments.yaml_file} >> before running hierarchical feature matching.')
        exit(0)
    
    # check that all required directories and files exist
    required_files = [os.path.join(benchmark, dataset), rgb_queries, rgb_dbs, rgb_txt_query, rgb_txt_db]
    hf.check_files(required_files)
    
    # load groundtruth matrix
    gt_matrix_path = os.path.join(benchmark, dataset, f'groundtruth_{query_folder}-{database_folder}.npy')
    if not os.path.exists(gt_matrix_path):
        print(f"    ERROR! Ground truth matrix not found: {gt_matrix_path}")
        print(f'Please run << ARGUMENT={arguments.yaml_file} pixi run create_groundtruth >> first to generate the groundtruth matrix.')
        exit(0)
    gt = np.load(gt_matrix_path)
    
    # binary gt matrix
    gt_positives = hf.get_gt_positives(gt, loc_rad)
    
    # create output directories
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
        
    hierarchical_log_path = os.path.join(log_path, 'superpoint-hierarchical')
    os.makedirs(hierarchical_log_path, exist_ok=True)
    
    reproj_log_path = os.path.join(log_path, 'reprojection-error')
    os.makedirs(reproj_log_path, exist_ok=True)
    
    homography_dir = os.path.join(hierarchical_log_path, f'homography')
    os.makedirs(homography_dir, exist_ok=True)

    # get topK pointer matrix (1s and 0s for the top K results per query)
    topK_file = os.path.join(hierarchical_log_path, f'{dataset.lower()}_{vpr_method}_topK-{K}.npy')
    if os.path.exists(topK_file):
        topK = np.load(topK_file)
        print(f'Loaded: {topK_file}')
    else:
        topK = get_topK(log_path, vpr_method, K, topK_file, gt_positives)
        print(f'Saved: {topK_file}')

    # setup device and lightglue models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    
    # load query and db filenames from txt files
    query_df = pd.read_csv(rgb_txt_query, sep=' ', header=None)
    db_df = pd.read_csv(rgb_txt_db, sep=' ', header=None)
    
    num_queries = len(query_df)
    num_dbs = len(db_df)
    
    # initialize matrices for num_inliers and reprojection_error with -1
    num_matches = -np.ones((num_dbs, num_queries))
    mean_reprojection_error = -np.ones((num_dbs, num_queries))
    
    feature_matching_start = datetime.now()
    
    for i in tqdm(range(num_queries), desc=f'{vpr_method.upper()}'):
        query_path = os.path.join(rgb_queries, query_df.iloc[i, 1])
        tensor_q = load_image(query_path).cuda()
        feats_q = extractor.extract(tensor_q)
        
        for j in range(num_dbs):
            # only compute matches if the topK pointer matrix has a 1
            if np.any(topK[j, i]):
                db_path = os.path.join(rgb_dbs, db_df.iloc[j, 1])
                tensor_db = load_image(db_path).cuda()
                feats_db = extractor.extract(tensor_db)
                
                matches01 = matcher({"image0": feats_q, "image1": feats_db})
                
                feats_q_i, feats_db_i, matches01_i = [
                    rbd(x) for x in [feats_q, feats_db, matches01]
                ]
                
                kpts_q, kpts_db, matches = feats_q_i["keypoints"], feats_db_i["keypoints"], matches01_i["matches"]
                m_kpts0, m_kpts1 = kpts_q[matches[..., 0]], kpts_db[matches[..., 1]]
                
                # populate matrix with number of matches (inliers)
                num_matches[j,i] = len(matches)
                
                # continue if there will not be enough keypoint correspondences to compute homography
                if len(matches) < inlier_threshold:
                    continue
                
                # convert keypoints to numpy for homography
                m_kpts0 = np.float32(m_kpts0.cpu())
                m_kpts1 = np.float32(m_kpts1.cpu())
                
                H, _ = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 5.0)
                
                if H is not None:
                    homography_path = os.path.join(homography_dir, f'{dataset.lower()}_H_{i}-{j}.npy')
                    np.save(homography_path, H)
                
                    projected_kpts_0 = cv2.perspectiveTransform(m_kpts0.reshape(1, -1, 2), H).reshape(-1, 2)
                    reproj_error_0 = np.linalg.norm(m_kpts1 - projected_kpts_0, axis=1)
                    
                    projected_kpts_1 = cv2.perspectiveTransform(m_kpts1.reshape(1, -1, 2), np.linalg.inv(H)).reshape(-1, 2)
                    reproj_error_1 = np.linalg.norm(m_kpts0 - projected_kpts_1, axis=1)
                    
                    # populate matrix with reprojection error
                    mean_reprojection_error[j,i] = (np.mean(reproj_error_1) + np.mean(reproj_error_0)) * 0.5

    feature_matching_end = datetime.now()
    elapsed_time = feature_matching_end - feature_matching_start
    print(f'ELAPSED TIME: {elapsed_time}')
        
    inliers_npy = os.path.join(hierarchical_log_path, f'superpoint-hierarchical_similarity_matrix_K{K}.npy')
    np.save(inliers_npy, num_matches)
    
    reprojection_npy = os.path.join(reproj_log_path, f'reprojection-error_similarity_matrix_K{K}.npy')
    np.save(reprojection_npy, mean_reprojection_error)
        

if __name__ == '__main__':
    main()