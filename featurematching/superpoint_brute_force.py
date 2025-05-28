import os
import sys
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


def load_image_paths(txt_path):
    df = pd.read_csv(txt_path, sep=' ', header=None)
    return df[1].tolist()

def main():
    
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
    
    # check that all required directories and files exist
    required_files = [os.path.join(benchmark, dataset), rgb_queries, rgb_dbs, rgb_txt_query, rgb_txt_db]
    hf.check_files(required_files)
    
    # load query and db filenames from txt files
    query_df = pd.read_csv(rgb_txt_query, sep=' ', header=None)
    db_df = pd.read_csv(rgb_txt_db, sep=' ', header=None)
    
    num_queries = len(query_df)
    num_dbs = len(db_df)

    # setup device and lightglue models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    
    method = 'superpoint-brute-force'
    method_log_path = os.path.join(log_path, method)
    os.makedirs(method_log_path, exist_ok=True)
    
    # initialize similarity matrix with -1
    num_matches = -np.ones((num_dbs, num_queries))
        
    feature_matching_start = datetime.now()
    
    # extract features for every database image against every query image
    for i in tqdm(range(num_queries), desc=f'{method.upper()}'):
        query_path = os.path.join(rgb_queries, query_df.iloc[i, 1])
        tensor_q = load_image(query_path).cuda()
        feats_q = extractor.extract(tensor_q)
        
        for j in range(num_dbs):
            db_path = os.path.join(rgb_dbs, db_df.iloc[j, 1])

            tensor_db = load_image(db_path).cuda()
            feats_db = extractor.extract(tensor_db)
            
            matches01 = matcher({"image0": feats_q, "image1": feats_db})
            
            feats_q_i, feats_db_i, matches01_i = [
                rbd(x) for x in [feats_q, feats_db, matches01]
            ]
            
            kpts_q, kpts_db, matches = feats_q_i["keypoints"], feats_db_i["keypoints"], matches01_i["matches"]
            
            # populate similarity matrix with number of matches (inliers)
            num_matches[j,i] = len(matches)

    feature_matching_end = datetime.now()
    elapsed_time = feature_matching_end - feature_matching_start
    print(f'ELAPSED TIME: {elapsed_time}')
    
    inliers_npy = os.path.join(method_log_path, f'superpoint-brute-force_similarity_matrix.npy')
    np.save(inliers_npy, num_matches)
        

if __name__ == '__main__':
    main()