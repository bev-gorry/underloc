
import os
import gc
import sys
import yaml
import torch
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

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

    "text.latex.preamble": r"\usepackage{amsmath}",})

sns.set_context("paper", font_scale=constants['font_scale'])


def move_to_cpu(features):
    """ Recursively move all tensors in a dictionary to CPU. """
    return {k: v.cpu() for k, v in features.items()}

def main():
        
    # load arguments from yaml file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml_file', type=str, default='../arguments/args_demo.yaml',
                        help='yaml file containing arguments')
    parser.add_argument('--batch_range', type=str, required=True,
                        help='Range of batches to process in the format start,end (e.g., 0,999)')

    arguments = parser.parse_args()
    
    with open (arguments.yaml_file, 'r') as f:
        args = yaml.safe_load(f)
    
    benchmark = hf.parse_yaml(arguments.yaml_file)[0]    
    dataset = args['dataset'].upper()
    log_path = os.path.abspath(os.path.join(root, 'logs', args['log_path'], dataset.lower(), 'superpoint-brute-force'))
    
    query_folder = os.path.join(benchmark, dataset, args['query_folder'])
    database_folder = os.path.join(benchmark, dataset, args['dbs_folder'])
    
    if arguments.batch_range:
        try:
            batch_start, batch_end = map(int, arguments.batch_range.split(','))
        except ValueError:
            print("Invalid batch range format. Please use the format start,end e.g., 0,999.")
            exit(0)
    
    rgb_txt_query = os.path.join(query_folder, 'rgb.txt')
    rgb_txt_db = os.path.join(database_folder, 'rgb.txt')
    
    required_files = [os.path.join(benchmark, dataset), query_folder, database_folder, rgb_txt_query, rgb_txt_db]
    hf.check_files(required_files)
    
    # load query and db filenames from txt files
    query_df = pd.read_csv(rgb_txt_query, sep=' ', header=None)
    db_df = pd.read_csv(rgb_txt_db, sep=' ', header=None)
    
    num_queries = len(query_df)
    num_dbs = len(db_df)
    
    if batch_end > num_queries:
        batch_end = num_queries - 1
        print(f'Batch end has been adjusted to {batch_end}.')
        if batch_start >= num_queries:
            print(f"Batch range {batch_start}-{batch_end} is out of bounds. Max start index is {num_queries - 2}.")
            exit(0)
    
    print(f'Query: {query_df.shape}')
    print(f'DB: {db_df.shape}')
    
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    
    batch_log_path = os.path.join(log_path, f'lightglue_batch_{batch_start}-{batch_end}')
    if os.path.exists(batch_log_path):
        print(f'Batch log path already exists: {batch_log_path}')
        print('Please delete the folder or change the batch range.')
        exit(0)
    os.makedirs(batch_log_path, exist_ok=True)
    
    # setup device and lightglue models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    
    # determine batch size and first/last query image
    batch_size = batch_end - batch_start + 1
    first_query_image = query_df.iloc[batch_start, 1]
    
    print(f'\nProcessing batch: {batch_start}-{batch_end}')
    print(f'Total queries in this batch: {batch_size}')
    print(f'First query image: {first_query_image}')
    
    num_matches = -np.ones((num_dbs, batch_size))

    # load or compute descriptors
    queries_feats_path = os.path.join(log_path, 'queries_feats.pth')
    db_feats_path = os.path.join(log_path, 'db_feats.pth')

    if os.path.exists(queries_feats_path) and os.path.exists(db_feats_path):
        print(f'Found existing features at: {queries_feats_path} and {db_feats_path}')
        
        queries_feats = torch.load(queries_feats_path, weights_only=False)
        db_feats = torch.load(db_feats_path, weights_only=False)
        
        print(f'Loaded {len(queries_feats)} query features and {len(db_feats)} db features.')
    else:
        queries_feats = {}
        db_feats = {}
  
        # compute features in batches
        for i in tqdm(range(0, num_queries, batch_size), desc='Extracting query features in batches'):
            batch_feats = {}  # temporary dictionary for this batch
            
            for j in range(i, min(i + batch_size, num_queries)):
                query_path = os.path.join(query_folder, query_df.iloc[j, 1])
                
                tensor_q = load_image(query_path).cuda()
                feats_q = extractor.extract(tensor_q)
                
                batch_feats[query_path] = move_to_cpu(feats_q)  # move to CPU to free GPU memory
            
            queries_feats.update(batch_feats)                # merge batch into main dictionary
            torch.save(queries_feats, queries_feats_path)    # save after each batch
            
            del batch_feats, tensor_q, feats_q  # free memory
            torch.cuda.empty_cache()
            gc.collect()

        for i in tqdm(range(0, num_dbs, batch_size), desc='Extracting db features in batches'):
            batch_feats = {}  # temporary dictionary for this batch
            
            for j in range(i, min(i + batch_size, num_dbs)):
                db_path = os.path.join(database_folder, db_df.iloc[j, 1])
                
                tensor_db = load_image(db_path).cuda()
                feats_db = extractor.extract(tensor_db)
                
                batch_feats[db_path] = move_to_cpu(feats_db)  # move to CPU to free GPU memory
            
            db_feats.update(batch_feats)           # merge batch into main dictionary
            torch.save(db_feats, db_feats_path)    # save after each batch
            
            del batch_feats, tensor_db, feats_db  # free memory
            torch.cuda.empty_cache()
            gc.collect()

        print("Feature extraction complete. Features saved to disk.")
    
    # filter query features to match batch range
    queries_feats = {k: v for i, (k, v) in enumerate(queries_feats.items()) if batch_start <= i <= batch_end}
    
    # feature matching
    for i, (query_path, feats_q) in tqdm(enumerate(queries_feats.items()), total=batch_size, desc='Feature matching'):
        feats_q = {k: v.cuda() for k, v in feats_q.items()}
        
        for j, (db_path, feats_db) in enumerate(db_feats.items()):
            feats_db = {k: v.cuda() for k, v in feats_db.items()}

            matches01 = matcher({"image0": feats_q, "image1": feats_db})
            feats_q_i, feats_db_i, matches01_i = [
                rbd(x) for x in [feats_q, feats_db, matches01]
            ]
            
            kpts_q, kpts_db, matches = feats_q_i["keypoints"], feats_db_i["keypoints"], matches01_i["matches"]
            
            # populate similarity matrix with number of matches (inliers)
            num_matches[j, i] = len(matches)
    
    inliers_npy = os.path.join(batch_log_path, f'superpoint-brute-force_similarity_matrix.npy')
    np.save(inliers_npy, num_matches)

    torch.cuda.empty_cache()
    gc.collect()
        

if __name__ == '__main__':
    main()