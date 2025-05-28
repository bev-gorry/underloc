import os
import sys
import yaml
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from constants import constants
import utils.utils as hf

def run_vpr(vpr, log_dir, database_path, query_path, args):
    """Runs the VPR pipeline with the specified input arguments."""
    cwd = os.path.dirname(vpr)
    
    method = args['vpr_method']
    num_preds_to_save = args['vpr_num_preds_to_save'] if 'vpr_num_preds_to_save' in args else 0
    
    # automatically set the backbone and descriptors_dimension
    args_vpr = [f'--log_dir={log_dir}', f'--method={method}', f'--num_preds_to_save={num_preds_to_save}', f'--database_folder={database_path}', f'--queries_folder={query_path}'] + (['--no_labels'])

    print(' ')
    print(f'    Running VPR pipeline with arguments: {args_vpr}')
    exit_code_vpr = hf.run_script(sys.executable, [vpr] + args_vpr, cwd)
    if exit_code_vpr != 0:
        print(f'VPR pipeline failed with exit code {exit_code_vpr}. Exiting.')
        sys.exit(exit_code_vpr)

def random_vpr(gt_npy_file, gt_threshold, recalls_k):
    """Calculates random recall values for VPR."""
    gt_matrix = np.load(gt_npy_file)
    
    mont_size = 10

    positive_matrix = (gt_matrix <= gt_threshold).astype(int)  

    num_ones = np.count_nonzero(positive_matrix == 1)

    valid_columns = []
    for col in range(positive_matrix.shape[1]):
        num_ones = np.count_nonzero(positive_matrix[:, col] == 1)
        if num_ones > 0:
            valid_columns.append(col)

    num_rows = positive_matrix.shape[0]
    random_recalls = []
    for k in recalls_k:
        num_tries = 0
        succesful_tries = 0
        for col in tqdm(valid_columns):
            column_data = positive_matrix[:, col]
            num_ones = np.count_nonzero(column_data == 1)
            if num_ones > 0:
                for i in range(mont_size):
                    num_tries += 1
                    indexes = np.random.choice(range(num_rows), size=k, replace=False).tolist()
                    contains_one = np.any(column_data[indexes] == 1)
                    succesful_tries += contains_one
        random_recalls.append((succesful_tries/num_tries)*100)

    print("Random Recalls:", random_recalls)
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(recalls_k, random_recalls, marker='o')
    plt.title("Random Recall")
    plt.xlabel("K")
    plt.ylabel("Recall @ K")
    plt.grid()
    # plt.show()
    
    return random_recalls, fig

def main():
    
    vpr = Path('../Baselines/VPR-methods-evaluation/main.py').resolve()
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml_file', type=str, default='../arguments/args_demo.yaml',
                        help='yaml file containing arguments')
    arguments = parser.parse_args()
    
    with open (arguments.yaml_file, 'r') as f:
        args = yaml.safe_load(f)
    
    benchmark = hf.parse_yaml(arguments.yaml_file)[0]    
    method = args['vpr_method']
    dataset = args['dataset'].upper()
    log_path = os.path.abspath(os.path.join(root, 'logs', args['log_path'], dataset.lower(), method))
    
    query_folder = str(args['query_folder'])
    database_folder = str(args['dbs_folder'])
    query_path = os.path.join(benchmark, dataset, query_folder, 'rgb')
    database_path = os.path.join(benchmark, dataset, database_folder, 'rgb')
    
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    
    print(f'METHOD: {method}')

    # check that all required directories and files exist
    required_files = [os.path.join(benchmark, dataset), os.path.join(benchmark, dataset, query_folder), os.path.join(benchmark, dataset, database_folder)]
    hf.check_files(required_files)
    
    query_image_files = sorted([f for f in os.listdir(query_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))])
    database_image_files = sorted([f for f in os.listdir(database_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))])
    
    input_data = [(query_path, query_image_files), (database_path, database_image_files)]
    
    # downsample images if they are too large
    for path, img_files in input_data:
        
        downsampled_path = f'{path.rstrip("/")}_downsampled'
        
        if path == query_path and os.path.exists(downsampled_path):
            query_path = downsampled_path
            print('FOUND DOWNSAMPLED QUERY PATH')
            continue
        if path == database_path and os.path.exists(downsampled_path):
            database_path = downsampled_path
            print('FOUND DOWNSAMPLED DATABASE PATH')
            continue
        
        if img_files:
            with Image.open(os.path.join(path, img_files[0])) as img:
                if path == query_path:
                    query_path = hf.downsample_images(query_path, f'{query_path.rstrip('/')}_downsampled', 0.1) if img.width > 4000 else query_path
                elif path == database_path:
                    database_path = hf.downsample_images(database_path, f'{database_path.rstrip('/')}_downsampled', 0.1) if img.width > 4000 else database_path
        else:
            print(f'    No images found in {query_path}. Exiting.')
            sys.exit(0)

    print(' ')
    print(f'    Using database_path: {database_path}')
    print(f'    Database size: {len(database_image_files)} images.')
    print(' ')
    print(f'    Using query_path: {query_path}')
    print(f'    Query size: {len(query_image_files)} images.')

    start_time = datetime.now()
    
    # run random VPR
    if method == 'random':
        gt_npy_file = os.path.join(benchmark, dataset, f'groundtruth_{query_folder}-{database_folder}.npy')
        gt_threshold = args['vpr_loc_radius']
        k_values = constants['k_values']
        
        start_time = datetime.now()
        
        recalls, fig = random_vpr(gt_npy_file, gt_threshold, k_values)
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        np.save(os.path.join(log_path, 'random_recalls.npy'), recalls)
        
        fig.savefig(os.path.join(log_path, 'random_recalls.png'))
        
        exit(0)
    else:
        run_vpr(vpr, log_path, database_path, query_path, args)
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f'ELAPSED TIME: {elapsed_time}')


if __name__ == '__main__':
    main()