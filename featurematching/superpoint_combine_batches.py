import os
import sys
import yaml
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from constants import constants

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],

    "text.latex.preamble": r"\usepackage{amsmath}",})

sns.set_context("paper", font_scale=constants['font_scale'])

def get_batches(log_path):
    log_path = Path(log_path)
    
    batches = sorted(
        [f.name for f in log_path.iterdir() if f.is_dir() and f.name.startswith("lightglue_batch_")],
        key=lambda x: int(x.split("_")[-1].split("-")[0]) 
    )
    
    print(f'Found {len(batches)} batches: {batches}')
    
    if len(batches) == 0:
        return None
    
    return batches

def stack_batches(log_path, dataset, batches):
    file_paths = [os.path.join(log_path, batch, f'superpoint-brute-force_similarity_matrix.npy') for batch in batches]
    
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f'File not found: {fp}')
            continue
        
    matrices = []
    for fp in file_paths:
        m = np.load(fp)
        matrices.append(m)
    
    print(f'Loaded {len(matrices)} matrices.')
    
    result = (np.hstack(matrices))
    result[result < 0] = np.nan
    
    return result

if __name__ == '__main__':
    
    # load arguments from yaml file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml_file', type=str, default='../arguments/args_demo.yaml',
                        help='yaml file containing arguments')
    arguments = parser.parse_args()
    
    with open (arguments.yaml_file, 'r') as f:
        args = yaml.safe_load(f)
    
    dataset = args['dataset'].upper()
    log_path = os.path.abspath(os.path.join(root, 'logs', args['log_path'], dataset.lower(), 'superpoint-brute-force'))
    
    batches = get_batches(log_path)
    
    if batches is None:
        print('No batches found. Exiting.')
        sys.exit(0)
    
    complete_inliers_npy = stack_batches(log_path, dataset, batches)
    
    np.save(os.path.join(log_path, f'superpoint-brute-force_similarity_matrix.npy'), complete_inliers_npy)
    
    plt.show()