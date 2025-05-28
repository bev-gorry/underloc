import os
import sys
import yaml
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm

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


def get_top_n_pairs(similarity_matrix, n):
    """Get the top n pairs of query and database images based on similarity scores."""
    d, q = similarity_matrix.shape
    
    row_indices = np.argmax(similarity_matrix, axis=0)
    col_indices = np.arange(q)
    similarities = similarity_matrix[row_indices, col_indices]
    
    df = pd.DataFrame({'query': col_indices, 'db': row_indices, 'similarity': similarities})
    
    df_top_n = df.nlargest(n, 'similarity')
    
    return df_top_n


def main():
    """
    This script retrieves the top n matches for each query image in the database using the similarity matrix. It serves as a
    sanity-check for the VPR methods - not as an evaluation tool for VPR methods and hierarchical VPR with local feature matching
    and reprojection error.
    """

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
    vpr_similarity_threshold = args['vpr_similarity_threshold']
    
    query_folder = str(args['query_folder'])
    database_folder = str(args['dbs_folder'])
    
    K = constants['K']
    visualize_top_n = constants['visualize_top_n']
    
    figures_dir = os.path.join(log_path, f'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # load and read rgb.txt files
    rgb_queries = os.path.join(benchmark, dataset, query_folder)
    rgb_dbs = os.path.join(benchmark, dataset, database_folder)
    rgb_txt_query = os.path.join(benchmark, dataset, query_folder, 'rgb.txt')
    rgb_txt_db = os.path.join(benchmark, dataset, database_folder, 'rgb.txt')
    
    # check that all required directories and files exist
    required_files = [os.path.join(benchmark, dataset), rgb_queries, rgb_dbs, rgb_txt_query, rgb_txt_db]
    hf.check_files(required_files)
    
    query_df = pd.read_csv(rgb_txt_query, sep=' ', header=None)
    db_df = pd.read_csv(rgb_txt_db, sep=' ', header=None)

    # get VPR methods to evaluate
    vpr_methods = constants['vpr_methods']
    method_folders = [f for f in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, f)) and f in vpr_methods]
    
    for method in method_folders:

        # if method != 'megaloc':                            ####################################### DEBUGGING
        #     continue
        
        # skip random as it does not produce a similarity matrix
        if method in ['random']:
            continue
        
        method_log_path = os.path.join(log_path, method)
        matches_file = os.path.join(method_log_path, f'{dataset.lower()}_{method}_top-{visualize_top_n}.csv')

        # load similarity matrix
        if method == 'superpoint-brute-force':
            similarity_matrix_path = os.path.join(method_log_path, f'{dataset.lower()}_superpoint_matrix.npy')
        elif method in ['superpoint-hierarchical', 'reprojection-error']:
            similarity_matrix_path = os.path.join(method_log_path, f'{method}_similarity_matrix_K{K}.npy')
        else:
            similarity_matrix_path = os.path.join(method_log_path, f'{method}_similarity_matrix.npy')
        
        if not os.path.exists(similarity_matrix_path):
                print(f"    ERROR! Similarity matrix not found: {similarity_matrix_path}")
                continue
        
        similarity_matrix = np.load(similarity_matrix_path)
        similarity_matrix = (similarity_matrix - np.nanmin(similarity_matrix)) / (np.nanmax(similarity_matrix) - np.nanmin(similarity_matrix))
        
        # retrieve the top n vpr matches
        top_n_results = get_top_n_pairs(similarity_matrix, visualize_top_n)
        
        # convert the query and database indices to integers
        matches_df = pd.DataFrame({
            'query': top_n_results['query'].astype(int),
            'database': top_n_results['db'].astype(int),
            'similarity': top_n_results['similarity']
        })

        # write the image indices and similarity scores to a csv file
        matches_df.to_csv(matches_file, index=False)
        print(f"Results saved to: {matches_file}")
        
        # create the output directory for figures and remove it if it already exists
        outdir = os.path.join(figures_dir, f'{method}_top-{visualize_top_n}')
        os.makedirs(outdir, exist_ok=True)
        
        for _, match in tqdm(top_n_results.iterrows(), total=visualize_top_n, desc=f'{method.upper()}'):
                                
            q_idx = int(match['query'])
            db_idx = int(match['db'])
            s_score = match['similarity']
                        
            query_img_name = query_df.iloc[q_idx, 1]
            db_img_name = db_df.iloc[db_idx, 1]
            
            query_img_path = os.path.join(benchmark, dataset, query_folder, query_img_name)
            db_img_path = os.path.join(benchmark, dataset, database_folder, db_img_name)
            
            query_img = plt.imread(query_img_path)
            db_img = plt.imread(db_img_path)
            
            fig, ax = plt.subplots(1, 2, figsize=(constants['fig_width'], constants['fig_height']))
            ax[0].imshow(query_img)
            ax[0].axis('off')
        
            ax[1].imshow(db_img)
            ax[1].axis('off')
            
            box_color = 'green' if s_score >= vpr_similarity_threshold else 'red'

            rect = patches.Rectangle((0, 0), query_img.shape[1], query_img.shape[0], linewidth=8, edgecolor=box_color, facecolor='none')
            ax[1].add_patch(rect)
            
            # write the threshold and similarity score
            ax[1].text(
                0.5, -0.10,
                f'Similarity: {s_score:.3f}, Threshold: {vpr_similarity_threshold:.2f}',
                fontsize=14,
                ha='center', va='top',
                transform=ax[1].transAxes,
                color=box_color
            )
            
            # write the filenames
            ax[0].text(
                0.5, -0.10,
                query_img_name,
                fontsize=14,
                ha='center', va='top',
                transform=ax[0].transAxes
            )
            ax[1].text(
                0.5, -0.20,
                db_img_name,
                fontsize=14,
                ha='center', va='top',
                transform=ax[1].transAxes
            )
            
            outfile = os.path.abspath(os.path.join(outdir, f'{dataset.lower()}_{method}_{q_idx}-{db_idx}_{s_score:.3f}.png'))
            fig.savefig(outfile, format='png', bbox_inches='tight')
                            
            plt.close(fig)

if __name__ == '__main__':
    main()
    
    
    