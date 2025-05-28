import os
import sys
import yaml
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import auc

root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from constants import constants
import utils.utils as hf

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],

    "text.latex.preamble": r"\usepackage{amsmath}",
    })

sns.set_context("paper", font_scale=constants['font_scale'], rc={"lines.linewidth": constants['linewidth'], "lines.markersize": constants['markersize']})


def get_valid_queries(gt_positives):
    """Get the number of query images that have at least one positive match in the database sequence."""
    num_valid_queries = 0
    num_queries = gt_positives.shape[1]
    
    for j in range(num_queries):
        if not np.any(gt_positives[:, j]):
            continue
        num_valid_queries += 1
        
    return num_valid_queries


def compute_vpr_metrics(gt_positives, similarity_matrix, similarity_threshold):
    """Compute the number of positives, true positives, false positives, false negatives, and true negatives for single match VPR."""
    P, TP, FP, FN, TN = 0, 0, 0, 0, 0

    num_queries = similarity_matrix.shape[1]
    num_valid_queries = get_valid_queries(gt_positives)

    for j in range(num_queries):
        if not np.any(gt_positives[:, j]):
            continue
        
        top_match_idx = np.argmax(similarity_matrix[:, j])
        top_match = similarity_matrix[top_match_idx, j]
        
        if top_match >= similarity_threshold:
            P += 1
            if gt_positives[top_match_idx, j] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if gt_positives[top_match_idx, j] == 1:
                FN += 1
            else:
                TN += 1
    
    assert TP + FP + FN + TN == num_valid_queries

    return P, TP, FP, FN, TN


def compute_precisions_recalls(method, gt_positives, similarity_matrix, similarity_thresholds):
    """Compute precision and recall values over the given similarity thresholds."""
    precisions = []
    recalls = []
    
    # only consider queries with at least one positive match in the database sequence
    num_valid_queries = get_valid_queries(gt_positives)
        
    for similarity_thresh in tqdm(similarity_thresholds, desc=f'PR - {method.upper()}'):
        P, TP_single, FP_single, FN_single, TN_single = compute_vpr_metrics(gt_positives, similarity_matrix, similarity_thresh)
        
        if P == 0:
            continue
        precision = TP_single / float(TP_single + FP_single)
        recall = TP_single / num_valid_queries
        precisions.append(precision * 100)
        recalls.append(recall * 100)
    
    # compute the area under the precision-recall curve (AUPRC)
    precisions_auc = [precision/100 for precision in precisions]
    recalls_auc = [recall/100 for recall in recalls]
    auprc = auc(recalls_auc, precisions_auc)
    
    return precisions, recalls, auprc


def recall_at_k(similarity_matrix, gt_positives, k=1):
    """Compute recall at k for the given similarity matrix and ground truth positives."""
    num_queries = similarity_matrix.shape[1]
    num_valid_queries = get_valid_queries(gt_positives)
    num_successful_queries = 0
    top_k_indices_dict = {}
    
    for i in range(num_queries):
        if not np.any(gt_positives[:, i]):
            continue
        
        top_k_indices = np.argsort(similarity_matrix[:, i])[-k:]
                
        top_k_indices_dict[i] = top_k_indices
        top_k = gt_positives[top_k_indices, i]

        if np.any(top_k):
            num_successful_queries += 1
    
    recall = num_successful_queries / num_valid_queries if num_valid_queries > 0 else np.nan

    return recall * 100, top_k_indices_dict


def plot_top_k_per_query(gt_positives, similarity_matrix, k_per_query, threshold=0.1):
    """Plot the top k matches per query using the similarity matrix. Filter out matches with a reprojection error greater than the threshold."""
    true_x, true_y = [], []
    false_x, false_y = [], []

    # iterate over each query (column) in the similarity matrix
    for query_idx in range(similarity_matrix.shape[1]):        
        best_match_indices = np.argsort(similarity_matrix[:, query_idx])[-k_per_query:]

        for best_match_idx in best_match_indices:
            
            # exclude queries with no positive matches
            if not np.any(gt_positives[:, query_idx]):
                continue
            
            # filter out matches with a reprojection error greater than the threshold (10 pixels)
            if similarity_matrix[best_match_idx, query_idx] < threshold:
                continue
            
            # if the match is a true positive (in gt_positives), it will be marked as a green cross
            if gt_positives[best_match_idx, query_idx]:
                true_x.append(query_idx)
                true_y.append(best_match_idx)
            
            # if the match is a false positive, it will be marked as a red cross
            else:
                false_x.append(query_idx)
                false_y.append(best_match_idx)

    fig = plt.figure(figsize=(constants['fig_width'], constants['fig_height']))
    plt.imshow(gt_positives, cmap='gray')
    plt.scatter(true_x, true_y, color='green', marker='x', s=500, zorder=10, label="True Positives")
    plt.scatter(false_x, false_y, color='red', marker='x', s=300, label="False Positives")
    
    ax = plt.gca()
    
    # manually set the x and y ticks
    max_x = similarity_matrix.shape[1] - 1
    max_y = similarity_matrix.shape[0] - 1
    plt.xlim(0, max_x)
    plt.ylim(max_y, 0)
    plt.xticks([])
    plt.yticks([max_y, 0])
    ax.text(1.085,
        1.015,
        f" {max_x}",
        ha='right',
        va='top',
        color='black',
        transform=ax.transAxes,
        clip_on=False,
        fontsize=plt.rcParams['xtick.labelsize'])
    
    plt.title(f"Top-{k_per_query} Matches per Query")
    plt.xlabel("Query Index")
    plt.ylabel("Database Index")
    plt.legend()
    
    return fig


def main():
        
    # load arguments from yaml file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml_file', type=str, default='arguments/args_demo.yaml',
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
    
    loc_rad = args['vpr_loc_radius']
    K = constants['K']
    k_per_query = constants['k_per_query']
    reprojection_error_threshold = constants['reprojection_error_threshold']
    
    if not os.path.exists(log_path):
        print(f"    ERROR! Log path not found: {log_path}")
        print(f'Nothing to evalaute. Please run << pixi run pipeline >> first to generate results.')
        exit(0)
    
    # load groundtruth matrix
    gt_matrix_path = os.path.join(benchmark, dataset, f'groundtruth_{query_folder}-{database_folder}.npy')
    if not os.path.exists(gt_matrix_path):
        print(f"    ERROR! Ground truth matrix not found: {gt_matrix_path}")
        print(f'Please run << ARGUMENT={arguments.yaml_file} pixi run create_groundtruth >> first to generate the groundtruth matrix.')
        exit(0)
    gt = np.load(gt_matrix_path)
    
    # binary gt matrix
    gt_positives = hf.get_gt_positives(gt, loc_rad)
    
    # get VPR methods to evaluate
    vpr_methods = constants['vpr_methods']
    method_folders = [f for f in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, f)) and f in vpr_methods]
    
    palette = sns.color_palette("tab10", len(vpr_methods))
    method_colors = dict(zip(vpr_methods, palette))
    
    # evaluation bounds
    k_values = constants['k_values']
    results = {method: {"precisions": [], "recalls": [], "recall_at_k": []} for method in method_folders}

    # output folder to save results
    eval_dir = os.path.join(log_path, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # prepare to write to evaluation file
    eval_file = os.path.join(eval_dir, 'evaluation.txt')
    f_mode = 'a' if os.path.exists(eval_file) else 'w'

    with open(eval_file, f_mode) as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'***** {dataset} ***** {current_time} *****')
        f.write(f'\nLocalisation Radius: {loc_rad}')
            
        for method in method_folders:            
            # if method not in ['superpoint-brute-force']:                            ####################################### DEBUGGING
            #     continue
            # visualize = True
            
            f.write(f'\n\n{method.upper()} - ')
            method_log_path = os.path.join(log_path, method)

            # load random recalls
            if method == 'random':
                random_recalls_path = os.path.join(method_log_path, 'random_recalls.npy')
                if not os.path.exists(random_recalls_path):
                    print(f"    ERROR! Random recalls file not found: {random_recalls_path}")
                    f.write(f"SKIPPED (random recalls file not found: {random_recalls_path})")
                    continue
                random_recalls = np.load(random_recalls_path)
                results[method]["recall_at_k"] = random_recalls.tolist()
                f.write(f'Recall@{k_values}: {random_recalls.tolist()}\n')
                for recall in random_recalls.tolist():
                    f.write(f'{recall:.3f} & ')
                continue
            
            # load similarity matrix
            if method in ['superpoint-hierarchical', 'reprojection-error']:
                similarity_matrix_path = os.path.join(method_log_path, f'{method}_similarity_matrix_K{K}.npy')
            else:
                similarity_matrix_path = os.path.join(method_log_path, f'{method}_similarity_matrix.npy')
            
            if not os.path.exists(similarity_matrix_path):
                print(f"    ERROR! Similarity matrix not found: {similarity_matrix_path}")
                f.write(f"SKIPPED (similarity matrix not found: {similarity_matrix_path})")
                continue
            similarity_matrix = np.load(similarity_matrix_path)
                        
            if method == 'reprojection-error':
                similarity_matrix = 1 / similarity_matrix 

            # plot the similarity matrix
            s_fig = hf.plot_matrix(similarity_matrix, f'{dataset}: {method.upper()} Similarity Matrix', cmap='viridis')
            s_fig.savefig(os.path.join(eval_dir, f'{dataset.lower()}_similarity_matrix_{method}.png'), format='png', bbox_inches='tight')
            
            # compute precision and recall values
            similarity_thresholds = np.linspace(np.nanmin(similarity_matrix), np.nanmax(similarity_matrix), 100)
            precisions, recalls, _ = compute_precisions_recalls(method, gt_positives, similarity_matrix, similarity_thresholds)
            results[method]["precisions"] = precisions
            results[method]["recalls"] = recalls

            # plot the top k matches per query, fitlering with reprojection error
            if method == 'reprojection-error':
                best_matches = plot_top_k_per_query(gt_positives, similarity_matrix, k_per_query=k_per_query, threshold=reprojection_error_threshold)
                best_matches.savefig(os.path.join(eval_dir, f'{dataset.lower()}_top-{k_per_query}_{method}.png'), format='png', bbox_inches='tight')
            
            # compute recall@K (except for superpoint-hierarchical)
            if method == 'superpoint-hierarchical':
                continue
            
            recall_k_values = []
            for k in k_values:
                recall_k, _ = recall_at_k(similarity_matrix, gt_positives, k=k)
                recall_k_values.append(recall_k)
            
            results[method]["recall_at_k"] = recall_k_values

            f.write(f'Recall@{k_values}: {recall_k_values}\n')
            for recall in recall_k_values:
                f.write(f'{recall:.1f} & ')
        f.write('\n----------\n\n')

    # plot precision-recall curve
    plt.figure(figsize=(constants['fig_width'], constants['fig_height']))

    all_recalls = [recall for data in results.values() for recall in data["recalls"]]

    if any(recall > 75 for recall in all_recalls):
        plt.xlim(0, 100)
    elif any(recall > 50 for recall in all_recalls):
        plt.xlim(0, 75)
    elif any(recall > 25 for recall in all_recalls):
        plt.xlim(0, 50)
    else:
        plt.xlim(0, 25)
    
    plt.ylim(0, 110)
    plt.xlabel(r'Recall (\%)')
    plt.ylabel(r'Precision (\%)')
    plt.title(f'{dataset}: Precision-Recall Curve')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(25))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(25))
    
    yticks = np.arange(0, 125, 25)
    plt.gca().set_yticks(yticks)
    ytick_labels = ["" if y == 0 else str(y) for y in yticks]
    plt.gca().set_yticklabels(ytick_labels)
    
    plt.grid(which='minor', linestyle='--', linewidth=0.25, color=(0.8, 0.8, 0.8, 0.5))
    plt.grid(which='major', linestyle='-', linewidth=0.25, color=(0.8, 0.8, 0.8, 0.5))
    
    for method, data in results.items():
        if method in ['random', 'reprojection-error']:
            continue
        
        if len(data["recalls"]) > 0:
            plt.plot(data["recalls"], data["precisions"], marker='o', markersize=0.5, label=f'{method.upper()}', color=method_colors[method])
    
    plt.legend()
    plt.savefig(os.path.join(eval_dir, f'{dataset.lower()}_precision_recall_curve_LR-{loc_rad}.png'), format='png', bbox_inches='tight')

    # plot recall@K
    plt.figure(figsize=(constants['fig_width'], constants['fig_height']))
    plt.xlim(0, max(k_values) * 1.1)
    plt.ylim(0, 110)
    plt.xlabel('K')
    plt.ylabel(r'Recall (\%)')
    plt.title(f'{dataset}: Recall@K')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(25))
    
    yticks = np.arange(0, 125, 25)
    plt.gca().set_yticks(yticks)
    ytick_labels = ["" if y == 0 else str(y) for y in yticks]
    plt.gca().set_yticklabels(ytick_labels)
    
    plt.grid(which='major', linestyle='-', linewidth=0.25, color=(0.8, 0.8, 0.8, 0.5))
    
    for method, data in results.items():
        if method in ['superpoint-hierarchical', 'reprojection-error']:
            continue
        if len(data["recall_at_k"]) > 0:
            plt.plot(k_values, data["recall_at_k"], marker='o', label=f'{method.upper()}', color=method_colors[method])
    plt.legend()
    plt.savefig(os.path.join(eval_dir, f'{dataset.lower()}_recall_at_k_LR-{loc_rad}.png'), format='png', bbox_inches='tight')
   
    if visualize:
        plt.show()
    plt.close('all')

if __name__ == '__main__':
    main()