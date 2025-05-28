### Usage: python underloc_utils.py <function> <yaml_file>

import os
import sys
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from haversine import haversine, Unit


root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from constants import constants

import utils.utils as hf

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],

    "text.latex.preamble": r"\usepackage{amsmath}",})

sns.set_context("paper", font_scale=constants['font_scale'], rc={"lines.linewidth": constants['linewidth'], "lines.markersize": constants['markersize']})


# ---------------------------------------------------------------------------------------------------- #

def plot_trajectories(yaml_file):
    """Plots the GPS trajectory of the sequences in the dataset."""
    two_dim = True
    
    _, dataset, dataset_path = hf.parse_yaml(yaml_file)
    hf.check_dataset(dataset_path)
        
    sequences = [seq for seq in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, seq)) and not seq.startswith('0_')]
    
    # first sequence is the reference (blue), second is the query (orange)
    sequences.sort()
    
    print(f'DATASET PATH: {dataset_path}')
    print(f'SEQUENCES: {sequences}')
    
    if len(sequences) == 0:
        print('No sequences found in the dataset.')
        return
        
    ref_sequence = hf.read_ground_truth(os.path.join(dataset_path, sequences[0], 'groundtruth.csv'))
    ref_key = next(iter(ref_sequence))
    lat_ref, lon_ref, alt_ref = ref_sequence[ref_key][1], ref_sequence[ref_key][2], ref_sequence[ref_key][3]
    
    fig = plt.figure(figsize=(12,12))
    
    if two_dim:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')
    
    palette = sns.color_palette("tab10", len(sequences))
    colors = dict(zip(sequences, palette))
    
    alpha_decrement = 1 / (len(sequences) + 1) # factor to decrease alpha for each sequence based on the number of sequences
    alpha = 1
    
    for sequence in sequences:
        
        print(f'seq: {sequence}, color: {colors[sequence]}')
        
        df = hf.read_ground_truth(os.path.join(dataset_path, sequence, 'groundtruth.csv'))
        
        tx, ty, tz = [], [], []
        
        for coords in df.values():
            lat, lon, alt = coords[1], coords[2], coords[3]
            
            if dataset.lower() == 'eiffel':
                x, y = lat, lon
            else:
                x, y = hf.latlon_to_meters(lat_ref, lon_ref, lat, lon)
            
            tx.append(x)
            ty.append(y)
            tz.append(alt)

        if two_dim:
            ax.plot(tx, ty, marker='o', markersize=6, linewidth=2, label=sequence.upper(), color=colors[sequence], alpha=alpha)
        else:
            ax.plot(tx, ty, tz, marker='o', markersize=6, linewidth=2, label=sequence.upper(), color=colors[sequence], alpha=alpha)

        alpha -= alpha_decrement

        tx = np.array(tx)
        ty = np.array(ty)
        tz = np.array(tz)

        x_range = tx.max() - tx.min()
        y_range = ty.max() - ty.min()
        
        def choose_locator(range_value, max_ticks=10, minor_factor=0.5):
            major_spacing = range_value / max_ticks
            minor_spacing = major_spacing / minor_factor
            return major_spacing, minor_spacing

        x_major, x_minor = choose_locator(x_range)
        y_major, y_minor = choose_locator(y_range)
    
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(x_minor))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(y_minor))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(x_major))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(y_major))
    plt.tick_params(axis='both', which='major', pad=20, labelsize=14)
    
    plt.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.axis('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.title(f'GPS Trajectory - {dataset.upper()}')
    plt.legend()

    # save plot of gps trajectory
    outfile = os.path.join(dataset_path, f'gps_trajectory_{dataset.lower()}.png')
    fig.savefig(outfile, format='png', bbox_inches='tight')
    print(f'Figure saved: {outfile}')
    
    plt.show()
    plt.close('all')

# ---------------------------------------------------------------------------------------------------- #

def get_sequence_duration(csv_path):
    """Computes the duration of the sequence based on timestamps in the CSV file."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    
    return duration

def compute_trajectory_distance(csv_file):
    """Computes the total distance of the trajectory based on latitude and longitude coordinates."""
    df = pd.read_csv(csv_file)

    if 'tx' not in df.columns or 'ty' not in df.columns:
        raise ValueError("CSV must contain 'tx' and 'ty' columns for latitude and longitude.")

    latitudes = df['tx'].values
    longitudes = df['ty'].values

    total_distance = sum(
        haversine((latitudes[i], longitudes[i]), (latitudes[i+1], longitudes[i+1]), unit=Unit.METERS)
        for i in range(len(latitudes) - 1)
    )

    return total_distance

def dataset_info(yaml_file):
    """Gets dataset info using benchmark and dataset arguments from a yaml file."""
    
    benchmark, dataset, dataset_path = hf.parse_yaml(yaml_file)
    hf.check_dataset(dataset_path)
    
    output_file = os.path.join(benchmark, dataset, 'dataset_info.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f'Writing output to: {output_file}')
    
    # get a list of sequences in the dataset excluding those starting with '0_'
    sequences = [seq for seq in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, seq)) and not seq.startswith('0_')]
    
    required_files = [os.path.join(dataset_path, seq, 'groundtruth.csv') for seq in sequences]
    hf.check_files(required_files)
    
    with open(output_file, 'w') as f:
        f.write(f'\n----- {dataset} -----\n')
        print(f'\n----- {dataset} -----')
        
        for sequence in sequences:
            print(f'Sequence: {dataset}/{sequence}')
            f.write(f'Sequence: {dataset}/{sequence}\n')
            f.write(f'    Duration & Distance\n')
            sequence_path = os.path.join(dataset_path, sequence)
            csv_file = os.path.join(sequence_path, 'groundtruth.csv')
            
            duration = get_sequence_duration(csv_file)
            distance = compute_trajectory_distance(csv_file)
            print(f'    DURATION: {duration}')
            print(f'    DISTANCE: {distance} meters')
            f.write(f'    {duration} & {distance} meters\n')

# ---------------------------------------------------------------------------------------------------- #

def create_gps_distance_matrix(gt_query, gt_db, latlon=True):
    """Creates a ground truth matrix based on the query and database ground truth data."""
    
    query_keys = list(gt_query.keys())
    db_keys = list(gt_db.keys())
    
    gt = np.zeros((len(db_keys), len(query_keys)), dtype=float)
    
    # determine the distance in meters from lat/lon coordinates
    for i, img_db in enumerate(db_keys):
        _, x_db, y_db, z_db = gt_db[img_db]
        for j, img_q in enumerate(query_keys):
            _, x_q, y_q, z_q = gt_query[img_q]
            
            if latlon:
                distance = haversine((x_q, y_q), (x_db, y_db), unit=Unit.METERS)
            else:
                distance = np.sqrt((x_q - x_db) ** 2 + (y_q - y_db) ** 2 + (z_q - z_db) ** 2)
            
            gt[i, j] = distance
            
    return gt

def create_groundtruth(yaml_file):
    with open (yaml_file, 'r') as f:
        args = yaml.safe_load(f)
        
    benchmark = hf.parse_yaml(yaml_file)[0]
    
    required_keys = ['dataset', 'log_path', 'visualize', 'query_folder', 'dbs_folder', 'vpr_loc_radius', ]
    for key in required_keys:
        if key not in args:
            print(f'    ERROR! Missing required argument from {yaml_file}: {key}')
            exit(0)
    
    dataset = args['dataset'].upper()
    log_path = os.path.abspath(os.path.join(root, 'logs', args['log_path'], dataset.lower()))
    visualize = args['visualize']
    
    loc_rad = args['vpr_loc_radius']

    query_folder = str(args['query_folder'])
    database_folder = str(args['dbs_folder'])
    
    hf.check_dataset(os.path.join(benchmark, dataset))

    # find and read groundtruth files
    gt_query_csv = os.path.join(benchmark, dataset, query_folder, 'groundtruth.csv')
    gt_db_csv = os.path.join(benchmark, dataset, database_folder, 'groundtruth.csv')
    
    required_files = [gt_query_csv, gt_db_csv]
    hf.check_files(required_files)

    gt_db = hf.read_ground_truth(gt_db_csv)
    gt_query = hf.read_ground_truth(gt_query_csv)
    
    latlon = False if dataset == 'EIFFEL' else True
    
    filename = os.path.join(benchmark, dataset, f'groundtruth_{query_folder}-{database_folder}.npy')
    
    eval_dir = os.path.join(log_path, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # only create ground truth matrix if it does not exist
    if os.path.exists(filename):
        gt = np.load(filename)
        print(f'Loaded GT: {filename}')
    else:
        gt = create_gps_distance_matrix(gt_query, gt_db, latlon)
        np.save(filename, gt)
        print(f'Saved GT: {filename}')
    
        # save plot of binary gt matix
        gt_fig = hf.plot_matrix(gt, f'{dataset}: Ground Truth Distance Matrix', cmap='viridis')
        gt_fig.savefig(os.path.join(eval_dir, f'{dataset.lower()}_ground_truth.png'), format='png', bbox_inches='tight')
    
    # compute the binary ground truth matrix in case the localization radius has changed
    gt_positives = hf.get_gt_positives(gt, loc_rad)
    gt_binary = hf.plot_matrix(gt_positives, f'{dataset} Ground Truth - Localisation Radius {loc_rad}', cmap='gray')
    gt_binary.savefig(os.path.join(eval_dir, f'{dataset.lower()}_ground_truth_binary_{query_folder}-{database_folder}_LR-{loc_rad}.png'), format='png', bbox_inches='tight')
    
    if visualize:
        plt.show()
    
    plt.close('all')

# ---------------------------------------------------------------------------------------------------- #


if __name__ == '__main__':

    if len(sys.argv) >= 3:
        function_name = sys.argv[1]
        yaml_file = sys.argv[2]
        
        if function_name == 'plot_trajectories':
            plot_trajectories(yaml_file)
        if function_name == 'dataset_info':
            dataset_info(yaml_file)
        if function_name == 'create_groundtruth':
            create_groundtruth(yaml_file)