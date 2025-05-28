import os
import sys
import yaml
import subprocess

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from constants import constants

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],

    "text.latex.preamble": r"\usepackage{amsmath}",
    })

sns.set_context("paper", font_scale=constants['font_scale'], rc={"lines.linewidth": constants['linewidth'], "lines.markersize": constants['markersize']})


def run_script(command, args, cwd=None):
    """Run a script with arguments and optional working directory."""
    try:
        result = subprocess.run(
            [command] + args,
            check=True,
            cwd=cwd
        )
        print(f'{command} completed successfully.')
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f'Error running {command}: {e}')
        return e.returncode


def downsample_images(input_dir, output_dir, resolution_scale):
    """Downsample images in a directory."""
    
    print(f'            Downsampling images in {input_dir} to {output_dir} with resolution scale {resolution_scale}...')
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))])

    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)
        output_img_path = os.path.join(output_dir, img_file)
        
        try:
            with Image.open(img_path) as img:
                scaled_width = int(img.size[0] * resolution_scale)

                if scaled_width % 2 != 0:
                    scaled_width -= 1
                
                scaled_height = int(scaled_width * img.size[1] / img.size[0])
                
                resized_img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
                resized_img.save(output_img_path)
        except Exception as e:
            print(f'    ERROR: Could not process image {img_path} ---- {e}')
            continue
    return output_dir

def plot_matrix(matrix, title, cmap='viridis'):
    """Plot a matrix with the specified title and colormap."""
    fig = plt.figure(figsize=(constants['fig_width'], constants['fig_height']))
    plt.imshow(matrix, cmap=cmap)
    
    max_x = matrix.shape[1] - 1
    max_y = matrix.shape[0] - 1
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.xticks([0, max_x])
    plt.yticks([max_y])
    
    plt.title(title)
    
    if cmap == 'viridis':
        plt.colorbar()
        
    return fig

def read_ground_truth(file_path):
    """Reads the ground truth data from a CSV file and returns a dictionary with image names as keys and their corresponding timestamp and coordinates."""
    gt_df = pd.read_csv(file_path)
    
    ground_truth = {}
    for _, row in gt_df.iterrows():
        timestamp = row['timestamp']
        ground_truth[row['image']] = (timestamp, row['tx'], row['ty'], row['tz'])
    
    return ground_truth

def get_gt_positives(gt, threshold):
    """Creates a binary ground truth matrix based on a distance threshold."""
    gt_positives = np.zeros_like(gt)
    gt_positives[gt <= threshold] = 1
            
    return gt_positives

def latlon_to_meters(lat_ref, lon_ref, lat, lon):
    """Converts latitude and longitude to meters using the Haversine formula."""
    R = 6378137  # Earth radius in meters
    
    lat_ref, lon_ref, lat, lon = map(np.radians, [lat_ref, lon_ref, lat, lon])
    
    d_north = (lat - lat_ref) * R  # North-South distance
    d_east = (lon - lon_ref) * R * np.cos(lat_ref)
    
    return d_east, d_north

def check_files(required_files):
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("    ERROR! The following required directories or files are missing:")
        for f in missing_files:
            print(f"      - {f}")
        exit(0)

def parse_yaml(yaml_file):
    with open (yaml_file, 'r') as f:
        args = yaml.safe_load(f)
    
    # get the dataset path from the yaml file, whether this is a dataset yaml or experiment yaml
    if 'benchmark' in args and 'dataset' in args:
        if args['benchmark'].startswith("../../"):
            benchmark = os.path.abspath(os.path.join(root, args['benchmark'][3:]))
        else:
            benchmark = os.path.abspath(args['benchmark'])
        dataset = args['dataset'].upper()
        dataset_path = os.path.join(benchmark, dataset)
    elif 'dataset_path' in args:
        if args['dataset_path'].startswith("../../"):
            dataset_path = os.path.abspath(os.path.join(root, args['dataset_path'][3:]))
        else:
            dataset_path = os.path.abspath(args['dataset_path'])
        benchmark, dataset = os.path.split(dataset_path)
    else:
        raise ValueError("yaml file arguments must include either 'benchmark' and 'dataset' or 'dataset_path'.")
    
    return benchmark, dataset, dataset_path

def check_dataset(dataset_path):
    """Check if the benchmark path exists."""
    if not os.path.exists(dataset_path):
        print(f"    ERROR! The dataset path does not exist: {dataset_path}")
        print(f'Please check your benchmark location and dataset name or run << pixi run download_dataset >> to download a dataset.')
        exit(0)