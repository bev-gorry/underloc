
import os
import cv2
import yaml
import time
import argparse
import requests

import pandas as pd

from tqdm import tqdm
from PIL import Image
from math import sqrt, radians
from datetime import datetime
from transforms3d.euler import euler2quat


def get_sequences(df):
    """Determine unique sequences from the dataframe based on the 'path_best' column."""
    
    filepaths = df['path_best'].tolist()
    common_path = os.path.commonpath(filepaths)
    
    filepaths = [os.path.dirname(url)[len(common_path)+2:] for url in filepaths]
    
    sequences = set(filepaths)
        
    return sequences

def extract_sequence(url, sequences):
    """Extract which sequence the URL belongs to based on the sequences list."""
    
    filepath = os.path.dirname(url)
    
    for seq in sequences:
        if seq in filepath:
            return seq
    return None

def download_images(df, rgb_path, resolution):
    """Download images from URLs in the dataframe and save them to the specified path."""

    urls = df['path_best'].tolist()
    
    for _, url in tqdm(enumerate(urls), total=len(urls), desc='Downloading and resizing images'):
        url = url.strip()
        if not url:
            continue
        
        filename = os.path.join(rgb_path, os.path.basename(url))
        if os.path.exists(filename):
            continue
        
        try:
            # request the image from the url
            response = requests.get(url, stream=True, timeout=20)
            response.raise_for_status()
            
            img = Image.open(response.raw)
            
            # resize the image if resolution is provided
            if resolution[0]:
                scaled_height = int(sqrt(img.size[1] / img.size[0] * resolution[0] * resolution[1]))
                scaled_width = int(resolution[0] * resolution[1] / scaled_height)

                resized_img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
                resized_img.save(filename)
            else:
                img.save(filename)

        except requests.exceptions.RequestException as e:
            tqdm.write(f"Failed to download {url}: {e}")
        except Exception as e:
            tqdm.write(f"Failed to process {url}: {e}")
        time.sleep(0.1)

    return df

def calculate_frame_rate(df):
    """Calculate the frame rate based on the timestamps in the dataframe."""
    
    if 'pose.timestamp' not in df.columns:
        print("pose.timestamp column not found in dataframe.")
        return None
    
    df['timestamp'] = pd.to_datetime(df['pose.timestamp'])
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    if df['time_diff'].notna().any():
        avg_time_diff = df['time_diff'].mean()
        fps = 1 / avg_time_diff
        return fps
    else:
        return None


def write_calibration_yaml(sequence_path, camera_model, fx, fy, k1=0, k2=0, p1=0, p2=0, k3=0, rgb_hz=10):
    """Write the camera calibration parameters to a calibration.yaml file."""
    
    calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
    
    if os.path.exists(calibration_yaml):
        print(f"Calibration file already exists: {calibration_yaml}")
        return

    # get the image size from the first image
    rgb_path = os.path.join(sequence_path, 'rgb')    
    rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
    image_0 = cv2.imread(os.path.join(rgb_path, rgb_files[0]))
    h, w, _ = image_0.shape

    yaml_content_lines = [
        "%YAML:1.0",
        "",
        "# Camera calibration and distortion parameters",
        "Camera.model: " + camera_model,
        "",
        "Camera.fx: " + str(fx),
        "Camera.fy: " + str(fy),
        "Camera.cx: " + str(w/2),
        "Camera.cy: " + str(h/2),
        "",
        "Camera.k1: " + str(k1),
        "Camera.k2: " + str(k2),
        "Camera.p1: " + str(p1),
        "Camera.p2: " + str(p2),
        "Camera.k3: " + str(k3),
        "",
        "Camera.w: " + str(w),
        "Camera.h: " + str(h),
        "",
        "# Camera frames per second",
        "Camera.fps: " + str(rgb_hz)
    ]

    with open(calibration_yaml, 'w') as file:
        for line in yaml_content_lines:
            file.write(f"{line}\n")

    print(f"Created calibration file: {calibration_yaml}")


def write_rgb_txt(df, sequence_path):
    """Write timestamps and image paths to an rgb.txt file."""
    
    rgb_txt_path = os.path.join(sequence_path, 'rgb.txt')
    
    if os.path.exists(rgb_txt_path):
        print(f"RGB file already exists: {rgb_txt_path}")
        return
    
    df['timestamp_seconds'] = pd.to_datetime(df['pose.timestamp']).map(lambda ts: (ts - datetime(1970, 1, 1)).total_seconds())
    df['image_name'] = df['path_best'].apply(os.path.basename)
    
    # format the output line as timestamp rgb/image_name
    df['output_line'] = df['timestamp_seconds'].map(lambda ts: f"{ts:.6f}") + " rgb/" + df['image_name']
    df['output_line'].to_csv(rgb_txt_path, index=False, header=False)
    
    print(f"Created rgb.txt file: {rgb_txt_path}")


def write_groundtruth(df, sequence_path, output_format='csv'):
    """Write groundtruth data to a file. Supports both CSV and TXT formats."""
    
    output_format = output_format.lower()
    if output_format not in ['csv', 'txt']:
        raise ValueError("Invalid output_format. Must be 'csv' or 'txt'.")

    file_extension = f".{output_format}"
    groundtruth_path = os.path.join(sequence_path, f'groundtruth{file_extension}')

    if os.path.exists(groundtruth_path):
        print(f"Groundtruth file already exists: {groundtruth_path}")
        if output_format == 'csv':
            return pd.read_csv(groundtruth_path)
        else:
            return
    
    output_data = []

    for _, row in df.iterrows():
        # extract pose data if available, default 0
        tx = row.get('pose.lat', 0)
        ty = row.get('pose.lon', 0)
        tz = -row.get('pose.alt', 0) # negating altitude to follow common convention
        
        # convert roll, pitch, yaw to radians for quaternion computation
        roll = radians(row.get('pose.data.roll', 0))
        pitch = radians(row.get('pose.data.pitch', 0))
        yaw = radians(row.get('pose.data.heading', 0))
        
        qw, qx, qy, qz = euler2quat(roll, pitch, yaw)  # format for quaternion is wxyz
        
        timestamp = row.get('timestamp_start', row.get('pose.timestamp'))
        image_name = row.get('path_best', '').split('/')[-1]
        
        # construct the output data
        if output_format == 'csv':
            output_data.append({
                'timestamp': timestamp,
                'image': image_name,
                'tx': tx,
                'ty': ty,
                'tz': tz,
                'qx': qx,
                'qy': qy,
                'qz': qz,
                'qw': qw,
            })
        elif output_format == 'txt':
            if isinstance(timestamp, str):
                try:
                    timestamp_seconds = pd.to_datetime(timestamp).timestamp()
                except ValueError:
                    print(f"WARNING: Could not parse timestamp '{timestamp}'. Skipping entry.")
                    continue
            elif isinstance(timestamp, pd.Timestamp):
                timestamp_seconds = timestamp.timestamp()
            else:
                print(f"WARNING: Unexpected timestamp format '{timestamp}'. Skipping entry.")
                continue
            output_data.append(f"{timestamp_seconds:.4f} {tx:.4f} {ty:.4f} {tz:.4f} {qw:.4f} {qx:.4f} {qy:.4f} {qz:.4f}\n")

    # write the output data to the specified file
    if output_format == 'csv':
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(groundtruth_path, index=False)
        
        print(f"Created groundtruth.csv file: {groundtruth_path}")
        return output_df
    
    elif output_format == 'txt':
        with open(groundtruth_path, 'w') as f:
            f.writelines(output_data)
            
        print(f"Created groundtruth.txt file: {groundtruth_path}")
        return None

def main():

    # load arguments from yaml file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml_file', type=str, default='../arguments/args_dataset.yaml',
                        help='yaml file containing arguments')
    arguments = parser.parse_args()
        
    with open (arguments.yaml_file, 'r') as f:
        args = yaml.safe_load(f)
        
    csv_file = os.path.abspath(args['csv_file'])
    dataset_path = os.path.abspath(args['dataset_path'])
    resolution = args['resolution'] if 'resolution' in args else None
    
    # ensure that resolution is in the format W,H
    try:
        width, height = map(int, resolution.split(','))
    except ValueError:
        print("Invalid resolution format. Please use the format W,H, e.g., 640,480.")
        print("Using original image resolution...")
        width, height = None, None
    
    print(f'Creating dataset in {dataset_path} from {csv_file}.\n')
    os.makedirs(dataset_path, exist_ok=True)
    
    # read the CSV file and sort rows by timestamp in case they are not already
    df = pd.read_csv(csv_file)
    df = df.sort_values('pose.timestamp')
    
    # get a list of unique sequences from the dataframe
    sequences = get_sequences(df)
    df['sequence'] = df['path_best'].apply(extract_sequence, sequences=sequences)
    grouped = dict(tuple(df.groupby('sequence')))
    
    print(f"Found {len(grouped)} sequences.")
    print(f"Sequences: {grouped.keys()}")
    
    for seq, sequence_df in grouped.items():
        seq_name = seq.replace('/', '_')
        print(f"\nSequence: {seq_name}, Number of Images: {len(sequence_df)}")
        
        rgb_path = os.path.join(os.path.join(dataset_path, seq_name), 'rgb')
        os.makedirs(rgb_path, exist_ok=True)
        
        # get the images that are already downloaded
        existing_images = set(os.listdir(os.path.join(dataset_path, seq_name, 'rgb')))
        
        if len(sequence_df) == len(existing_images):
            print(f"Images already downloaded: {len(existing_images)}")
            continue

        # try downloading images if some failed until all images are downloaded or max retries reached
        num_tries = 0
        while (len(existing_images) < len(sequence_df)) and (num_tries < 3):
            sequence_df = download_images(sequence_df, rgb_path, [width, height])
            existing_images = set(os.listdir(os.path.join(dataset_path, seq_name, 'rgb')))
            num_tries += 1
            
            print(f"Downloaded {len(existing_images)} images.")
            print(f"Expected {len(sequence_df)} images.")
            print(f"Missing {len(sequence_df)-len(existing_images)} images.")
            
            print(f"Retrying: {num_tries} of 3.\n") if len(sequence_df)-len(existing_images) else print("")
        
        # remove images that failed to download after the maximum number of retries
        if len(existing_images) < len(sequence_df):
            print(f"Removing {len(sequence_df)-len(existing_images)} images.")
            print(f"Sequence length: {len(sequence_df)}\n")
            sequence_df = sequence_df[sequence_df['path_best'].apply(lambda x: os.path.basename(x) in existing_images)]
        
        rgb_hz = calculate_frame_rate(sequence_df)
        
        # write to files
        write_calibration_yaml(os.path.join(dataset_path, seq_name), 'UNKNOWN', 200, 200, rgb_hz=rgb_hz)
        write_rgb_txt(sequence_df, os.path.join(dataset_path, seq_name))
        write_groundtruth(sequence_df, os.path.join(dataset_path, seq_name), output_format='csv')
        write_groundtruth(sequence_df, os.path.join(dataset_path, seq_name), output_format='txt')

if __name__ == '__main__':
    main()