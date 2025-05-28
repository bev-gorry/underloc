### The Eiffel Tower dataset can be manually downloaded from this link: https://www.seanoe.org/data/00810/92226/


import os
import cv2
import yaml
import time
import shutil
import argparse
import subprocess
import pandas as pd

from PIL import Image
from math import sqrt
from tqdm import tqdm
from datetime import datetime


def download_with_wget(url, output_path):
    try:
        subprocess.run(['wget', '-c', url, '-P', os.path.dirname(output_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading with wget: {e}")
        return False

def download_sequence(dataset_path, sequence, url):
    sequence_folder = os.path.join(dataset_path, sequence)
    if os.path.exists(sequence_folder):
        print(f"Sequence folder already exists: {sequence_folder}. Skipping download ...")
        return sequence_folder

    zip_filename = os.path.basename(url)
    zip_outfile = os.path.join(dataset_path, zip_filename)
    zip_exists = os.path.exists(zip_outfile)
    
    if not zip_exists:
        print(f"Downloading {zip_filename} from {url} using wget...")
        if download_with_wget(url, zip_outfile):
            print("Download successful!\n")
            zip_exists = True
        else:
            print("Download failed.\n")
            return None
    
    if zip_exists:
        try:
            shutil.unpack_archive(zip_outfile, dataset_path)
            os.remove(zip_outfile)
            
            # the extracted folder does not extract to the same name as the zip file
            # instead, it is extracted to the year
            # e.g. 98356.zip -> 2020

        except shutil.Error as e:
            print(f"\nError unzipping {zip_outfile}: {e}\n")
        except OSError as e:
            print(f"\nError renaming or removing files: {e}\n")
    return sequence_folder

def create_rgb_folder(sequence_path, resolution):
    """Create an rgb folder and resize images if resolution is specified."""

    images_path = os.path.join(sequence_path, 'images')
    rgb_path = os.path.join(sequence_path, 'rgb')
    
    if os.path.exists(rgb_path):
        tqdm.write(f"RGB folder already exists: {rgb_path}")
        return
    else:
        os.makedirs(rgb_path, exist_ok=True)
    
    chars = "/-\\|"
    idx = 0

    print(f"Creating rgb folder: {rgb_path} ...", end='', flush=True)
        
    for image_file in os.listdir(images_path):
        src_image = os.path.join(images_path, image_file)
        filename = os.path.join(rgb_path, image_file)
                
        img = Image.open(src_image)
                
        if resolution[0]:
            scaled_height = int(sqrt(img.size[1] / img.size[0] * resolution[0] * resolution[1]))
            scaled_width = int(resolution[0] * resolution[1] / scaled_height)

            resized_img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
            resized_img.save(filename)
        else:
            shutil.copy(src_image, filename)
        
        print(f"\b{chars[idx % len(chars)]}", end='', flush=True)
        idx += 1
        time.sleep(0.1)
    
    print("\b DONE", flush=True)

def write_calibration_yaml(sequence_path, camera_model, fx, fy, k1=0, k2=0, p1=0, p2=0, k3=0, rgb_hz=10):
    """Write the camera calibration parameters to a calibration.yaml file."""
    
    calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
    
    if os.path.exists(calibration_yaml):
        tqdm.write(f"Calibration file already exists: {calibration_yaml}")
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

    tqdm.write(f"Created calibration file: {calibration_yaml}")

def write_rgb_txt(df, sequence_path):
    """Write timestamps and image paths to an rgb.txt file."""
    
    rgb_txt_path = os.path.join(sequence_path, 'rgb.txt')
    
    if os.path.exists(rgb_txt_path):
        tqdm.write(f"RGB file already exists: {rgb_txt_path}")
        return
    
    if df is None:
        tqdm.write(f"    ERROR! Navigation file not found. Skipping rgb.txt ...")
        return

    with open(rgb_txt_path, 'w') as f:
        for _, row in df.iterrows():
            timestamp = datetime.strptime(row['image'][:-4], "%Y%m%dT%H%M%S.%fZ")
            timestamp_seconds = (timestamp - datetime(1970, 1, 1)).total_seconds()
            f.write(f"{timestamp_seconds:.6f} rgb/{row['image']}\n")
            
    tqdm.write(f"Created rgb.txt file: {rgb_txt_path}")

def read_poses_file(poses_file):
    """Reads and parses a COLMAP-style images.txt file."""
    
    if not os.path.exists(poses_file):
        print(f"    ERROR! Images file not found: {poses_file}")
        return None
    
    parsed_data = []
    print(f"Parsing images file: {poses_file}")
    try:
        with open(poses_file, 'r') as f:
            while True:
                line1 = f.readline()
                if not line1:
                    break
                
                # skip comment lines
                if line1.startswith('#'):
                    continue
                
                # two lines of data per image - the first is IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                parts = line1.strip().split()
                if len(parts) != 10: 
                    print(f"WARNING! Skipping unexpected line format: {line1.strip()}")
                    _ = f.readline()  # skip the line afterwards as well, assuming it contains points data
                    continue 

                try:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = int(parts[8])
                    image_name = parts[9]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing data fields in line: {line1.strip()}. \nError: {e}. Skipping.")
                    _ = f.readline() # skip points line
                    continue

                # two lines of data per image - the second is POINTS2D[] as (X, Y, POINT3D_ID)
                line2 = f.readline() 
                if not line2:
                    print(f"WARNING! Found image metadata line without corresponding points line for image: {image_name}")
                    break # unexpected end of file or format error

                # extract timestamp from image name
                try:
                    timestamp_str = image_name.split('.png')[0] # modify this if the extension varies
                    if '.' in timestamp_str:
                        base_ts, frag_ts = timestamp_str.split('.')
                        frag_ts = frag_ts.rstrip('Z')
                        frag_ts = frag_ts[:6] # microseconds
                        timestamp_str = f"{base_ts}.{frag_ts}"
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S.%f")
                    else:
                        timestamp_str = timestamp_str.rstrip('Z')
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
                except ValueError as e:
                    print(f"WARNING: Could not parse timestamp from image name '{image_name}'. \nError: {e}. Skipping image.")
                    continue

                parsed_data.append({
                    'image_id': image_id,
                    'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                    'tx': tx, 'ty': ty, 'tz': tz,
                    'camera_id': camera_id,
                    'image': image_name,
                    'timestamp': timestamp
                })

    except Exception as e:
        print(f"An error occurred while reading {poses_file}: {e}")
        return None
        
    if not parsed_data:
        print(f"No valid data parsed from {poses_file}.")
        return None

    df = pd.DataFrame(parsed_data)
    
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    print(f"Successfully parsed {len(df)} images.")
    return df

def write_groundtruth(df, sequence_path, output_format='csv'):
    """Write groundtruth data to a file.  Supports both CSV and TXT formats."""
    
    output_format = output_format.lower()
    if output_format not in ['csv', 'txt']:
        raise ValueError("Invalid output_format. Must be 'csv' or 'txt'.")

    groundtruth_path = os.path.join(sequence_path, f'groundtruth.{output_format}')

    if os.path.exists(groundtruth_path):
        tqdm.write(f"Groundtruth file already exists: {groundtruth_path}")
        if output_format == 'csv':
            return pd.read_csv(groundtruth_path)
        else:
            return None

    if df is None:
        tqdm.write(f"    ERROR! Poses file not found. Skipping groundtruth.{output_format} ...")
        return None
    
    df['timestamp'] = df['image'].apply(lambda x: datetime.strptime(x[:-4], "%Y%m%dT%H%M%S.%fZ"))
    
    if output_format == 'csv':
        df_output = df[['timestamp', 'image', 'tx', 'ty', 'tz']]
        df_output.to_csv(groundtruth_path, index=False)
        tqdm.write(f"Created groundtruth.csv file: {groundtruth_path}")
        
    elif output_format == 'txt':
        df_output = df[['timestamp', 'tx', 'ty', 'tz']]
        with open(groundtruth_path, 'w') as f:
            for _, row in df_output.iterrows():
                timestamp = row['timestamp']
                timestamp_seconds = (timestamp - datetime(1970, 1, 1)).total_seconds()
                f.write(f"{timestamp_seconds:.6f} {row['tx']} {row['ty']} {row['tz']}\n")
        tqdm.write(f"Created groundtruth.txt file: {groundtruth_path}")



def remove_unused_files(sequence_folder):
    """Removes images, sfm, and navigation files if they exist."""
    
    paths_to_remove = [
        os.path.join(sequence_folder, 'images'),
        os.path.join(sequence_folder, 'sfm'),
        os.path.join(sequence_folder, 'navigation.txt'),
    ]

    for path in paths_to_remove:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

def main():
    
    # load arguments from yaml file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml_file', type=str, default='../arguments/args_dataset_eiffel.yaml',
                        help='yaml file containing arguments')
    arguments = parser.parse_args()
    
    with open (arguments.yaml_file, 'r') as f:
        args = yaml.safe_load(f)

    dataset_path = os.path.abspath(args['dataset_path'])
    resolution = args['resolution']
    
    # ensure that resolution is in the format W,H
    try:
        width, height = map(int, resolution.split(','))
    except ValueError:
        print("Invalid resolution format. Please use the format W,H, e.g., 640,480.")
        print("Using original image resolution...")
        width, height = None, None
    
    os.makedirs(dataset_path, exist_ok=True)
    
    resolution = [width, height]
    
    eiffel_sequences = {
        "2020": "https://www.seanoe.org/data/00810/92226/data/98356.zip",
        "2018": "https://www.seanoe.org/data/00810/92226/data/98314.zip",
        "2016": "https://www.seanoe.org/data/00810/92226/data/98289.zip",
        "2015": "https://www.seanoe.org/data/00810/92226/data/98240.zip",
        }
    
    # download and unarchive the zip file for each sequence
    for sequence, url in eiffel_sequences.items():
        download_sequence(dataset_path, sequence, url)
    
    print("All sequences downloaded successfully!\n")
    
    for sequence in tqdm(eiffel_sequences.keys(), desc=f'Processing sequences'):
        sequence_folder = os.path.join(dataset_path, sequence)
        if not os.path.exists(sequence_folder):
            tqdm.write(f"    ERROR! Sequence folder does not exist: {sequence_folder}")
            exit(0)
        
        # move and resize images to an rgb folder
        create_rgb_folder(sequence_folder, resolution)
        
        # write to files
        poses_file = os.path.join(sequence_folder, 'sfm', 'images.txt')
        df = read_poses_file(poses_file) if os.path.exists(poses_file) else None
        write_calibration_yaml(sequence_folder, "UNKNOWN", 200, 200, rgb_hz=10)
        write_rgb_txt(df, sequence_folder)
        write_groundtruth(df, sequence_folder, output_format='csv')
        write_groundtruth(df, sequence_folder, output_format='txt')
        
        tqdm.write(f"Finished processing sequence: {sequence}\n")
        
        remove_unused_files(sequence_folder)

if __name__ == "__main__":
    main()
