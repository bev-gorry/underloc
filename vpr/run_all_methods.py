
import os
import sys
import argparse
import subprocess


root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root)

from constants import constants

def execute_vpr_methods(yaml_file, vpr_methods):
    """Iterate over the list of vpr_methods, modify the input yaml file to the pipeline, then execute the VPR pipeline for each method."""
    
    print(f'yaml_file: {yaml_file}')
        
    for method in vpr_methods:
        # skip methods that are not run with vpr-methods-evaluation
        if method in ["superpoint-brute-force", "superpoint-hierarchical", "reprojection-error"]:
            continue
        
        print(f"--- Running VPR with method: {method}")

        try:
            with open(yaml_file, 'r') as f:
                lines = f.readlines()

            # modify the vpr_method line
            updated_lines = []
            updated = False
            for line in lines:
                if line.startswith("vpr_method:"):
                    updated_lines.append(f"vpr_method: {method}\n")
                    updated = True
                else:
                    updated_lines.append(line)

            # write the updated content back to the file
            if updated:
                with open(yaml_file, 'w') as f:
                    f.writelines(updated_lines)
            else:
                print(f"Error: Could not find 'vpr_method:' in {yaml_file}")
                continue

            subprocess.run(["pixi", "run", "-e", "vpr", "execute", "--yaml_file", yaml_file], check=True)

            print(f"Finished running with method: {method}")
            print(" ")

        except FileNotFoundError:
            print(f"Error: YAML file not found at {yaml_file}")
            break
        except subprocess.CalledProcessError as e:
            print(f"Error running pixi command: {e}")
            continue


def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml_file', type=str, default='../arguments/args_demo.yaml',
                        help='yaml file containing arguments')
    arguments = parser.parse_args()
    
    yaml_file = arguments.yaml_file
    vpr_methods = constants['vpr_methods']

    execute_vpr_methods(yaml_file, vpr_methods)


if __name__ == '__main__':
    main()