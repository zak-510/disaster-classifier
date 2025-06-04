import os
import shutil
import argparse
from glob import glob

"""
split_into_disasters.py
======================
Organizes raw xBD data into a directory structure by disaster, with images/ and labels/ subfolders.

Usage:
    python split_into_disasters.py --input_dir Data --output_dir xbd_pipeline_output/disasters [--test_mode]

Arguments:
    --input_dir: Path to the raw xBD data directory (containing train/test/images/labels).
    --output_dir: Path to the output directory for organized disasters.
    --test_mode: (Optional) If set, only processes a small subset (one disaster).

Example:
    python split_into_disasters.py --input_dir Data --output_dir xbd_pipeline_output/disasters

This script does NOT modify the input directory.
"""

def get_disaster_name(filename):
    # Disaster name is the prefix before the first underscore
    return filename.split('_')[0]

def organize_split(split_dir, out_dir, test_mode=False):
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    image_files = glob(os.path.join(images_dir, '*.png'))
    disasters = {}
    for img_path in image_files:
        fname = os.path.basename(img_path)
        disaster = fname.split('_')[0]
        if disaster not in disasters:
            disasters[disaster] = []
        disasters[disaster].append(fname)
    # If test_mode, only process one disaster
    if test_mode:
        disasters = {k: v for k, v in list(disasters.items())[:1]}
    for disaster, files in disasters.items():
        d_img_dir = os.path.join(out_dir, disaster, 'images')
        d_lbl_dir = os.path.join(out_dir, disaster, 'labels')
        os.makedirs(d_img_dir, exist_ok=True)
        os.makedirs(d_lbl_dir, exist_ok=True)
        for fname in files:
            shutil.copy2(os.path.join(images_dir, fname), os.path.join(d_img_dir, fname))
            # Copy both pre and post label files if they exist
            for phase in ['pre_disaster', 'post_disaster']:
                lbl_name = fname.replace('.png', f'_{phase}.json').replace(f'_{phase}.png', f'_{phase}.json')
                lbl_path = os.path.join(labels_dir, lbl_name)
                if os.path.exists(lbl_path):
                    shutil.copy2(lbl_path, os.path.join(d_lbl_dir, lbl_name))

def main():
    parser = argparse.ArgumentParser(description="Organize raw xBD data by disaster.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to raw xBD data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--test_mode', action='store_true', help='Process only a small subset (one disaster)')
    args = parser.parse_args()

    for split in ['train', 'test']:
        split_dir = os.path.join(args.input_dir, split)
        if not os.path.exists(split_dir):
            print(f"[Warning] Split directory not found: {split_dir}")
            continue
        out_dir = os.path.join(args.output_dir, split)
        print(f"Organizing {split} split...")
        organize_split(split_dir, out_dir, test_mode=args.test_mode)
    print("[Done] Data organized by disaster.")

if __name__ == "__main__":
    main() 