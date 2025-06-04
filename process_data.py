import os
import argparse
import json
import pandas as pd
from shapely.geometry import shape, Polygon, MultiPolygon
from tqdm import tqdm

"""
process_data.py
===============
Extracts building polygons and damage labels from SpaceNet-style data and generates train/test CSVs for damage classification.

Usage:
    python process_data.py --input_dir xbd_pipeline_output/spacenet_structure --output_dir xbd_pipeline_output/damage_data [--test_mode]

Arguments:
    --input_dir: Path to SpaceNet-style data directory (with train/val/images/labels).
    --output_dir: Path to save CSVs and extracted data.
    --test_mode: (Optional) If set, runs on a small subset for quick testing.

Example:
    python process_data.py --input_dir xbd_pipeline_output/spacenet_structure --output_dir xbd_pipeline_output/damage_data
"""

def extract_polygons(labels_dir, images_dir, test_mode=False):
    records = []
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('_post_disaster.json')]
    if test_mode:
        label_files = label_files[:2]
    for lbl_file in tqdm(label_files, desc=os.path.basename(labels_dir)):
        lbl_path = os.path.join(labels_dir, lbl_file)
        img_file = lbl_file.replace('_post_disaster.json', '.png')
        img_path = os.path.join(images_dir, img_file)
        if not os.path.exists(img_path):
            continue
        with open(lbl_path) as f:
            data = json.load(f)
        for feat in data.get('features', {}).get('xy', []):
            props = feat.get('properties', {})
            damage = props.get('subtype', 'unknown')
            try:
                poly = shape(feat['geometry'])
                if not isinstance(poly, (Polygon, MultiPolygon)):
                    continue
            except Exception:
                continue
            records.append({
                'image': img_file,
                'polygon_wkt': poly.wkt,
                'damage': damage
            })
    return records

def process_split(split_dir, out_dir, test_mode):
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    records = extract_polygons(labels_dir, images_dir, test_mode)
    return records

def main():
    parser = argparse.ArgumentParser(description="Extract polygons and generate CSVs for damage classification.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to SpaceNet-style data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save CSVs and extracted data')
    parser.add_argument('--test_mode', action='store_true', help='Run on a small subset for quick testing')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'val']:
        split_dir = os.path.join(args.input_dir, split)
        if not os.path.exists(split_dir):
            continue
        print(f"Processing {split} split...")
        records = process_split(split_dir, args.output_dir, args.test_mode)
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(args.output_dir, f'{split}_damage.csv'), index=False)
    print("[Done] Damage CSVs generated.")

if __name__ == "__main__":
    main() 