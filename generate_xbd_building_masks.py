import os
import argparse
import json
import numpy as np
import cv2
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

# --- Argument parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Generate building masks from xBD label files')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--labels_dir', type=str, required=True, help='Directory with label files (GeoJSON/WKT)')
    parser.add_argument('--masks_dir', type=str, required=True, help='Directory to save output masks')
    return parser.parse_args()

# --- Main mask generation logic ---
def main():
    args = get_args()
    os.makedirs(args.masks_dir, exist_ok=True)
    image_files = [f for f in os.listdir(args.images_dir) if f.endswith('.png')]
    total_buildings = 0
    for img_file in tqdm(image_files, desc='Generating masks'):
        base = img_file.replace('_pre_disaster.png', '')
        label_file = os.path.join(args.labels_dir, base + '_pre_disaster.json')
        if not os.path.exists(label_file):
            print(f"Label file not found: {label_file}")
            continue
        with open(label_file, 'r') as f:
            data = json.load(f)
        xy_features = data.get('features', {}).get('xy', [])
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        building_count = 0
        for feat in xy_features:
            wkt_str = feat.get('wkt')
            if not wkt_str:
                continue
            try:
                geom = wkt.loads(wkt_str)
            except Exception as e:
                print(f"Failed to parse WKT: {wkt_str} ({e})")
                continue
            if isinstance(geom, Polygon):
                polygons = [geom]
            elif isinstance(geom, MultiPolygon):
                polygons = list(geom.geoms)
            else:
                continue
            for poly in polygons:
                coords = np.array(poly.exterior.coords).round().astype(np.int32)
                cv2.fillPoly(mask, [coords], 255)
                building_count += 1
        if building_count == 0:
            print(f"No valid buildings for {img_file}")
            continue
        mask_path = os.path.join(args.masks_dir, base + '_mask.png')
        cv2.imwrite(mask_path, mask)
        print(f"{img_file}: {building_count} buildings -> {mask_path}")
        total_buildings += building_count
    print(f"Total buildings: {total_buildings}")

if __name__ == '__main__':
    main() 