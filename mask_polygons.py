import os
import argparse
import json
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from PIL import Image
from tqdm import tqdm

"""
mask_polygons.py
================
Generates building mask files from xBD polygon labels, with optional border width.

Usage:
    python mask_polygons.py --input_dir xbd_pipeline_output/disasters --output_dir xbd_pipeline_output/masks [--border 2] [--test_mode]

Arguments:
    --input_dir: Path to the organized disasters directory (from split_into_disasters.py).
    --output_dir: Path to the output directory for mask files.
    --border: (Optional) Border width (in pixels) to add around polygons (default: 2).
    --test_mode: (Optional) If set, only processes a small subset (one disaster).

Example:
    python mask_polygons.py --input_dir xbd_pipeline_output/disasters --output_dir xbd_pipeline_output/masks --border 2

Mask files are saved as PNGs with the same name as the input images.
"""

def polygons_to_mask(polygons, img_shape, border=2):
    from PIL import ImageDraw
    mask = Image.new('L', img_shape, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        coords = [(x, y) for x, y in np.array(poly.exterior.coords)]
        draw.polygon(coords, outline=1, fill=1)
        if border > 0:
            for b in range(1, border+1):
                draw.polygon(coords, outline=1, fill=None)
    return np.array(mask)

def process_disaster(disaster_dir, out_dir, border, test_mode):
    images_dir = os.path.join(disaster_dir, 'images')
    labels_dir = os.path.join(disaster_dir, 'labels')
    mask_dir = os.path.join(out_dir, os.path.basename(disaster_dir), 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    if test_mode:
        image_files = image_files[:2]
    for img_file in tqdm(image_files, desc=f"{os.path.basename(disaster_dir)}"):
        img_path = os.path.join(images_dir, img_file)
        lbl_file = img_file.replace('.png', '.json')
        lbl_path = os.path.join(labels_dir, lbl_file)
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path) as f:
            data = json.load(f)
        polygons = []
        for feat in data.get('features', {}).get('xy', []):
            geom = feat.get('wkt')
            if geom:
                try:
                    poly = shape(feat['geometry'])
                    if isinstance(poly, (Polygon, MultiPolygon)):
                        polygons.append(poly)
                except Exception:
                    continue
        if not polygons:
            continue
        # Assume all images are 1024x1024
        mask = polygons_to_mask(polygons, (1024, 1024), border=border)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(os.path.join(mask_dir, img_file.replace('.png', '_mask.png')))

def main():
    parser = argparse.ArgumentParser(description="Generate building masks from polygons.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to organized disasters directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory for masks')
    parser.add_argument('--border', type=int, default=2, help='Border width in pixels (default: 2)')
    parser.add_argument('--test_mode', action='store_true', help='Process only a small subset (one disaster)')
    args = parser.parse_args()

    output_dirname = os.path.basename(args.output_dir)
    disasters = [d for d in os.listdir(args.input_dir) 
                if os.path.isdir(os.path.join(args.input_dir, d)) and d != output_dirname]
    if args.test_mode:
        disasters = disasters[:1]
    for disaster in disasters:
        disaster_dir = os.path.join(args.input_dir, disaster)
        process_disaster(disaster_dir, args.output_dir, args.border, args.test_mode)
    print("[Done] Masks generated.")

if __name__ == "__main__":
    main() 