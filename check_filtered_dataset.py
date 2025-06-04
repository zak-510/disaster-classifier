import os
import logging
import numpy as np
from train_localization import FilteredXBDDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_building_ratios(dataset):
    """Analyze building ratios in the dataset."""
    building_ratios = []
    
    for i in tqdm(range(len(dataset)), desc="Analyzing samples"):
        sample = dataset[i]
        mask = sample['mask'].numpy().squeeze()
        ratio = np.sum(mask > 0) / mask.size
        building_ratios.append(ratio)
    
    return np.array(building_ratios)

def plot_ratio_distribution(ratios, output_dir):
    """Plot histogram of building ratios."""
    plt.figure(figsize=(10, 6))
    plt.hist(ratios * 100, bins=50, edgecolor='black')
    plt.xlabel('Building Ratio (%)')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Building Ratios')
    plt.savefig(os.path.join(output_dir, 'building_ratio_distribution.png'))
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze filtered dataset statistics')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save analysis results')
    parser.add_argument('--min_building_ratio', type=float, default=0.005, help='Minimum building ratio threshold')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize filtered dataset
    dataset = FilteredXBDDataset(
        data_dir=args.data_dir,
        min_building_ratio=args.min_building_ratio
    )
    
    # Analyze building ratios
    ratios = analyze_building_ratios(dataset)
    
    # Print statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Average building ratio: {np.mean(ratios):.3%}")
    logger.info(f"Median building ratio: {np.median(ratios):.3%}")
    logger.info(f"Std building ratio: {np.std(ratios):.3%}")
    logger.info(f"Min building ratio: {np.min(ratios):.3%}")
    logger.info(f"Max building ratio: {np.max(ratios):.3%}")
    
    # Plot distribution
    plot_ratio_distribution(ratios, args.output_dir)

if __name__ == '__main__':
    main() 