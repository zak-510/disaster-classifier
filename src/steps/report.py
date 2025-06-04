"""Report generation step."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import pandas as pd
import geopandas as gpd
from jinja2 import Environment, FileSystemLoader
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_report(
    input_path: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Generate HTML report and export data in various formats.
    
    Args:
        input_path: Path to classification results
        output_path: Path to save reports
        config: Optional configuration dictionary
    """
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    try:
        with open(input_path / 'classification_results.json') as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results from {input_path}: {e}")
        raise
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate statistics
    stats = {
        'total_buildings': len(df),
        'damage_levels': df['damage_level'].value_counts().to_dict(),
        'avg_confidence': float(df['confidence'].mean()),
        'min_confidence': float(df['confidence'].min()),
        'max_confidence': float(df['confidence'].max()),
        'generated_at': datetime.now().isoformat()
    }
    
    # Save CSV
    try:
        csv_path = output_path / 'damage_assessment.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV report to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV report: {e}")
        raise
    
    # Generate GeoJSON
    try:
        # Convert to GeoDataFrame using lon/lat columns
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['lon'], df['lat'])
        )
        
        # Save GeoJSON
        geojson_path = output_path / 'damage_assessment.geojson'
        gdf.to_file(geojson_path, driver='GeoJSON')
        logger.info(f"Saved GeoJSON report to {geojson_path}")
    except Exception as e:
        logger.error(f"Failed to save GeoJSON report: {e}")
        raise
    
    # Generate HTML report
    try:
        # Load template using package-relative path
        template_dir = Path(__file__).resolve().parent.parent.parent / 'templates'
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template('report.html')
        
        # Render template
        html_content = template.render(
            stats=stats,
            damage_levels=config.get('damage_levels', {}),
            results=results[:100]  # Show first 100 results in table
        )
        
        # Save HTML
        html_path = output_path / 'damage_assessment.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {html_path}")
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        raise
    
    # Save statistics
    try:
        stats_path = output_path / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")
    except Exception as e:
        logger.error(f"Failed to save statistics: {e}")
        raise 