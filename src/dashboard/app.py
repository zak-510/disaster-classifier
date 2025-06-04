"""Dashboard application."""

import os
from pathlib import Path
from typing import Dict, Any, List
from flask import Flask, render_template, jsonify, request, send_file
import plotly
import plotly.express as px
import pandas as pd
import json
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

def ensure_output_dirs():
    """Ensure output directories exist."""
    output_dir = Path('output')
    for subdir in ['classification', 'localization', 'report']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir

def validate_pagination_params(page, page_size):
    """Validate pagination parameters."""
    try:
        page = int(page)
        page_size = int(page_size)
        if page < 1 or page_size < 1:
            return None, None
        return page, page_size
    except (ValueError, TypeError):
        return None, None

def load_results(results_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    try:
        with open(results_path / 'classification_results.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'results': [], 'total': 0, 'page': 1, 'page_size': 10}

def create_damage_distribution(damage_levels):
    """Create damage distribution chart."""
    if not damage_levels:
        return {'data': [], 'labels': []}
    
    data = []
    labels = []
    for level, stats in damage_levels.items():
        data.append(stats['count'])
        labels.append(['No Damage', 'Minor', 'Major', 'Destroyed'][int(level)])
    
    return {'data': data, 'labels': labels}

def create_confidence_histogram(damage_levels):
    """Create confidence histogram chart."""
    if not damage_levels:
        return {'data': [], 'bins': []}
    
    confidences = []
    for stats in damage_levels.values():
        confidences.extend(stats['confidences'])
    
    bins = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return {'data': confidences, 'bins': bins}

def create_progress_timeline(results):
    """Create progress timeline chart."""
    if not results:
        return {'timestamps': [], 'counts': []}
    
    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timeline = df.groupby('timestamp').size().cumsum()
    
    return {
        'timestamps': timeline.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'counts': timeline.values.tolist()
    }

@app.route('/')
def index():
    """Render dashboard."""
    return render_template('index.html')

@app.route('/api/results')
def get_results():
    """Get paginated results with charts."""
    ensure_output_dirs()
    
    # Validate pagination parameters
    page, page_size = validate_pagination_params(
        request.args.get('page', 1),
        request.args.get('page_size', 10)
    )
    if page is None or page_size is None:
        return jsonify({'error': 'Invalid pagination parameters'}), 400
    
    # Load results
    try:
        with open('output/classification/classification_results.json', 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {'results': [], 'total': 0}
    
    # Calculate pagination
    start = (page - 1) * page_size
    end = start + page_size
    
    # Get page of results
    results = data.get('results', [])
    paginated_results = results[start:end]
    
    # Load statistics for charts
    try:
        with open('output/classification/statistics.json', 'r') as f:
            stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        stats = {'damage_levels': {}}
    
    # Create charts
    charts = {
        'damage_distribution': create_damage_distribution(stats.get('damage_levels', {})),
        'confidence_histogram': create_confidence_histogram(stats.get('damage_levels', {})),
        'timeline': create_progress_timeline(results)
    }
    
    return jsonify({
        'results': paginated_results,
        'total': len(results),
        'page': page,
        'page_size': page_size,
        'charts': charts
    })

@app.route('/api/statistics')
def get_statistics():
    """Get damage assessment statistics."""
    ensure_output_dirs()
    
    try:
        with open('output/classification/statistics.json', 'r') as f:
            stats = json.load(f)
        return jsonify(stats)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({'error': 'Statistics not found'}), 404

@app.route('/api/images/<filename>')
def get_image(filename):
    """Get an image file."""
    # Sanitize filename to prevent path traversal
    filename = secure_filename(filename)
    image_path = Path('output/classification/images') / filename
    
    if not image_path.is_file():
        return jsonify({'error': 'Image not found'}), 404
    
    return send_file(str(image_path))

if __name__ == '__main__':
    app.run(debug=True) 