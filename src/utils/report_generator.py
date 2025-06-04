import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import base64
from jinja2 import Template
import webbrowser

class HTMLReportGenerator:
    def __init__(self, output_dir: Path, disaster_name: str):
        self.output_dir = output_dir
        self.disaster_name = disaster_name
        self.template_dir = Path(__file__).parent / 'templates'
        self.template_dir.mkdir(exist_ok=True)
        
        # Create template if it doesn't exist
        self._create_template()
    
    def _create_template(self):
        """Create the HTML template if it doesn't exist."""
        template_path = self.template_dir / 'report_template.html'
        if not template_path.exists():
            template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Damage Assessment Report - {{ disaster_name }}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.7.1/jszip.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
                    .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }
                    .tab button:hover { background-color: #ddd; }
                    .tab button.active { background-color: #ccc; }
                    .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }
                    .stats-table { border-collapse: collapse; width: 100%; }
                    .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    .stats-table tr:nth-child(even) { background-color: #f2f2f2; }
                    .export-btn { margin: 10px; padding: 8px 16px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
                    .export-btn:hover { background-color: #45a049; }
                    .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
                    .image-container { position: relative; }
                    .image-container img { width: 100%; height: auto; }
                    .image-caption { text-align: center; margin-top: 5px; }
                </style>
            </head>
            <body>
                <h1>Damage Assessment Report - {{ disaster_name }}</h1>
                <p>Generated: {{ timestamp }}</p>

                <div class="tab">
                    <button class="tablinks active" onclick="openTab(event, 'Overview')">Overview</button>
                    <button class="tablinks" onclick="openTab(event, 'Statistics')">Statistics</button>
                    <button class="tablinks" onclick="openTab(event, 'Visualizations')">Visualizations</button>
                    <button class="tablinks" onclick="openTab(event, 'Examples')">Examples</button>
                </div>

                <div id="Overview" class="tabcontent" style="display: block;">
                    <h2>Overview</h2>
                    <div id="overview-stats"></div>
                    <div id="overview-chart"></div>
                </div>

                <div id="Statistics" class="tabcontent">
                    <h2>Detailed Statistics</h2>
                    <div id="stats-table"></div>
                    <button class="export-btn" onclick="exportStats()">Export Statistics (CSV)</button>
                </div>

                <div id="Visualizations" class="tabcontent">
                    <h2>Interactive Visualizations</h2>
                    <div id="damage-distribution"></div>
                    <div id="confidence-distribution"></div>
                    <div id="confidence-boxplot"></div>
                </div>

                <div id="Examples" class="tabcontent">
                    <h2>Example Images</h2>
                    <div class="image-grid" id="example-images"></div>
                </div>

                <script>
                    function openTab(evt, tabName) {
                        var i, tabcontent, tablinks;
                        tabcontent = document.getElementsByClassName("tabcontent");
                        for (i = 0; i < tabcontent.length; i++) {
                            tabcontent[i].style.display = "none";
                        }
                        tablinks = document.getElementsByClassName("tablinks");
                        for (i = 0; i < tablinks.length; i++) {
                            tablinks[i].className = tablinks[i].className.replace(" active", "");
                        }
                        document.getElementById(tabName).style.display = "block";
                        evt.currentTarget.className += " active";
                    }

                    function exportStats() {
                        const stats = {{ stats_json }};
                        const csv = Papa.unparse(stats);
                        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                        saveAs(blob, '{{ disaster_name }}_statistics.csv');
                    }

                    // Plotly charts
                    {{ plotly_js }}
                </script>
            </body>
            </html>
            """
            with open(template_path, 'w') as f:
                f.write(template)
    
    def _create_damage_distribution_chart(self, stats):
        """Create interactive bar chart of damage distribution."""
        damage_names = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
        counts = []
        for i in range(4):
            if i in stats['damage_levels']:
                counts.append(stats['damage_levels'][i]['count'])
            else:
                counts.append(0)
        
        fig = go.Figure(data=[
            go.Bar(
                x=damage_names,
                y=counts,
                text=counts,
                textposition='auto',
                hovertemplate="Damage Level: %{x}<br>Count: %{y}<extra></extra>"
            )
        ])
        
        fig.update_layout(
            title='Building Counts by Damage Level',
            xaxis_title='Damage Level',
            yaxis_title='Count',
            showlegend=False
        )
        
        return fig.to_json()
    
    def _create_confidence_distribution_chart(self, stats):
        """Create interactive histogram of confidence distributions."""
        damage_names = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
        
        fig = go.Figure()
        for level, name in enumerate(damage_names):
            if level in stats['damage_levels'] and 'confidences' in stats['damage_levels'][level]:
                confidences = stats['damage_levels'][level]['confidences']
                fig.add_trace(go.Histogram(
                    x=confidences,
                    name=name,
                    opacity=0.7,
                    hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>"
                ))
        
        fig.update_layout(
            title='Confidence Distribution by Damage Level',
            xaxis_title='Confidence',
            yaxis_title='Count',
            barmode='overlay',
            showlegend=True
        )
        
        return fig.to_json()
    
    def _create_confidence_boxplot(self, stats):
        """Create interactive boxplot of confidence distributions."""
        damage_names = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
        
        fig = go.Figure()
        for level, name in enumerate(damage_names):
            if level in stats['damage_levels'] and 'confidences' in stats['damage_levels'][level]:
                confidences = stats['damage_levels'][level]['confidences']
                fig.add_trace(go.Box(
                    y=confidences,
                    name=name,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    hovertemplate="Damage Level: %{x}<br>Confidence: %{y:.2f}<extra></extra>"
                ))
        
        fig.update_layout(
            title='Confidence Distribution Boxplot',
            yaxis_title='Confidence',
            showlegend=False
        )
        
        return fig.to_json()
    
    def _create_stats_table(self, stats):
        """Create HTML table of detailed statistics."""
        damage_names = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
        
        table_html = """
        <table class="stats-table">
            <tr>
                <th>Damage Level</th>
                <th>Count</th>
                <th>Avg Confidence</th>
                <th>Min Confidence</th>
                <th>Max Confidence</th>
                <th>Std Confidence</th>
                <th>Threshold</th>
            </tr>
        """
        
        for level, name in enumerate(damage_names):
            if level in stats['damage_levels']:
                data = stats['damage_levels'][level]
                threshold = stats.get('thresholds', {}).get(level, 0.0)
                table_html += f"""
                <tr>
                    <td>{name}</td>
                    <td>{data['count']}</td>
                    <td>{data.get('avg_confidence', 0.0):.4f}</td>
                    <td>{data.get('min_confidence', 0.0):.4f}</td>
                    <td>{data.get('max_confidence', 0.0):.4f}</td>
                    <td>{data.get('std_confidence', 0.0):.4f}</td>
                    <td>{threshold:.4f}</td>
                </tr>
                """
        
        table_html += "</table>"
        return table_html
    
    def _create_overview_stats(self, stats):
        """Create HTML for overview statistics."""
        total_detected = stats.get('total_detected', 0)
        total_retained = sum(d.get('count', 0) for d in stats.get('damage_levels', {}).values())
        total_filtered = total_detected - total_retained
        
        return f"""
        <div>
            <h3>Overall Statistics</h3>
            <p>Total buildings detected: {total_detected}</p>
            <p>Buildings retained after filtering: {total_retained}</p>
            <p>Buildings filtered out: {total_filtered}</p>
            <p>Filtering rate: {(total_filtered/total_detected*100 if total_detected > 0 else 0):.1f}%</p>
        </div>
        """
    
    def generate_report(self, stats, example_images=None):
        """Generate the complete HTML report."""
        # Create visualizations
        damage_dist = self._create_damage_distribution_chart(stats)
        conf_dist = self._create_confidence_distribution_chart(stats)
        conf_box = self._create_confidence_boxplot(stats)
        
        # Create statistics table
        stats_table = self._create_stats_table(stats)
        overview_stats = self._create_overview_stats(stats)
        
        # Prepare example images
        example_images_html = ""
        if example_images:
            for img_path in example_images:
                # Convert string path to Path object if needed
                if isinstance(img_path, str):
                    img_path = Path(img_path)
                img_name = img_path.name
                example_images_html += f"""
                <div class="image-container">
                    <img src="{img_path}" alt="{img_name}">
                    <p class="image-caption">{img_name}</p>
                </div>
                """
        
        # Load template
        template_path = self.template_dir / 'report_template.html'
        with open(template_path) as f:
            template = Template(f.read())
        
        # Prepare data for template
        template_data = {
            'disaster_name': self.disaster_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stats_json': json.dumps(stats),
            'plotly_js': f"""
                Plotly.newPlot('damage-distribution', {damage_dist});
                Plotly.newPlot('confidence-distribution', {conf_dist});
                Plotly.newPlot('confidence-boxplot', {conf_box});
            """,
            'overview_stats': overview_stats,
            'stats_table': stats_table,
            'example_images': example_images_html
        }
        
        # Generate report
        report_html = template.render(**template_data)
        report_path = self.output_dir / f"{self.disaster_name}_report.html"
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        return report_path 