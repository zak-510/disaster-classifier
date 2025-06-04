# Report Generation Test Suite

This directory contains tests for the HTML report generation functionality of the xBD pipeline.

## Quick Start

Run the automated setup and test script:
```bash
python setup_and_test.py
```

This will:
1. Check Python version
2. Install dependencies
3. Run tests with coverage
4. Generate HTML report
5. Open report in browser (optional)

## Manual Setup

### 1. Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- Web browser (for viewing reports)

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually
pip install plotly>=5.13.0
pip install jinja2>=3.1.2
pip install pandas>=1.5.3
pip install opencv-python>=4.5.3
pip install numpy>=1.21.0
pip install torch>=1.10.0
pip install torchvision>=0.11.0
pip install pyyaml>=5.4.1
pip install coverage>=7.0.0
```

### 3. Verify Installation

```bash
# Check Python version
python --version

# Verify package installation
python -c "import plotly, jinja2, pandas, cv2, numpy, torch, torchvision, yaml"
```

## Running Tests

### Automated Testing

```bash
# Run complete test suite with coverage
python setup_and_test.py
```

### Manual Testing

```bash
# Run basic test
python test_report_generation.py

# Run with coverage
coverage run test_report_generation.py
coverage report
coverage html
```

## Test Output

The test suite generates:

1. **HTML Report**
   - Location: `{temp_dir}/output/sample_disaster_report.html`
   - Contains interactive visualizations
   - Includes example images
   - Provides exportable statistics

2. **Coverage Report**
   - Location: `coverage_report/index.html`
   - Shows test coverage statistics
   - Highlights uncovered code

## Verifying Results

### 1. Check Report Generation

```bash
# Verify report exists
ls -l {temp_dir}/output/sample_disaster_report.html

# Check report content
grep "Damage Assessment Report" {temp_dir}/output/sample_disaster_report.html
```

### 2. View Reports

```bash
# Open HTML report in browser
python -c "import webbrowser; webbrowser.open('file://{temp_dir}/output/sample_disaster_report.html')"

# Open coverage report
python -c "import webbrowser; webbrowser.open('file://coverage_report/index.html')"
```

### 3. Verify Visualizations

1. Open the HTML report in a browser
2. Check each tab:
   - Overview: Summary statistics
   - Statistics: Detailed data table
   - Visualizations: Interactive charts
   - Examples: Damage assessment images
3. Test interactive features:
   - Hover over charts
   - Zoom in/out
   - Export data
   - Switch between tabs

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   # Check installed packages
   pip list | grep -E "plotly|jinja2|pandas|opencv|numpy|torch|pyyaml"
   
   # Reinstall if needed
   pip install --force-reinstall -r requirements.txt
   ```

2. **Import Errors**
   ```bash
   # Verify module imports
   python -c "from src.utils.report_generator import HTMLReportGenerator"
   ```

3. **File Permissions**
   ```bash
   # Check temp directory
   python -c "import tempfile; print(tempfile.gettempdir())"
   
   # Ensure write permissions
   chmod -R 755 {temp_dir}
   ```

4. **Browser Issues**
   - Clear browser cache
   - Try different browser
   - Check JavaScript console for errors

### Getting Help

1. Check the coverage report for untested code
2. Review the test output for specific errors
3. Verify all dependencies are correctly installed
4. Ensure proper file permissions
5. Check browser console for JavaScript errors

## Development

### Adding New Tests

1. Create new test file in `tests/` directory
2. Import test functions in `setup_and_test.py`
3. Add test to coverage reporting
4. Update README with new test information

### Modifying Reports

1. Edit `src/utils/report_generator.py`
2. Update test data in `test_report_generation.py`
3. Run tests to verify changes
4. Check coverage report for new code

### Best Practices

1. Always run tests before committing changes
2. Maintain test coverage above 80%
3. Document new features in README
4. Keep test data minimal but representative
5. Verify report generation in multiple browsers 