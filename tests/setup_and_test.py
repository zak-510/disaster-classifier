#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import time
import shutil
import coverage

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_python_version():
    """Check if Python version is 3.6 or higher."""
    import platform
    version = platform.python_version_tuple()
    print(f"\nPython version {version[0]}.{version[1]} detected")
    if int(version[0]) < 3 or (int(version[0]) == 3 and int(version[1]) < 6):
        raise RuntimeError("Python 3.6 or higher is required")

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    requirements = [
        "plotly>=5.13.0",
        "jinja2>=3.1.2",
        "pandas>=1.5.3",
        "opencv-python>=4.5.3",
        "numpy>=1.21.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "pyyaml>=5.4.1",
        "coverage>=7.0.0"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)

def verify_installation():
    """Verify that all dependencies are installed correctly."""
    print("\nVerifying installation...")
    packages = [
        "plotly",
        "jinja2",
        "pandas",
        "cv2",
        "numpy",
        "torch",
        "torchvision",
        "yaml"
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError as e:
            print(f"✗ {package} not installed: {e}")
            sys.exit(1)

def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    print("\nRunning tests with coverage...")
    
    # Initialize coverage
    cov = coverage.Coverage()
    cov.start()
    
    # Run tests
    from test_report_generation import test_report_generation
    report_path = test_report_generation()
    
    # Stop coverage and generate report
    cov.stop()
    cov.save()
    
    # Generate coverage report
    print("\nGenerating coverage report...")
    cov.html_report(directory='coverage_report')
    
    return report_path

def verify_report(report_path):
    """Verify the generated report."""
    print(f"\nVerifying report at {report_path}...")
    
    # Check if report exists
    if not report_path.exists():
        print("Error: Report file not found")
        return False
    
    # Check report content
    with open(report_path) as f:
        content = f.read()
        required_elements = [
            "Damage Assessment Report",
            "Plotly.newPlot",
            "damage-distribution",
            "confidence-distribution",
            "confidence-boxplot"
        ]
        
        for element in required_elements:
            if element not in content:
                print(f"Error: Required element '{element}' not found in report")
                return False
    
    print("✓ Report verification passed")
    return True

def open_report_in_browser(report_path):
    """Open the report in the default web browser."""
    print(f"\nOpening report in browser: {report_path}")
    webbrowser.open(f"file://{report_path.absolute()}")

def main():
    """Main setup and test execution function."""
    print("=== xBD Report Generation Test Suite ===")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Verify installation
    verify_installation()
    
    # Run tests with coverage
    report_path = run_tests_with_coverage()
    
    # Verify report
    if verify_report(report_path):
        # Ask user if they want to open the report
        response = input("\nWould you like to open the report in your browser? (y/n): ")
        if response.lower() == 'y':
            open_report_in_browser(report_path)
    
    print("\n=== Test Suite Complete ===")
    print(f"Coverage report: {Path('coverage_report').absolute()}/index.html")
    print(f"HTML report: {report_path.absolute()}")

if __name__ == '__main__':
    main() 