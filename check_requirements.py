#!/usr/bin/env python3
"""
File: check_requirements.py
Path: NeuroCluster-Elite/check_requirements.py
Description: Simple requirements checker for NeuroCluster Elite

A lightweight script that quickly checks if all required packages are installed
and provides simple installation guidance.

Usage:
    python check_requirements.py
    python check_requirements.py --install-missing
    python check_requirements.py --generate-requirements

Author: NeuroCluster Elite Team
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name}")
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple requirements checker for NeuroCluster Elite")
    parser.add_argument('--install-missing', action='store_true', help='Install missing packages')
    parser.add_argument('--generate-requirements', action='store_true', help='Generate requirements.txt')
    args = parser.parse_args()

    print("🔍 NeuroCluster Elite - Simple Requirements Check")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    if not python_ok:
        print("\n❌ Please upgrade to Python 3.8 or higher")
        return False
    
    print("\n📦 Checking Core Packages:")
    
    # Define required packages
    required_packages = [
        ('streamlit', 'streamlit'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('plotly', 'plotly'),
        ('scikit-learn', 'sklearn'),
        ('yfinance', 'yfinance'),
        ('requests', 'requests'),
        ('aiohttp', 'aiohttp'),
        ('python-dotenv', 'dotenv'),
        ('PyYAML', 'yaml'),
        ('ta', 'ta'),
        ('numba', 'numba'),
        ('scipy', 'scipy'),
        ('psutil', 'psutil'),
        ('redis', 'redis'),
        ('sqlalchemy', 'sqlalchemy')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    print(f"\n📊 Results:")
    print(f"  ✅ Installed: {len(required_packages) - len(missing_packages)}")
    print(f"  ❌ Missing: {len(missing_packages)}")
    
    if missing_packages:
        print(f"\n📋 Missing packages: {', '.join(missing_packages)}")
        
        if args.install_missing:
            print("\n🔧 Installing missing packages...")
            for package in missing_packages:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"  ✅ {package} installed")
                else:
                    print(f"  ❌ Failed to install {package}")
        else:
            print(f"\n🚀 To install all missing packages:")
            print(f"pip install {' '.join(missing_packages)}")
            print(f"\nOr run: python check_requirements.py --install-missing")
    else:
        print("\n🎉 All required packages are installed!")
        print("\n🚀 You can now run:")
        print("  python main_console.py")
        print("  streamlit run main_dashboard.py")
    
    if args.generate_requirements:
        print("\n📝 Generating requirements.txt...")
        requirements_content = """# NeuroCluster Elite Requirements
# Core dependencies for the trading platform

# Web Framework
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0

# Data Analysis
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization
plotly>=5.15.0

# Financial Data
yfinance>=0.2.18
ta>=0.10.2

# HTTP Clients
requests>=2.31.0
aiohttp>=3.8.0

# Configuration
python-dotenv>=1.0.0
PyYAML>=6.0.1

# Optional Trading APIs
ccxt>=4.0.0
alpaca-trade-api>=3.0.0

# Optional Analysis Features
textblob>=0.17.1
tweepy>=4.14.0

# Optional Voice Features
SpeechRecognition>=3.10.0
pyttsx3>=2.90

# Optional Cloud Features
boto3>=1.29.0

# Development Tools
pytest>=7.4.0
black>=23.0.0
"""
        
        with open('requirements.txt', 'w') as f:
            f.write(requirements_content)
        print("✅ requirements.txt generated")
    
    return len(missing_packages) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)