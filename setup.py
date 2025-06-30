#!/usr/bin/env python3
"""
File: setup.py
Path: NeuroCluster-Elite/setup.py
Description: Installation script for NeuroCluster Elite Trading Platform

This setup script installs the NeuroCluster Elite trading platform with all
dependencies and optional components.

Usage:
    pip install -e .                    # Development install
    pip install -e ".[full]"            # Full install with all features
    pip install -e ".[trading]"         # Trading-focused install
    pip install -e ".[analysis]"        # Analysis-focused install
    pip install -e ".[voice]"           # Include voice commands
    pip install -e ".[crypto]"          # Include crypto trading
    pip install -e ".[dev]"             # Development tools

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read the requirements file
def read_requirements(filename):
    """Read requirements from file"""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Base requirements
base_requirements = [
    # Core dependencies
    "streamlit>=1.28.0",
    "plotly>=5.15.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    
    # Data sources
    "yfinance>=0.2.18",
    "requests>=2.31.0",
    "aiohttp>=3.8.0",
    
    # Technical analysis
    "ta>=0.10.2",
    
    # Configuration
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
    
    # Database
    "sqlalchemy>=2.0.0",
    
    # Security
    "bcrypt>=4.0.1",
    "PyJWT>=2.8.0",
]

# Optional dependencies
extras_require = {
    # Full installation with all features
    "full": [
        # Voice recognition
        "SpeechRecognition>=3.10.0",
        "pyttsx3>=2.90",
        "pyaudio>=0.2.11",
        
        # Advanced trading APIs
        "ccxt>=4.0.0",
        "ib-insync>=0.9.86",
        "alpaca-trade-api>=3.0.0",
        "polygon-api-client>=1.12.0",
        
        # Sentiment analysis
        "textblob>=0.17.1",
        "nltk>=3.8.1",
        "tweepy>=4.14.0",
        "newspaper3k>=0.2.8",
        
        # Advanced analytics
        "TA-Lib>=0.4.25",
        
        # Notifications
        "discord.py>=2.3.0",
        "python-telegram-bot>=20.5",
        "twilio>=8.5.0",
        
        # Caching
        "redis>=4.6.0",
        
        # Performance
        "numba>=0.57.0",
        "joblib>=1.3.0",
        
        # Additional visualization
        "seaborn>=0.12.0",
        "bokeh>=3.2.0",
    ],
    
    # Trading-focused installation
    "trading": [
        "ccxt>=4.0.0",
        "alpaca-trade-api>=3.0.0",
        "ib-insync>=0.9.86",
        "TA-Lib>=0.4.25",
        "redis>=4.6.0",
    ],
    
    # Analysis-focused installation
    "analysis": [
        "TA-Lib>=0.4.25",
        "textblob>=0.17.1",
        "nltk>=3.8.1",
        "seaborn>=0.12.0",
        "bokeh>=3.2.0",
        "newspaper3k>=0.2.8",
    ],
    
    # Voice commands
    "voice": [
        "SpeechRecognition>=3.10.0",
        "pyttsx3>=2.90",
        "pyaudio>=0.2.11",
    ],
    
    # Cryptocurrency trading
    "crypto": [
        "ccxt>=4.0.0",
        "websocket-client>=1.6.0",
    ],
    
    # Development tools
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0",
    ],
    
    # Documentation
    "docs": [
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.1.0",
        "mkdocstrings>=0.22.0",
    ],
    
    # Production deployment
    "production": [
        "gunicorn>=21.2.0",
        "uvicorn>=0.23.0",
        "redis>=4.6.0",
        "psycopg2-binary>=2.9.0",  # PostgreSQL support
    ],
}

# Platform-specific dependencies
if sys.platform.startswith('win'):
    # Windows-specific
    extras_require["windows"] = ["pywin32>=306"]
elif sys.platform.startswith('darwin'):
    # macOS-specific
    extras_require["macos"] = ["pyobjc>=9.2"]
elif sys.platform.startswith('linux'):
    # Linux-specific
    extras_require["linux"] = []

# Entry points for command-line tools
entry_points = {
    'console_scripts': [
        'neurocluster=src.interfaces.main_console:main',
        'neurocluster-elite=src.interfaces.main_console:main',
        'neurocluster-web=src.interfaces.main_dashboard:main',
        'neurocluster-gui=src.interfaces.main_gui:main',
        'neurocluster-test=src.utils.test_runner:main',
        'neurocluster-setup=src.utils.setup_wizard:main',
    ],
}

# Read version from __init__.py
def get_version():
    """Extract version from package"""
    version_file = this_directory / "src" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.0"

# Setup configuration
setup(
    name="neurocluster-elite",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="Ultimate AI-Powered Multi-Asset Trading Platform with NeuroCluster Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neurocluster-elite",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/neurocluster-elite/issues",
        "Source": "https://github.com/yourusername/neurocluster-elite",
        "Documentation": "https://neurocluster-elite.readthedocs.io/",
        "Changelog": "https://github.com/yourusername/neurocluster-elite/blob/main/CHANGELOG.md",
    },
    
    # Package information
    packages=find_packages(include=['src', 'src.*']),
    include_package_data=True,
    package_data={
        'src': [
            'config/*.yaml',
            'config/*.json',
            'data/*.sql',
            'interfaces/components/*.py',
            'trading/strategies/*.py',
        ],
    },
    
    # Requirements
    python_requires=">=3.8",
    install_requires=base_requirements,
    extras_require=extras_require,
    
    # Entry points
    entry_points=entry_points,
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Web Environment",
        "Environment :: Console",
        "Framework :: AsyncIO",
    ],
    
    # Keywords
    keywords=[
        "trading", "algorithmic-trading", "financial-analysis", "machine-learning",
        "ai", "neurocluster", "market-regime", "portfolio-management",
        "cryptocurrency", "forex", "stocks", "real-time", "streamlit",
        "quantitative-finance", "risk-management", "backtesting"
    ],
    
    # Additional metadata
    license="MIT",
    platforms=["any"],
    zip_safe=False,
)

# Post-installation script
def post_install():
    """Run post-installation setup"""
    
    print("\n🚀 NeuroCluster Elite Installation Complete!")
    print("=" * 50)
    
    # Create necessary directories
    directories = [
        "data",
        "data/cache",
        "data/logs",
        "data/exports",
        "config",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create default configuration files
    config_dir = Path("config")
    
    # Default configuration
    default_config = """# NeuroCluster Elite Configuration
# File: config/default_config.yaml

# Algorithm settings
algorithm:
  similarity_threshold: 0.75
  learning_rate: 0.14
  decay_rate: 0.02
  max_clusters: 12

# Trading settings
trading:
  paper_trading: true
  initial_capital: 100000
  max_position_size: 0.10
  stop_loss_pct: 0.05

# Data sources
data:
  cache:
    enabled: true
    ttl_seconds: 30
  sources:
    yahoo_finance:
      enabled: true
      priority: 1
    coingecko:
      enabled: true
      priority: 2

# Risk management
risk:
  max_portfolio_risk: 0.02
  max_positions: 20
  kelly_fraction_limit: 0.25

# Symbols to track
symbols:
  stocks:
    - AAPL
    - GOOGL
    - MSFT
    - TSLA
    - NVDA
  crypto:
    - BTC-USD
    - ETH-USD
    - ADA-USD
  forex:
    - EURUSD
    - GBPUSD
"""
    
    config_file = config_dir / "default_config.yaml"
    if not config_file.exists():
        with open(config_file, 'w') as f:
            f.write(default_config)
        print(f"✅ Created configuration: {config_file}")
    
    # Environment file template
    env_template = """# NeuroCluster Elite Environment Configuration
# File: .env
# Copy this to .env and fill in your API keys

# Market Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
POLYGON_API_KEY=your_polygon_key_here
FINNHUB_API_KEY=your_finnhub_key_here

# Trading APIs (for live trading)
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
IB_TWS_PORT=7497
TD_AMERITRADE_API_KEY=your_td_key_here

# Cryptocurrency APIs
BINANCE_API_KEY=your_binance_key_here
BINANCE_SECRET_KEY=your_binance_secret_here
COINBASE_API_KEY=your_coinbase_key_here
COINBASE_SECRET_KEY=your_coinbase_secret_here

# Notification Services
DISCORD_WEBHOOK_URL=your_discord_webhook_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
TWILIO_SID=your_twilio_sid_here
TWILIO_TOKEN=your_twilio_token_here

# News & Sentiment APIs
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
NEWS_API_KEY=your_news_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///data/neurocluster.db
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your_jwt_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
WEB_PORT=8501
"""
    
    env_file = Path(".env.example")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_template)
        print(f"✅ Created environment template: {env_file}")
    
    print("\n📋 Next Steps:")
    print("1. Copy .env.example to .env and add your API keys")
    print("2. Run: neurocluster --help")
    print("3. Start with: neurocluster-web")
    print("4. Or try: neurocluster --demo")
    
    print("\n🎯 Quick Start Commands:")
    print("neurocluster-web              # Launch web dashboard")
    print("neurocluster --console        # Console interface")
    print("neurocluster --demo           # Demo mode")
    print("neurocluster-test             # Run tests")
    
    print("\n💡 Documentation:")
    print("https://neurocluster-elite.readthedocs.io/")
    print("\n🎉 Happy Trading!")

if __name__ == "__main__":
    # Run post-installation if called directly
    post_install()