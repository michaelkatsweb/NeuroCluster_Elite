#!/usr/bin/env python3
"""
File: create_directories.py
Path: NeuroCluster-Elite/create_directories.py
Description: Directory structure creation and initialization script

This script creates all necessary directories and initializes the file structure
for the NeuroCluster Elite trading platform. It ensures proper permissions,
creates initial configuration files, and sets up the database structure.

Features:
- Creates complete directory structure
- Sets appropriate file permissions
- Initializes configuration files
- Creates database directories
- Sets up logging directories
- Creates cache and temporary directories
- Validates directory structure
- Provides detailed logging of operations

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import yaml
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import stat
import shutil
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== DIRECTORY STRUCTURE DEFINITION ====================

DIRECTORY_STRUCTURE = {
    "root": {
        "path": ".",
        "description": "Project root directory",
        "files": ["README.md", "requirements.txt", "setup.py", ".env.example", ".gitignore", "LICENSE"],
        "subdirs": ["src", "config", "data", "logs", "docs", "scripts", "docker", "tests", "examples"]
    },
    
    "src": {
        "path": "src",
        "description": "Source code directory",
        "subdirs": ["core", "data", "trading", "analysis", "interfaces", "integrations", "utils", "tests"]
    },
    
    "src_core": {
        "path": "src/core",
        "description": "Core algorithm components",
        "files": ["__init__.py"],
        "existing_files": ["neurocluster_elite.py", "regime_detector.py", "feature_extractor.py", "pattern_recognition.py"]
    },
    
    "src_data": {
        "path": "src/data",
        "description": "Data management modules",
        "files": ["__init__.py"],
        "existing_files": ["multi_asset_manager.py", "stock_data.py", "crypto_data.py", "forex_data.py", "commodity_data.py", "data_validator.py"]
    },
    
    "src_trading": {
        "path": "src/trading",
        "description": "Trading engine components",
        "files": ["__init__.py"],
        "subdirs": ["strategies"],
        "existing_files": ["trading_engine.py", "strategy_selector.py", "risk_manager.py", "portfolio_manager.py", "order_manager.py"]
    },
    
    "src_trading_strategies": {
        "path": "src/trading/strategies",
        "description": "Trading strategies",
        "files": ["__init__.py"],
        "existing_files": ["base_strategy.py", "bull_strategy.py", "bear_strategy.py", "volatility_strategy.py", "breakout_strategy.py", "range_strategy.py", "crypto_strategies.py"]
    },
    
    "src_analysis": {
        "path": "src/analysis",
        "description": "Analysis and indicators",
        "files": ["__init__.py"],
        "existing_files": ["technical_indicators.py", "sentiment_analyzer.py", "news_processor.py", "social_sentiment.py", "market_scanner.py"]
    },
    
    "src_interfaces": {
        "path": "src/interfaces",
        "description": "User interfaces",
        "files": ["__init__.py"],
        "subdirs": ["components"],
        "existing_files": ["streamlit_dashboard.py", "console_interface.py", "voice_commands.py", "mobile_api.py"]
    },
    
    "src_interfaces_components": {
        "path": "src/interfaces/components",
        "description": "UI components",
        "files": ["__init__.py"],
        "existing_files": ["charts.py", "widgets.py", "layouts.py"]
    },
    
    "src_integrations": {
        "path": "src/integrations",
        "description": "External integrations",
        "files": ["__init__.py"],
        "subdirs": ["brokers", "exchanges", "notifications"]
    },
    
    "src_integrations_brokers": {
        "path": "src/integrations/brokers",
        "description": "Broker integrations",
        "files": ["__init__.py"],
        "existing_files": ["interactive_brokers.py", "td_ameritrade.py", "alpaca.py", "paper_trading.py"]
    },
    
    "src_integrations_exchanges": {
        "path": "src/integrations/exchanges",
        "description": "Crypto exchanges",
        "files": ["__init__.py"],
        "existing_files": ["binance.py", "coinbase.py", "kraken.py"]
    },
    
    "src_integrations_notifications": {
        "path": "src/integrations/notifications",
        "description": "Notification systems",
        "files": ["__init__.py"],
        "existing_files": ["email_alerts.py", "discord_bot.py", "telegram_bot.py", "mobile_push.py", "alert_system.py"]
    },
    
    "src_utils": {
        "path": "src/utils",
        "description": "Utility functions",
        "files": ["__init__.py"],
        "existing_files": ["config_manager.py", "logger.py", "security.py", "database.py", "cache.py", "helpers.py"]
    },
    
    "src_tests": {
        "path": "src/tests",
        "description": "Test suite",
        "files": ["__init__.py"],
        "existing_files": ["test_neurocluster.py", "test_trading_engine.py", "test_data_manager.py", "test_strategies.py", "test_integrations.py"]
    },
    
    "config": {
        "path": "config",
        "description": "Configuration files",
        "existing_files": ["default_config.yaml", "trading_config.yaml", "risk_config.yaml", "api_config.yaml", "alerts_config.yaml"]
    },
    
    "data": {
        "path": "data",
        "description": "Data storage",
        "subdirs": ["cache", "logs", "exports", "backups", "databases"]
    },
    
    "data_cache": {
        "path": "data/cache",
        "description": "Cached market data",
        "subdirs": ["market_data", "technical_indicators", "sentiment_data"]
    },
    
    "data_cache_market_data": {
        "path": "data/cache/market_data",
        "description": "Cached market data by asset type",
        "subdirs": ["stocks", "crypto", "forex", "commodities"]
    },
    
    "data_logs": {
        "path": "data/logs",
        "description": "Application log files",
        "subdirs": ["trading", "analysis", "system", "errors"]
    },
    
    "data_exports": {
        "path": "data/exports",
        "description": "Exported data and reports",
        "subdirs": ["reports", "backups", "csvs"]
    },
    
    "data_backups": {
        "path": "data/backups",
        "description": "Database and configuration backups",
        "subdirs": ["daily", "weekly", "monthly"]
    },
    
    "data_databases": {
        "path": "data/databases",
        "description": "Database files",
        "files": [".gitkeep"]
    },
    
    "logs": {
        "path": "logs",
        "description": "System log files",
        "subdirs": ["application", "trading", "analysis", "errors", "audit"]
    },
    
    "docs": {
        "path": "docs",
        "description": "Documentation",
        "subdirs": ["api", "guides", "screenshots", "research"],
        "existing_files": ["API_REFERENCE.md", "STRATEGY_GUIDE.md", "DEPLOYMENT_GUIDE.md", "VOICE_COMMANDS.md"]
    },
    
    "docs_api": {
        "path": "docs/api",
        "description": "API documentation",
        "files": ["README.md"]
    },
    
    "docs_guides": {
        "path": "docs/guides",
        "description": "User guides",
        "files": ["README.md"]
    },
    
    "docs_screenshots": {
        "path": "docs/screenshots",
        "description": "Dashboard screenshots",
        "files": ["README.md"]
    },
    
    "scripts": {
        "path": "scripts",
        "description": "Utility scripts",
        "files": ["README.md"]
    },
    
    "docker": {
        "path": "docker",
        "description": "Docker configuration",
        "files": ["README.md"]
    },
    
    "tests": {
        "path": "tests",
        "description": "Integration tests",
        "files": ["__init__.py", "README.md"],
        "subdirs": ["unit", "integration", "performance"]
    },
    
    "examples": {
        "path": "examples",
        "description": "Example code and configurations",
        "files": ["README.md"],
        "subdirs": ["strategies", "configurations", "notebooks"]
    }
}

# ==================== HELPER FUNCTIONS ====================

def create_directory(path: Path, description: str = "") -> bool:
    """Create a directory with proper error handling"""
    
    try:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {path} ({description})")
            return True
        else:
            logger.debug(f"📁 Directory already exists: {path}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to create directory {path}: {e}")
        return False

def create_file(file_path: Path, content: str = "", description: str = "") -> bool:
    """Create a file with optional content"""
    
    try:
        if not file_path.exists():
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file with content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Created file: {file_path} ({description})")
            return True
        else:
            logger.debug(f"📄 File already exists: {file_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to create file {file_path}: {e}")
        return False

def check_file_exists(file_path: Path) -> bool:
    """Check if a file exists"""
    return file_path.exists() and file_path.is_file()

def set_permissions(path: Path, is_executable: bool = False) -> bool:
    """Set appropriate permissions for files and directories"""
    
    try:
        if platform.system() == "Windows":
            return True  # Windows doesn't use Unix permissions
        
        if path.is_dir():
            # Directory: rwxr-xr-x (755)
            path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        elif is_executable:
            # Executable file: rwxr-xr-x (755)
            path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        else:
            # Regular file: rw-r--r-- (644)
            path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        
        return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to set permissions for {path}: {e}")
        return False

# ==================== CONTENT TEMPLATES ====================

def get_init_py_content() -> str:
    """Get __init__.py file content"""
    return '''"""
Package initialization file
Generated by NeuroCluster Elite setup script
"""

__version__ = "1.0.0"
'''

def get_readme_content(directory_name: str, description: str) -> str:
    """Get README.md content for directories"""
    return f'''# {directory_name.title()}

{description}

This directory is part of the NeuroCluster Elite trading platform.

## Contents

This directory contains files related to {description.lower()}.

## Generated

This file was automatically generated by the NeuroCluster Elite setup script.
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''

def get_gitkeep_content() -> str:
    """Get .gitkeep file content"""
    return '''# This file ensures the directory is tracked by Git
# Generated by NeuroCluster Elite setup script
'''

# ==================== MAIN FUNCTIONS ====================

def create_directory_structure(base_path: Path = None) -> Dict[str, int]:
    """Create the complete directory structure"""
    
    if base_path is None:
        base_path = Path.cwd()
    
    logger.info(f"🏗️ Creating NeuroCluster Elite directory structure in: {base_path}")
    
    stats = {
        "directories_created": 0,
        "files_created": 0,
        "directories_existed": 0,
        "files_existed": 0,
        "errors": 0
    }
    
    # Process each directory in the structure
    for dir_key, dir_info in DIRECTORY_STRUCTURE.items():
        dir_path = base_path / dir_info["path"]
        description = dir_info.get("description", "")
        
        # Create directory
        if create_directory(dir_path, description):
            stats["directories_created"] += 1
        else:
            if dir_path.exists():
                stats["directories_existed"] += 1
            else:
                stats["errors"] += 1
        
        # Set permissions
        set_permissions(dir_path)
        
        # Create subdirectories
        for subdir in dir_info.get("subdirs", []):
            subdir_path = dir_path / subdir
            if create_directory(subdir_path, f"Subdirectory of {description}"):
                stats["directories_created"] += 1
            else:
                if subdir_path.exists():
                    stats["directories_existed"] += 1
            set_permissions(subdir_path)
        
        # Create required files
        for file_name in dir_info.get("files", []):
            file_path = dir_path / file_name
            content = ""
            
            # Generate appropriate content based on file type
            if file_name == "__init__.py":
                content = get_init_py_content()
            elif file_name == "README.md":
                content = get_readme_content(dir_path.name, description)
            elif file_name == ".gitkeep":
                content = get_gitkeep_content()
            
            if create_file(file_path, content, f"Required file for {description}"):
                stats["files_created"] += 1
            else:
                if file_path.exists():
                    stats["files_existed"] += 1
            
            set_permissions(file_path, file_name.endswith(('.sh', '.py')))
        
        # Check existing files
        for file_name in dir_info.get("existing_files", []):
            file_path = dir_path / file_name
            if check_file_exists(file_path):
                logger.debug(f"✅ Verified existing file: {file_path}")
            else:
                logger.warning(f"⚠️ Missing expected file: {file_path}")
    
    return stats

def create_database_structure(base_path: Path = None) -> bool:
    """Create database directory structure and initialize databases"""
    
    if base_path is None:
        base_path = Path.cwd()
    
    logger.info("🗄️ Setting up database structure...")
    
    try:
        db_dir = base_path / "data" / "databases"
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main application database
        main_db_path = db_dir / "neurocluster_elite.db"
        if not main_db_path.exists():
            conn = sqlite3.connect(main_db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                INSERT INTO metadata (key, value) VALUES 
                ('database_version', '1.0.0'),
                ('created_by', 'NeuroCluster Elite Setup Script')
            """)
            conn.commit()
            conn.close()
            logger.info(f"✅ Created main database: {main_db_path}")
        
        # Create trading database
        trading_db_path = db_dir / "trading.db"
        if not trading_db_path.exists():
            conn = sqlite3.connect(trading_db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
            logger.info(f"✅ Created trading database: {trading_db_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to create database structure: {e}")
        return False

def validate_structure(base_path: Path = None) -> Dict[str, List[str]]:
    """Validate the created directory structure"""
    
    if base_path is None:
        base_path = Path.cwd()
    
    logger.info("🔍 Validating directory structure...")
    
    validation_results = {
        "missing_directories": [],
        "missing_files": [],
        "permission_issues": [],
        "warnings": []
    }
    
    # Check each directory
    for dir_key, dir_info in DIRECTORY_STRUCTURE.items():
        dir_path = base_path / dir_info["path"]
        
        if not dir_path.exists():
            validation_results["missing_directories"].append(str(dir_path))
            continue
        
        # Check required files
        for file_name in dir_info.get("files", []):
            file_path = dir_path / file_name
            if not file_path.exists():
                validation_results["missing_files"].append(str(file_path))
        
        # Check existing files
        for file_name in dir_info.get("existing_files", []):
            file_path = dir_path / file_name
            if not file_path.exists():
                validation_results["warnings"].append(f"Expected file not found: {file_path}")
    
    # Report validation results
    if not any(validation_results.values()):
        logger.info("✅ Directory structure validation passed")
    else:
        for category, issues in validation_results.items():
            if issues:
                logger.warning(f"⚠️ {category.replace('_', ' ').title()}: {len(issues)} issues found")
    
    return validation_results

def cleanup_empty_directories(base_path: Path = None) -> int:
    """Remove empty directories"""
    
    if base_path is None:
        base_path = Path.cwd()
    
    logger.info("🧹 Cleaning up empty directories...")
    
    removed_count = 0
    
    # Walk through directories in reverse order (deepest first)
    for root, dirs, files in os.walk(base_path, topdown=False):
        root_path = Path(root)
        
        # Skip important directories
        if any(important in str(root_path) for important in ['.git', '__pycache__', '.env']):
            continue
        
        # Check if directory is empty
        try:
            if not any(root_path.iterdir()):
                # Don't remove directories that should exist according to structure
                should_exist = any(
                    root_path == base_path / dir_info["path"] 
                    for dir_info in DIRECTORY_STRUCTURE.values()
                )
                
                if not should_exist:
                    root_path.rmdir()
                    logger.debug(f"🗑️ Removed empty directory: {root_path}")
                    removed_count += 1
        except (OSError, PermissionError):
            continue
    
    if removed_count > 0:
        logger.info(f"✅ Removed {removed_count} empty directories")
    
    return removed_count

def print_summary(stats: Dict[str, int], validation_results: Dict[str, List[str]]) -> None:
    """Print setup summary"""
    
    print("\n" + "="*60)
    print("🚀 NEUROCLUSTER ELITE DIRECTORY SETUP COMPLETE")
    print("="*60)
    
    print(f"\n📊 STATISTICS:")
    print(f"   • Directories created: {stats['directories_created']}")
    print(f"   • Files created: {stats['files_created']}")
    print(f"   • Directories existed: {stats['directories_existed']}")
    print(f"   • Files existed: {stats['files_existed']}")
    print(f"   • Errors: {stats['errors']}")
    
    if any(validation_results.values()):
        print(f"\n⚠️ VALIDATION ISSUES:")
        for category, issues in validation_results.items():
            if issues:
                print(f"   • {category.replace('_', ' ').title()}: {len(issues)}")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"     - {issue}")
                if len(issues) > 3:
                    print(f"     ... and {len(issues) - 3} more")
    else:
        print(f"\n✅ VALIDATION: All checks passed")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review the created directory structure")
    print(f"   2. Configure your environment variables (.env file)")
    print(f"   3. Install dependencies: pip install -r requirements.txt")
    print(f"   4. Run the application: streamlit run main_dashboard.py")
    
    print("\n" + "="*60)

def main():
    """Main function to execute directory setup"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroCluster Elite Directory Setup")
    parser.add_argument("--path", type=str, default=".", help="Base path for directory creation")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing structure")
    parser.add_argument("--cleanup", action="store_true", help="Clean up empty directories")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    base_path = Path(args.path).resolve()
    
    print(f"🏗️ NeuroCluster Elite Directory Setup")
    print(f"📍 Working directory: {base_path}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.validate_only:
            # Validate existing structure
            validation_results = validate_structure(base_path)
            
            if not any(validation_results.values()):
                print("✅ Directory structure validation passed")
                return 0
            else:
                print("❌ Directory structure validation failed")
                for category, issues in validation_results.items():
                    if issues:
                        print(f"\n{category.replace('_', ' ').title()}:")
                        for issue in issues:
                            print(f"  - {issue}")
                return 1
        
        else:
            # Create directory structure
            stats = create_directory_structure(base_path)
            
            # Create database structure
            create_database_structure(base_path)
            
            # Validate structure
            validation_results = validate_structure(base_path)
            
            # Cleanup if requested
            if args.cleanup:
                cleanup_empty_directories(base_path)
            
            # Print summary
            print_summary(stats, validation_results)
            
            return 0 if stats["errors"] == 0 else 1
    
    except KeyboardInterrupt:
        logger.info("🛑 Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)