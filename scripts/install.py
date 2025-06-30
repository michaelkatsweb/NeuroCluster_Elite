#!/usr/bin/env python3
"""
File: install.py
Path: NeuroCluster-Elite/scripts/install.py
Description: One-click installation script for NeuroCluster Elite Trading Platform

This script provides a comprehensive installation experience for the NeuroCluster Elite
trading platform, handling dependency installation, environment setup, configuration,
and initial validation. It supports multiple installation modes and platforms.

Features:
- Automatic dependency detection and installation
- Virtual environment creation and management
- Platform-specific optimizations (Windows, macOS, Linux)
- Optional components installation (voice, crypto, full stack)
- Configuration file generation
- Database initialization
- Health checks and validation
- Rollback capabilities on failure
- Detailed progress reporting and logging

Installation Modes:
- minimal: Core components only
- standard: Recommended installation
- full: All features including optional components
- development: Development tools and testing frameworks

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import os
import sys
import subprocess
import shutil
import logging
import platform
import venv
import json
import yaml
import tempfile
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== ENUMS AND DATA STRUCTURES ====================

class InstallationMode(Enum):
    """Installation mode options"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"
    DEVELOPMENT = "development"

class PlatformType(Enum):
    """Supported platform types"""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"

@dataclass
class InstallationConfig:
    """Installation configuration"""
    mode: InstallationMode = InstallationMode.STANDARD
    python_version: str = "3.11"
    create_venv: bool = True
    venv_name: str = "neurocluster-env"
    install_dev_tools: bool = False
    install_voice: bool = False
    install_crypto: bool = True
    install_gpu: bool = False
    force_reinstall: bool = False
    skip_validation: bool = False
    backup_existing: bool = True
    
@dataclass
class InstallationProgress:
    """Track installation progress"""
    total_steps: int = 0
    current_step: int = 0
    current_task: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

# ==================== PLATFORM DETECTION ====================

class PlatformDetector:
    """Detect platform and system capabilities"""
    
    @staticmethod
    def get_platform() -> PlatformType:
        """Detect the current platform"""
        system = platform.system().lower()
        if system == "windows":
            return PlatformType.WINDOWS
        elif system == "darwin":
            return PlatformType.MACOS
        elif system == "linux":
            return PlatformType.LINUX
        else:
            return PlatformType.UNKNOWN
    
    @staticmethod
    def get_python_version() -> Tuple[int, int, int]:
        """Get current Python version"""
        return sys.version_info[:3]
    
    @staticmethod
    def has_gpu() -> bool:
        """Check if GPU is available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except ImportError:
            # Try nvidia-smi
            try:
                result = subprocess.run(
                    ["nvidia-smi"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "platform": PlatformDetector.get_platform().value,
            "python_version": ".".join(map(str, PlatformDetector.get_python_version())),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "has_gpu": PlatformDetector.has_gpu(),
            "total_memory": psutil.virtual_memory().total if 'psutil' in sys.modules else "unknown",
            "cpu_count": os.cpu_count()
        }

# ==================== DEPENDENCY MANAGER ====================

class DependencyManager:
    """Manage Python dependencies and system packages"""
    
    def __init__(self, config: InstallationConfig):
        self.config = config
        self.platform = PlatformDetector.get_platform()
        
    def get_base_dependencies(self) -> List[str]:
        """Get base Python dependencies"""
        return [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "yfinance>=0.2.18",
            "requests>=2.31.0",
            "aiohttp>=3.8.0",
            "ta>=0.10.2",
            "python-dotenv>=1.0.0",
            "pyyaml>=6.0.1",
            "sqlalchemy>=2.0.0",
            "bcrypt>=4.0.1",
            "PyJWT>=2.8.0",
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0"
        ]
    
    def get_optional_dependencies(self) -> Dict[str, List[str]]:
        """Get optional dependencies by category"""
        return {
            "voice": [
                "SpeechRecognition>=3.10.0",
                "pyttsx3>=2.90",
                "pyaudio>=0.2.11"
            ],
            "crypto": [
                "ccxt>=4.0.0",
                "python-binance>=1.0.19",
                "coinbase-pro>=0.3.0"
            ],
            "trading": [
                "ib-insync>=0.9.86",
                "alpaca-trade-api>=3.0.0",
                "polygon-api-client>=1.12.0"
            ],
            "analysis": [
                "textblob>=0.17.1",
                "nltk>=3.8.1",
                "tweepy>=4.14.0",
                "newspaper3k>=0.2.8"
            ],
            "gpu": [
                "tensorflow>=2.15.0",
                "torch>=2.1.1",
                "cupy-cuda11x"
            ],
            "development": [
                "pytest>=7.4.0",
                "pytest-asyncio>=0.21.0",
                "pytest-cov>=4.1.0",
                "black>=23.0.0",
                "flake8>=6.0.0",
                "mypy>=1.5.0",
                "jupyter>=1.0.0",
                "ipython>=8.0.0"
            ],
            "monitoring": [
                "prometheus-client>=0.17.0",
                "grafana-api>=1.0.3",
                "psutil>=5.9.0"
            ]
        }
    
    def get_system_dependencies(self) -> Dict[PlatformType, Dict[str, List[str]]]:
        """Get system dependencies by platform"""
        return {
            PlatformType.WINDOWS: {
                "base": [],
                "voice": [],
                "gpu": []
            },
            PlatformType.MACOS: {
                "base": ["portaudio"],
                "voice": ["portaudio"],
                "gpu": []
            },
            PlatformType.LINUX: {
                "base": ["sqlite3", "curl"],
                "voice": ["portaudio19-dev", "libasound2-dev"],
                "gpu": ["nvidia-driver", "cuda-toolkit"]
            }
        }
    
    def install_system_dependencies(self, categories: List[str] = None) -> bool:
        """Install system dependencies"""
        if categories is None:
            categories = ["base"]
        
        system_deps = self.get_system_dependencies()
        platform_deps = system_deps.get(self.platform, {})
        
        all_deps = []
        for category in categories:
            all_deps.extend(platform_deps.get(category, []))
        
        if not all_deps:
            logger.info("📦 No system dependencies required for this platform")
            return True
        
        logger.info(f"📦 Installing system dependencies: {', '.join(all_deps)}")
        
        try:
            if self.platform == PlatformType.LINUX:
                # Try different package managers
                if shutil.which("apt-get"):
                    cmd = ["sudo", "apt-get", "update", "&&", "sudo", "apt-get", "install", "-y"] + all_deps
                    subprocess.run(cmd, check=True, shell=True)
                elif shutil.which("yum"):
                    cmd = ["sudo", "yum", "install", "-y"] + all_deps
                    subprocess.run(cmd, check=True)
                elif shutil.which("dnf"):
                    cmd = ["sudo", "dnf", "install", "-y"] + all_deps
                    subprocess.run(cmd, check=True)
                else:
                    logger.warning("⚠️ No supported package manager found. Please install dependencies manually.")
                    return False
            
            elif self.platform == PlatformType.MACOS:
                if shutil.which("brew"):
                    for dep in all_deps:
                        subprocess.run(["brew", "install", dep], check=True)
                else:
                    logger.warning("⚠️ Homebrew not found. Please install dependencies manually.")
                    return False
            
            elif self.platform == PlatformType.WINDOWS:
                logger.info("📦 System dependencies will be handled by pip on Windows")
                return True
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install system dependencies: {e}")
            return False
    
    def install_python_dependencies(self, mode: InstallationMode) -> bool:
        """Install Python dependencies based on mode"""
        
        dependencies = self.get_base_dependencies()
        optional_deps = self.get_optional_dependencies()
        
        # Add dependencies based on mode
        if mode == InstallationMode.MINIMAL:
            pass  # Only base dependencies
        elif mode == InstallationMode.STANDARD:
            dependencies.extend(optional_deps["crypto"])
            dependencies.extend(optional_deps["analysis"])
        elif mode == InstallationMode.FULL:
            for category, deps in optional_deps.items():
                if category != "development":
                    dependencies.extend(deps)
        elif mode == InstallationMode.DEVELOPMENT:
            for category, deps in optional_deps.items():
                dependencies.extend(deps)
        
        # Add optional components based on config
        if self.config.install_voice:
            dependencies.extend(optional_deps["voice"])
        
        if self.config.install_gpu and PlatformDetector.has_gpu():
            dependencies.extend(optional_deps["gpu"])
        
        logger.info(f"📦 Installing {len(dependencies)} Python packages...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install dependencies in batches to handle conflicts
            batch_size = 10
            for i in range(0, len(dependencies), batch_size):
                batch = dependencies[i:i+batch_size]
                logger.info(f"📦 Installing batch {i//batch_size + 1}: {', '.join(batch)}")
                
                subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    "--upgrade" if self.config.force_reinstall else "--upgrade-strategy", "only-if-needed"
                ] + batch, check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install Python dependencies: {e}")
            return False

# ==================== INSTALLER CLASS ====================

class NeuroClusterInstaller:
    """Main installer class for NeuroCluster Elite"""
    
    def __init__(self, config: InstallationConfig):
        self.config = config
        self.progress = InstallationProgress()
        self.root_path = Path.cwd()
        self.venv_path = None
        self.dependency_manager = DependencyManager(config)
        self.rollback_actions = []
        
    def install(self) -> bool:
        """Main installation method"""
        
        logger.info("🚀 Starting NeuroCluster Elite Installation")
        logger.info("=" * 60)
        
        # Print system information
        system_info = PlatformDetector.get_system_info()
        logger.info(f"🖥️ Platform: {system_info['platform']} ({system_info['architecture']})")
        logger.info(f"🐍 Python: {system_info['python_version']}")
        logger.info(f"💾 CPUs: {system_info['cpu_count']}")
        logger.info(f"🎮 GPU: {'Available' if system_info['has_gpu'] else 'Not available'}")
        logger.info(f"📦 Mode: {self.config.mode.value}")
        
        try:
            # Define installation steps
            steps = [
                ("Validating system requirements", self._validate_system),
                ("Creating virtual environment", self._create_virtual_environment),
                ("Installing system dependencies", self._install_system_dependencies),
                ("Installing Python dependencies", self._install_python_dependencies),
                ("Setting up directory structure", self._setup_directories),
                ("Generating configuration files", self._generate_config_files),
                ("Initializing database", self._initialize_database),
                ("Fixing import paths", self._fix_imports),
                ("Running validation tests", self._run_validation),
                ("Creating shortcuts and scripts", self._create_shortcuts)
            ]
            
            self.progress.total_steps = len(steps)
            
            # Execute installation steps
            for step_name, step_func in steps:
                self.progress.current_step += 1
                self.progress.current_task = step_name
                
                self._log_progress()
                
                try:
                    if not step_func():
                        raise Exception(f"Step failed: {step_name}")
                        
                    logger.info(f"✅ {step_name}")
                    
                except Exception as e:
                    logger.error(f"❌ {step_name}: {e}")
                    self.progress.errors.append(f"{step_name}: {str(e)}")
                    
                    if not self._handle_step_failure(step_name, e):
                        return False
            
            # Installation completed successfully
            self._log_completion()
            return True
            
        except KeyboardInterrupt:
            logger.warning("🛑 Installation interrupted by user")
            self._rollback()
            return False
            
        except Exception as e:
            logger.error(f"❌ Installation failed: {e}")
            self.progress.errors.append(str(e))
            self._rollback()
            return False
    
    def _log_progress(self):
        """Log current progress"""
        percent = (self.progress.current_step / self.progress.total_steps) * 100
        elapsed = time.time() - self.progress.start_time
        
        logger.info(f"📊 Step {self.progress.current_step}/{self.progress.total_steps} ({percent:.1f}%) - {self.progress.current_task}")
        logger.info(f"⏱️ Elapsed: {elapsed:.1f}s")
    
    def _validate_system(self) -> bool:
        """Validate system requirements"""
        
        # Check Python version
        python_version = PlatformDetector.get_python_version()
        if python_version < (3, 8):
            raise Exception(f"Python 3.8+ required, found {'.'.join(map(str, python_version))}")
        
        # Check available disk space
        try:
            free_space = shutil.disk_usage(self.root_path).free
            required_space = 2 * 1024 * 1024 * 1024  # 2 GB
            if free_space < required_space:
                self.progress.warnings.append(f"Low disk space: {free_space / 1024**3:.1f} GB available")
        except:
            pass
        
        # Check internet connectivity
        try:
            urllib.request.urlopen('https://pypi.org', timeout=10)
        except:
            self.progress.warnings.append("Internet connectivity issues detected")
        
        return True
    
    def _create_virtual_environment(self) -> bool:
        """Create Python virtual environment"""
        
        if not self.config.create_venv:
            logger.info("📦 Skipping virtual environment creation")
            return True
        
        self.venv_path = self.root_path / self.config.venv_name
        
        if self.venv_path.exists():
            if self.config.force_reinstall:
                logger.info(f"🗑️ Removing existing virtual environment: {self.venv_path}")
                shutil.rmtree(self.venv_path)
            else:
                logger.info(f"📦 Using existing virtual environment: {self.venv_path}")
                return True
        
        logger.info(f"📦 Creating virtual environment: {self.venv_path}")
        venv.create(self.venv_path, with_pip=True, upgrade_deps=True)
        
        # Activate virtual environment
        self._activate_virtual_environment()
        
        self.rollback_actions.append(("remove_venv", self.venv_path))
        return True
    
    def _activate_virtual_environment(self):
        """Activate the virtual environment"""
        if not self.venv_path:
            return
        
        if PlatformDetector.get_platform() == PlatformType.WINDOWS:
            activate_script = self.venv_path / "Scripts" / "activate"
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            activate_script = self.venv_path / "bin" / "activate"
            python_exe = self.venv_path / "bin" / "python"
        
        # Update PATH and Python executable
        if python_exe.exists():
            sys.executable = str(python_exe)
            os.environ["VIRTUAL_ENV"] = str(self.venv_path)
            os.environ["PATH"] = f"{python_exe.parent}{os.pathsep}{os.environ['PATH']}"
    
    def _install_system_dependencies(self) -> bool:
        """Install system-level dependencies"""
        
        categories = ["base"]
        
        if self.config.install_voice:
            categories.append("voice")
        
        if self.config.install_gpu:
            categories.append("gpu")
        
        return self.dependency_manager.install_system_dependencies(categories)
    
    def _install_python_dependencies(self) -> bool:
        """Install Python dependencies"""
        return self.dependency_manager.install_python_dependencies(self.config.mode)
    
    def _setup_directories(self) -> bool:
        """Set up directory structure"""
        
        try:
            # Run the directory creation script
            subprocess.run([sys.executable, "create_directories.py"], check=True, cwd=self.root_path)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Directory setup failed: {e}")
            return False
    
    def _generate_config_files(self) -> bool:
        """Generate configuration files"""
        
        try:
            # Create .env file if it doesn't exist
            env_file = self.root_path / ".env"
            if not env_file.exists():
                env_content = self._generate_env_content()
                env_file.write_text(env_content)
                logger.info(f"✅ Created .env file: {env_file}")
            
            # Ensure config files exist
            config_dir = self.root_path / "config"
            for config_file in ["default_config.yaml", "trading_config.yaml", "risk_config.yaml"]:
                config_path = config_dir / config_file
                if not config_path.exists():
                    self.progress.warnings.append(f"Missing config file: {config_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration generation failed: {e}")
            return False
    
    def _generate_env_content(self) -> str:
        """Generate .env file content"""
        
        return f"""# NeuroCluster Elite Environment Configuration
# Generated by installer on {time.strftime('%Y-%m-%d %H:%M:%S')}

# Trading Configuration
PAPER_TRADING=true
INITIAL_CAPITAL=100000
DEFAULT_STOCKS=AAPL,GOOGL,MSFT,TSLA

# Risk Management
RISK_LEVEL=moderate
MAX_POSITION_SIZE=0.10
DAILY_LOSS_LIMIT=0.03

# System Configuration
LOG_LEVEL=INFO
DEBUG=false
ENVIRONMENT=production

# Algorithm Settings
ALGORITHM_EFFICIENCY_TARGET=99.59
PROCESSING_TIME_TARGET=0.045

# API Keys (replace with your own)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
POLYGON_API_KEY=your_polygon_key_here
BINANCE_API_KEY=your_binance_key_here
BINANCE_SECRET=your_binance_secret_here

# Database
DATABASE_URL=sqlite:///./data/databases/neurocluster_elite.db

# Optional Features
ENABLE_VOICE_COMMANDS={'true' if self.config.install_voice else 'false'}
ENABLE_GPU_ACCELERATION={'true' if self.config.install_gpu else 'false'}

# Notification Settings (optional)
DISCORD_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
EMAIL_SMTP_SERVER=
EMAIL_USERNAME=
EMAIL_PASSWORD=
"""
    
    def _initialize_database(self) -> bool:
        """Initialize database"""
        
        try:
            # Run database setup script
            subprocess.run([sys.executable, "database_setup.py"], check=True, cwd=self.root_path)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Database initialization failed: {e}")
            return False
        except FileNotFoundError:
            # Create a simple database setup if script doesn't exist
            import sqlite3
            
            db_dir = self.root_path / "data" / "databases"
            db_dir.mkdir(parents=True, exist_ok=True)
            
            db_path = db_dir / "neurocluster_elite.db"
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS installation_info (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                INSERT OR REPLACE INTO installation_info (key, value) VALUES 
                ('installed_version', '1.0.0'),
                ('installation_mode', ?),
                ('platform', ?)
            """, (self.config.mode.value, PlatformDetector.get_platform().value))
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Initialized database: {db_path}")
            return True
    
    def _fix_imports(self) -> bool:
        """Fix import paths and dependencies"""
        
        try:
            # Run import fixer
            subprocess.run([sys.executable, "fix_imports.py", "--fix"], check=True, cwd=self.root_path)
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Import fixing had issues: {e}")
            return True  # Don't fail installation for import issues
        except FileNotFoundError:
            logger.info("Import fixer not found, skipping")
            return True
    
    def _run_validation(self) -> bool:
        """Run validation tests"""
        
        if self.config.skip_validation:
            logger.info("⏭️ Skipping validation tests")
            return True
        
        try:
            # Test basic imports
            test_imports = [
                "import streamlit",
                "import plotly",
                "import pandas",
                "import numpy",
                "import yfinance",
                "from src.core.neurocluster_elite import NeuroClusterElite"
            ]
            
            for test_import in test_imports:
                try:
                    subprocess.run([sys.executable, "-c", test_import], 
                                 check=True, capture_output=True, timeout=30)
                except subprocess.CalledProcessError:
                    self.progress.warnings.append(f"Import test failed: {test_import}")
            
            # Test main components
            try:
                subprocess.run([sys.executable, "-c", "import main_dashboard"], 
                             check=True, capture_output=True, timeout=30, cwd=self.root_path)
            except subprocess.CalledProcessError:
                self.progress.warnings.append("Main dashboard import test failed")
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation tests had issues: {e}")
            return True  # Don't fail installation for validation issues
    
    def _create_shortcuts(self) -> bool:
        """Create shortcuts and launcher scripts"""
        
        try:
            # Create startup script
            startup_script = self.root_path / "start_neurocluster.py"
            startup_content = f"""#!/usr/bin/env python3
\"\"\"
NeuroCluster Elite Startup Script
Generated by installer
\"\"\"

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Starting NeuroCluster Elite...")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    
    try:
        # Start the dashboard
        subprocess.run([
            sys.executable, "main_dashboard.py"
        ], cwd=project_dir)
    except KeyboardInterrupt:
        print("\\n🛑 NeuroCluster Elite stopped")
    except Exception as e:
        print(f"❌ Error starting NeuroCluster Elite: {{e}}")

if __name__ == "__main__":
    main()
"""
            startup_script.write_text(startup_content)
            startup_script.chmod(0o755)
            
            # Create launcher script for different platforms
            if PlatformDetector.get_platform() == PlatformType.WINDOWS:
                batch_script = self.root_path / "start_neurocluster.bat"
                batch_content = f"""@echo off
echo 🚀 Starting NeuroCluster Elite...
cd /d "{self.root_path}"
{sys.executable} main_dashboard.py
pause
"""
                batch_script.write_text(batch_content)
            
            else:
                shell_script = self.root_path / "start_neurocluster.sh"
                shell_content = f"""#!/bin/bash
echo "🚀 Starting NeuroCluster Elite..."
cd "{self.root_path}"
{sys.executable} main_dashboard.py
"""
                shell_script.write_text(shell_content)
                shell_script.chmod(0o755)
            
            logger.info("✅ Created startup scripts")
            return True
            
        except Exception as e:
            logger.warning(f"Shortcut creation failed: {e}")
            return True  # Don't fail installation for this
    
    def _handle_step_failure(self, step_name: str, error: Exception) -> bool:
        """Handle step failure and decide whether to continue"""
        
        # Some steps are optional and shouldn't fail the installation
        optional_steps = [
            "Running validation tests",
            "Creating shortcuts and scripts",
            "Fixing import paths"
        ]
        
        if step_name in optional_steps:
            logger.warning(f"⚠️ Optional step failed, continuing: {step_name}")
            return True
        
        # Critical steps should fail the installation
        logger.error(f"❌ Critical step failed: {step_name}")
        return False
    
    def _rollback(self):
        """Rollback installation changes"""
        
        logger.info("🔄 Rolling back installation changes...")
        
        for action, *args in reversed(self.rollback_actions):
            try:
                if action == "remove_venv":
                    venv_path = args[0]
                    if venv_path.exists():
                        shutil.rmtree(venv_path)
                        logger.info(f"🗑️ Removed virtual environment: {venv_path}")
                
            except Exception as e:
                logger.warning(f"Rollback action failed: {action} - {e}")
    
    def _log_completion(self):
        """Log installation completion"""
        
        elapsed = time.time() - self.progress.start_time
        
        print("\n" + "="*60)
        print("🎉 NEUROCLUSTER ELITE INSTALLATION COMPLETE!")
        print("="*60)
        
        print(f"\n📊 INSTALLATION SUMMARY:")
        print(f"   • Mode: {self.config.mode.value}")
        print(f"   • Platform: {PlatformDetector.get_platform().value}")
        print(f"   • Python: {'.'.join(map(str, PlatformDetector.get_python_version()))}")
        print(f"   • Virtual Environment: {'Yes' if self.config.create_venv else 'No'}")
        print(f"   • Installation Time: {elapsed:.1f} seconds")
        print(f"   • Warnings: {len(self.progress.warnings)}")
        print(f"   • Errors: {len(self.progress.errors)}")
        
        if self.progress.warnings:
            print(f"\n⚠️ WARNINGS:")
            for warning in self.progress.warnings:
                print(f"   • {warning}")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Review the .env file and add your API keys")
        print(f"   2. Start the application:")
        print(f"      streamlit run main_dashboard.py")
        print(f"   3. Open your browser to: http://localhost:8501")
        print(f"   4. Check the documentation: docs/")
        
        print(f"\n📚 QUICK COMMANDS:")
        print(f"   • Dashboard: streamlit run main_dashboard.py")
        print(f"   • API Server: python main_server.py")
        print(f"   • Console: python main_console.py")
        print(f"   • Help: python main_dashboard.py --help")
        
        print("\n" + "="*60)

# ==================== MAIN FUNCTION ====================

def main():
    """Main installation function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroCluster Elite Installation Script")
    parser.add_argument("--mode", choices=[mode.value for mode in InstallationMode], 
                       default="standard", help="Installation mode")
    parser.add_argument("--no-venv", action="store_true", help="Don't create virtual environment")
    parser.add_argument("--venv-name", default="neurocluster-env", help="Virtual environment name")
    parser.add_argument("--install-voice", action="store_true", help="Install voice command components")
    parser.add_argument("--install-gpu", action="store_true", help="Install GPU acceleration components")
    parser.add_argument("--force", action="store_true", help="Force reinstallation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create installation configuration
    config = InstallationConfig(
        mode=InstallationMode(args.mode),
        create_venv=not args.no_venv,
        venv_name=args.venv_name,
        install_voice=args.install_voice,
        install_gpu=args.install_gpu,
        force_reinstall=args.force,
        skip_validation=args.skip_validation
    )
    
    # Auto-detect GPU if available
    if PlatformDetector.has_gpu() and not args.install_gpu:
        config.install_gpu = True
        logger.info("🎮 GPU detected, enabling GPU acceleration")
    
    # Run installation
    installer = NeuroClusterInstaller(config)
    
    try:
        success = installer.install()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("🛑 Installation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Installation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)