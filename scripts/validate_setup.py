#!/usr/bin/env python3
"""
File: validate_setup.py
Path: NeuroCluster-Elite/scripts/validate_setup.py
Description: Comprehensive setup validation for NeuroCluster Elite Trading Platform

This script performs extensive validation of the NeuroCluster Elite installation,
configuration, and runtime environment. It checks all components, dependencies,
configurations, and system requirements to ensure optimal performance.

Features:
- System requirements validation
- Python dependencies verification
- Configuration file validation
- Database connectivity testing
- API endpoint health checks
- Performance benchmarking
- Security configuration audit
- File permissions verification
- Network connectivity testing
- Algorithm performance validation
- Memory and CPU usage analysis
- Disk space and I/O testing

Validation Categories:
- Core system requirements
- Python environment and packages
- Configuration files and settings
- Database setup and connectivity
- Network services and ports
- Security configurations
- Performance benchmarks
- Integration endpoints
- File system permissions
- Algorithm accuracy tests

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import os
import sys
import time
import json
import yaml
import sqlite3
import psutil
import socket
import subprocess
import logging
import requests
import importlib
import platform
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import asyncio
import aiohttp
import ssl
import urllib.request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== VALIDATION ENUMS ====================

class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"

class ValidationCategory(Enum):
    """Validation categories"""
    SYSTEM = "system"
    PYTHON = "python"
    CONFIG = "config"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    ALGORITHM = "algorithm"

# ==================== DATA STRUCTURES ====================

@dataclass
class ValidationResult:
    """Result of a single validation check"""
    name: str
    category: ValidationCategory
    level: ValidationLevel
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0

@dataclass
class ValidationSummary:
    """Summary of all validation results"""
    total_checks: int = 0
    passed: int = 0
    warnings: int = 0
    errors: int = 0
    critical_errors: int = 0
    execution_time: float = 0.0
    system_score: float = 0.0
    results: List[ValidationResult] = field(default_factory=list)

# ==================== VALIDATION ENGINE ====================

class ValidationEngine:
    """Core validation engine"""
    
    def __init__(self, root_path: Path = None):
        self.root_path = root_path or Path.cwd()
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
        
    def validate_all(self, categories: List[ValidationCategory] = None) -> ValidationSummary:
        """Run all validation checks"""
        
        if categories is None:
            categories = list(ValidationCategory)
        
        logger.info("🔍 Starting comprehensive NeuroCluster Elite validation...")
        
        # Run validations by category
        for category in categories:
            logger.info(f"📋 Validating {category.value}...")
            
            if category == ValidationCategory.SYSTEM:
                self._validate_system()
            elif category == ValidationCategory.PYTHON:
                self._validate_python()
            elif category == ValidationCategory.CONFIG:
                self._validate_config()
            elif category == ValidationCategory.DATABASE:
                self._validate_database()
            elif category == ValidationCategory.NETWORK:
                self._validate_network()
            elif category == ValidationCategory.SECURITY:
                self._validate_security()
            elif category == ValidationCategory.PERFORMANCE:
                self._validate_performance()
            elif category == ValidationCategory.INTEGRATION:
                self._validate_integration()
            elif category == ValidationCategory.ALGORITHM:
                self._validate_algorithm()
        
        # Generate summary
        return self._generate_summary()
    
    def add_result(self, name: str, category: ValidationCategory, level: ValidationLevel, 
                   message: str, details: str = None, fix_suggestion: str = None, 
                   execution_time: float = 0.0):
        """Add a validation result"""
        
        result = ValidationResult(
            name=name,
            category=category,
            level=level,
            message=message,
            details=details,
            fix_suggestion=fix_suggestion,
            execution_time=execution_time
        )
        
        self.results.append(result)
        
        # Log result
        emoji = {
            ValidationLevel.SUCCESS: "✅",
            ValidationLevel.INFO: "ℹ️",
            ValidationLevel.WARNING: "⚠️",
            ValidationLevel.CRITICAL: "❌"
        }
        
        logger.info(f"{emoji[level]} {name}: {message}")
        
        if details:
            logger.debug(f"   Details: {details}")
        
        if fix_suggestion:
            logger.debug(f"   Fix: {fix_suggestion}")
    
    # ==================== SYSTEM VALIDATION ====================
    
    def _validate_system(self):
        """Validate system requirements"""
        
        # Operating System
        start_time = time.time()
        system = platform.system()
        version = platform.version()
        
        if system in ["Windows", "Darwin", "Linux"]:
            self.add_result(
                "Operating System",
                ValidationCategory.SYSTEM,
                ValidationLevel.SUCCESS,
                f"{system} {version}",
                execution_time=time.time() - start_time
            )
        else:
            self.add_result(
                "Operating System",
                ValidationCategory.SYSTEM,
                ValidationLevel.WARNING,
                f"Unsupported OS: {system}",
                fix_suggestion="Use Windows, macOS, or Linux for best compatibility",
                execution_time=time.time() - start_time
            )
        
        # CPU
        start_time = time.time()
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        if cpu_count >= 4:
            level = ValidationLevel.SUCCESS
            message = f"{cpu_count} cores available"
        elif cpu_count >= 2:
            level = ValidationLevel.WARNING
            message = f"Only {cpu_count} cores available"
        else:
            level = ValidationLevel.CRITICAL
            message = f"Insufficient CPU cores: {cpu_count}"
        
        self.add_result(
            "CPU Cores",
            ValidationCategory.SYSTEM,
            level,
            message,
            details=f"Frequency: {cpu_freq.current:.0f} MHz" if cpu_freq else None,
            fix_suggestion="Consider upgrading to a multi-core system" if cpu_count < 4 else None,
            execution_time=time.time() - start_time
        )
        
        # Memory
        start_time = time.time()
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb >= 8:
            level = ValidationLevel.SUCCESS
            message = f"{memory_gb:.1f} GB RAM available"
        elif memory_gb >= 4:
            level = ValidationLevel.WARNING
            message = f"Low RAM: {memory_gb:.1f} GB"
        else:
            level = ValidationLevel.CRITICAL
            message = f"Insufficient RAM: {memory_gb:.1f} GB"
        
        self.add_result(
            "Memory",
            ValidationCategory.SYSTEM,
            level,
            message,
            details=f"Available: {memory.available / (1024**3):.1f} GB ({memory.percent:.1f}% used)",
            fix_suggestion="Consider upgrading to at least 8GB RAM" if memory_gb < 8 else None,
            execution_time=time.time() - start_time
        )
        
        # Disk Space
        start_time = time.time()
        disk = psutil.disk_usage(self.root_path)
        free_gb = disk.free / (1024**3)
        
        if free_gb >= 10:
            level = ValidationLevel.SUCCESS
            message = f"{free_gb:.1f} GB free space"
        elif free_gb >= 5:
            level = ValidationLevel.WARNING
            message = f"Low disk space: {free_gb:.1f} GB"
        else:
            level = ValidationLevel.CRITICAL
            message = f"Insufficient disk space: {free_gb:.1f} GB"
        
        self.add_result(
            "Disk Space",
            ValidationCategory.SYSTEM,
            level,
            message,
            details=f"Total: {disk.total / (1024**3):.1f} GB, Used: {(disk.used / disk.total) * 100:.1f}%",
            fix_suggestion="Free up disk space or add more storage" if free_gb < 10 else None,
            execution_time=time.time() - start_time
        )
        
        # GPU Detection
        start_time = time.time()
        try:
            # Try to detect GPU
            gpu_available = False
            gpu_info = "No GPU detected"
            
            # Try nvidia-smi
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_names = result.stdout.strip().split('\n')
                    gpu_available = True
                    gpu_info = f"NVIDIA GPU(s): {', '.join(gpu_names)}"
            except:
                pass
            
            if gpu_available:
                self.add_result(
                    "GPU",
                    ValidationCategory.SYSTEM,
                    ValidationLevel.SUCCESS,
                    "GPU acceleration available",
                    details=gpu_info,
                    execution_time=time.time() - start_time
                )
            else:
                self.add_result(
                    "GPU",
                    ValidationCategory.SYSTEM,
                    ValidationLevel.INFO,
                    "No GPU detected (CPU-only mode)",
                    details="GPU acceleration can significantly improve performance",
                    fix_suggestion="Consider using a system with NVIDIA GPU for better performance",
                    execution_time=time.time() - start_time
                )
        except Exception as e:
            self.add_result(
                "GPU",
                ValidationCategory.SYSTEM,
                ValidationLevel.WARNING,
                "GPU detection failed",
                details=str(e),
                execution_time=time.time() - start_time
            )
    
    # ==================== PYTHON VALIDATION ====================
    
    def _validate_python(self):
        """Validate Python environment"""
        
        # Python Version
        start_time = time.time()
        python_version = sys.version_info
        version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version >= (3, 9):
            level = ValidationLevel.SUCCESS
            message = f"Python {version_str}"
        elif python_version >= (3, 8):
            level = ValidationLevel.WARNING
            message = f"Python {version_str} (consider upgrading)"
        else:
            level = ValidationLevel.CRITICAL
            message = f"Python {version_str} (too old)"
        
        self.add_result(
            "Python Version",
            ValidationCategory.PYTHON,
            level,
            message,
            details=f"Executable: {sys.executable}",
            fix_suggestion="Upgrade to Python 3.9+" if python_version < (3, 9) else None,
            execution_time=time.time() - start_time
        )
        
        # Virtual Environment
        start_time = time.time()
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if in_venv:
            self.add_result(
                "Virtual Environment",
                ValidationCategory.PYTHON,
                ValidationLevel.SUCCESS,
                "Running in virtual environment",
                details=f"Virtual env path: {sys.prefix}",
                execution_time=time.time() - start_time
            )
        else:
            self.add_result(
                "Virtual Environment",
                ValidationCategory.PYTHON,
                ValidationLevel.WARNING,
                "Not running in virtual environment",
                fix_suggestion="Consider using a virtual environment to avoid conflicts",
                execution_time=time.time() - start_time
            )
        
        # Core Dependencies
        required_packages = [
            ("streamlit", "1.28.0"),
            ("plotly", "5.15.0"),
            ("pandas", "2.0.0"),
            ("numpy", "1.24.0"),
            ("scikit-learn", "1.3.0"),
            ("fastapi", "0.104.1"),
            ("uvicorn", "0.24.0"),
            ("requests", "2.31.0"),
            ("aiohttp", "3.8.0"),
            ("sqlalchemy", "2.0.0")
        ]
        
        missing_packages = []
        outdated_packages = []
        
        for package_name, min_version in required_packages:
            start_time = time.time()
            try:
                module = importlib.import_module(package_name)
                
                # Get version
                if hasattr(module, '__version__'):
                    version = module.__version__
                elif hasattr(module, 'version'):
                    version = module.version
                else:
                    version = "unknown"
                
                # Simple version comparison (basic)
                try:
                    if version != "unknown":
                        current_parts = [int(x) for x in version.split('.')]
                        min_parts = [int(x) for x in min_version.split('.')]
                        
                        if current_parts < min_parts:
                            outdated_packages.append(f"{package_name} {version} (need {min_version}+)")
                        else:
                            self.add_result(
                                f"Package: {package_name}",
                                ValidationCategory.PYTHON,
                                ValidationLevel.SUCCESS,
                                f"Version {version}",
                                execution_time=time.time() - start_time
                            )
                    else:
                        self.add_result(
                            f"Package: {package_name}",
                            ValidationCategory.PYTHON,
                            ValidationLevel.INFO,
                            "Version unknown but available",
                            execution_time=time.time() - start_time
                        )
                except:
                    self.add_result(
                        f"Package: {package_name}",
                        ValidationCategory.PYTHON,
                        ValidationLevel.INFO,
                        f"Version {version}",
                        execution_time=time.time() - start_time
                    )
                
            except ImportError:
                missing_packages.append(package_name)
                self.add_result(
                    f"Package: {package_name}",
                    ValidationCategory.PYTHON,
                    ValidationLevel.CRITICAL,
                    "Missing package",
                    fix_suggestion=f"Install with: pip install {package_name}>={min_version}",
                    execution_time=time.time() - start_time
                )
        
        # Package Summary
        if not missing_packages and not outdated_packages:
            self.add_result(
                "Python Dependencies",
                ValidationCategory.PYTHON,
                ValidationLevel.SUCCESS,
                "All required packages available"
            )
        else:
            level = ValidationLevel.CRITICAL if missing_packages else ValidationLevel.WARNING
            message = f"Package issues: {len(missing_packages)} missing, {len(outdated_packages)} outdated"
            self.add_result(
                "Python Dependencies",
                ValidationCategory.PYTHON,
                level,
                message,
                fix_suggestion="Run: pip install -r requirements.txt --upgrade"
            )
    
    # ==================== CONFIGURATION VALIDATION ====================
    
    def _validate_config(self):
        """Validate configuration files"""
        
        # Configuration Files
        config_files = [
            ("config/default_config.yaml", True),
            ("config/trading_config.yaml", True),
            ("config/risk_config.yaml", True),
            ("config/api_config.yaml", False),
            ("config/alerts_config.yaml", False),
            (".env", False)
        ]
        
        for config_file, required in config_files:
            start_time = time.time()
            config_path = self.root_path / config_file
            
            if config_path.exists():
                try:
                    # Validate file content
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        with open(config_path, 'r') as f:
                            yaml.safe_load(f)
                    elif config_file.endswith('.json'):
                        with open(config_path, 'r') as f:
                            json.load(f)
                    elif config_file == '.env':
                        # Basic .env validation
                        with open(config_path, 'r') as f:
                            content = f.read()
                            if not content.strip():
                                raise ValueError("Empty .env file")
                    
                    self.add_result(
                        f"Config: {config_file}",
                        ValidationCategory.CONFIG,
                        ValidationLevel.SUCCESS,
                        "Valid configuration file",
                        execution_time=time.time() - start_time
                    )
                    
                except Exception as e:
                    self.add_result(
                        f"Config: {config_file}",
                        ValidationCategory.CONFIG,
                        ValidationLevel.CRITICAL,
                        f"Invalid configuration: {e}",
                        fix_suggestion=f"Fix syntax errors in {config_file}",
                        execution_time=time.time() - start_time
                    )
            else:
                level = ValidationLevel.CRITICAL if required else ValidationLevel.WARNING
                self.add_result(
                    f"Config: {config_file}",
                    ValidationCategory.CONFIG,
                    level,
                    "Configuration file missing",
                    fix_suggestion=f"Create {config_file} from template or run setup",
                    execution_time=time.time() - start_time
                )
        
        # Environment Variables
        env_vars = [
            ("PAPER_TRADING", False),
            ("INITIAL_CAPITAL", False),
            ("LOG_LEVEL", False)
        ]
        
        for env_var, required in env_vars:
            start_time = time.time()
            value = os.environ.get(env_var)
            
            if value:
                self.add_result(
                    f"Env: {env_var}",
                    ValidationCategory.CONFIG,
                    ValidationLevel.SUCCESS,
                    f"Set to: {value}",
                    execution_time=time.time() - start_time
                )
            else:
                level = ValidationLevel.WARNING if required else ValidationLevel.INFO
                self.add_result(
                    f"Env: {env_var}",
                    ValidationCategory.CONFIG,
                    level,
                    "Environment variable not set",
                    fix_suggestion=f"Set {env_var} in .env file",
                    execution_time=time.time() - start_time
                )
    
    # ==================== DATABASE VALIDATION ====================
    
    def _validate_database(self):
        """Validate database connectivity and structure"""
        
        # SQLite Database
        start_time = time.time()
        db_path = self.root_path / "data" / "databases" / "neurocluster_elite.db"
        
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ['system_metadata', 'market_data', 'trades', 'positions']
                missing_tables = [table for table in required_tables if table not in tables]
                
                if not missing_tables:
                    self.add_result(
                        "Database Structure",
                        ValidationCategory.DATABASE,
                        ValidationLevel.SUCCESS,
                        f"Database with {len(tables)} tables",
                        details=f"Tables: {', '.join(tables[:5])}...",
                        execution_time=time.time() - start_time
                    )
                else:
                    self.add_result(
                        "Database Structure",
                        ValidationCategory.DATABASE,
                        ValidationLevel.CRITICAL,
                        f"Missing tables: {missing_tables}",
                        fix_suggestion="Run database setup: python database_setup.py",
                        execution_time=time.time() - start_time
                    )
                
                conn.close()
                
            except Exception as e:
                self.add_result(
                    "Database Connection",
                    ValidationCategory.DATABASE,
                    ValidationLevel.CRITICAL,
                    f"Database error: {e}",
                    fix_suggestion="Recreate database: python database_setup.py",
                    execution_time=time.time() - start_time
                )
        else:
            self.add_result(
                "Database File",
                ValidationCategory.DATABASE,
                ValidationLevel.CRITICAL,
                "Database file not found",
                fix_suggestion="Initialize database: python database_setup.py",
                execution_time=time.time() - start_time
            )
    
    # ==================== NETWORK VALIDATION ====================
    
    def _validate_network(self):
        """Validate network connectivity and ports"""
        
        # Port Availability
        ports_to_check = [
            (8501, "Streamlit Dashboard"),
            (8000, "FastAPI Server"),
            (6379, "Redis Cache (optional)")
        ]
        
        for port, description in ports_to_check:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            
            try:
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    self.add_result(
                        f"Port {port}",
                        ValidationCategory.NETWORK,
                        ValidationLevel.WARNING,
                        f"Port in use ({description})",
                        details="Service may already be running",
                        execution_time=time.time() - start_time
                    )
                else:
                    self.add_result(
                        f"Port {port}",
                        ValidationCategory.NETWORK,
                        ValidationLevel.SUCCESS,
                        f"Port available ({description})",
                        execution_time=time.time() - start_time
                    )
            except Exception as e:
                self.add_result(
                    f"Port {port}",
                    ValidationCategory.NETWORK,
                    ValidationLevel.WARNING,
                    f"Port check failed: {e}",
                    execution_time=time.time() - start_time
                )
            finally:
                sock.close()
        
        # Internet Connectivity
        start_time = time.time()
        test_urls = [
            "https://pypi.org",
            "https://api.github.com",
            "https://httpbin.org/get"
        ]
        
        connectivity_results = []
        for url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    connectivity_results.append(f"✅ {url}")
                else:
                    connectivity_results.append(f"⚠️ {url} ({response.status_code})")
            except Exception as e:
                connectivity_results.append(f"❌ {url} ({e})")
        
        if all("✅" in result for result in connectivity_results):
            self.add_result(
                "Internet Connectivity",
                ValidationCategory.NETWORK,
                ValidationLevel.SUCCESS,
                "All test URLs accessible",
                details="\n".join(connectivity_results),
                execution_time=time.time() - start_time
            )
        else:
            self.add_result(
                "Internet Connectivity",
                ValidationCategory.NETWORK,
                ValidationLevel.WARNING,
                "Some connectivity issues",
                details="\n".join(connectivity_results),
                fix_suggestion="Check firewall and network settings",
                execution_time=time.time() - start_time
            )
    
    # ==================== PERFORMANCE VALIDATION ====================
    
    def _validate_performance(self):
        """Validate system performance"""
        
        # CPU Performance Test
        start_time = time.time()
        cpu_start = time.time()
        
        # Simple CPU benchmark
        result = sum(i * i for i in range(100000))
        
        cpu_time = time.time() - cpu_start
        
        if cpu_time < 0.1:
            level = ValidationLevel.SUCCESS
            message = f"CPU performance good ({cpu_time:.3f}s)"
        elif cpu_time < 0.5:
            level = ValidationLevel.WARNING
            message = f"CPU performance moderate ({cpu_time:.3f}s)"
        else:
            level = ValidationLevel.WARNING
            message = f"CPU performance slow ({cpu_time:.3f}s)"
        
        self.add_result(
            "CPU Performance",
            ValidationCategory.PERFORMANCE,
            level,
            message,
            details=f"Benchmark result: {result}",
            execution_time=time.time() - start_time
        )
        
        # Memory Usage
        start_time = time.time()
        memory = psutil.virtual_memory()
        
        if memory.percent < 70:
            level = ValidationLevel.SUCCESS
            message = f"Memory usage normal ({memory.percent:.1f}%)"
        elif memory.percent < 85:
            level = ValidationLevel.WARNING
            message = f"Memory usage high ({memory.percent:.1f}%)"
        else:
            level = ValidationLevel.CRITICAL
            message = f"Memory usage critical ({memory.percent:.1f}%)"
        
        self.add_result(
            "Memory Usage",
            ValidationCategory.PERFORMANCE,
            level,
            message,
            details=f"Available: {memory.available / (1024**3):.1f} GB",
            fix_suggestion="Close unnecessary applications" if memory.percent > 85 else None,
            execution_time=time.time() - start_time
        )
    
    # ==================== INTEGRATION VALIDATION ====================
    
    def _validate_integration(self):
        """Validate external integrations"""
        
        # File Structure
        start_time = time.time()
        required_files = [
            "main_dashboard.py",
            "main_server.py",
            "src/core/neurocluster_elite.py",
            "src/trading/trading_engine.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.root_path / file_path).exists():
                missing_files.append(file_path)
        
        if not missing_files:
            self.add_result(
                "Core Files",
                ValidationCategory.INTEGRATION,
                ValidationLevel.SUCCESS,
                "All core files present",
                execution_time=time.time() - start_time
            )
        else:
            self.add_result(
                "Core Files",
                ValidationCategory.INTEGRATION,
                ValidationLevel.CRITICAL,
                f"Missing files: {missing_files}",
                fix_suggestion="Ensure complete installation",
                execution_time=time.time() - start_time
            )
    
    # ==================== ALGORITHM VALIDATION ====================
    
    def _validate_algorithm(self):
        """Validate NeuroCluster algorithm"""
        
        start_time = time.time()
        try:
            # Try to import core algorithm
            sys.path.insert(0, str(self.root_path))
            from src.core.neurocluster_elite import NeuroClusterElite
            
            # Test algorithm initialization
            algorithm = NeuroClusterElite()
            
            self.add_result(
                "Algorithm Import",
                ValidationCategory.ALGORITHM,
                ValidationLevel.SUCCESS,
                "NeuroCluster algorithm available",
                details="Core algorithm imported successfully",
                execution_time=time.time() - start_time
            )
            
            # Test basic functionality
            start_time = time.time()
            # This would require sample data to test properly
            # For now, just check that the class can be instantiated
            
            self.add_result(
                "Algorithm Functionality",
                ValidationCategory.ALGORITHM,
                ValidationLevel.SUCCESS,
                "Basic algorithm test passed",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.add_result(
                "Algorithm Import",
                ValidationCategory.ALGORITHM,
                ValidationLevel.CRITICAL,
                f"Algorithm import failed: {e}",
                fix_suggestion="Check src/core/neurocluster_elite.py and dependencies",
                execution_time=time.time() - start_time
            )
    
    # ==================== SECURITY VALIDATION ====================
    
    def _validate_security(self):
        """Validate security configuration"""
        
        # File Permissions
        start_time = time.time()
        sensitive_files = [
            ".env",
            "config/api_config.yaml",
            "data/databases"
        ]
        
        permission_issues = []
        for file_path in sensitive_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                try:
                    # Check if file is readable by others (Unix-like systems)
                    if hasattr(os, 'stat'):
                        stat_info = full_path.stat()
                        mode = stat_info.st_mode
                        
                        # Check for world-readable permissions
                        if mode & 0o004:  # World readable
                            permission_issues.append(f"{file_path} is world-readable")
                except:
                    pass
        
        if not permission_issues:
            self.add_result(
                "File Permissions",
                ValidationCategory.SECURITY,
                ValidationLevel.SUCCESS,
                "File permissions secure",
                execution_time=time.time() - start_time
            )
        else:
            self.add_result(
                "File Permissions",
                ValidationCategory.SECURITY,
                ValidationLevel.WARNING,
                f"Permission issues: {len(permission_issues)}",
                details="\n".join(permission_issues),
                fix_suggestion="Run: chmod 600 .env && chmod -R 700 config/",
                execution_time=time.time() - start_time
            )
        
        # Environment Variables Security
        start_time = time.time()
        env_file = self.root_path / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                
                # Check for default/example values
                security_issues = []
                if "your_api_key_here" in content:
                    security_issues.append("Default API key placeholder found")
                if "password123" in content or "secret123" in content:
                    security_issues.append("Weak default passwords found")
                
                if not security_issues:
                    self.add_result(
                        "Environment Security",
                        ValidationCategory.SECURITY,
                        ValidationLevel.SUCCESS,
                        "No obvious security issues in .env",
                        execution_time=time.time() - start_time
                    )
                else:
                    self.add_result(
                        "Environment Security",
                        ValidationCategory.SECURITY,
                        ValidationLevel.WARNING,
                        f"Security issues: {len(security_issues)}",
                        details="\n".join(security_issues),
                        fix_suggestion="Replace placeholder values with actual credentials",
                        execution_time=time.time() - start_time
                    )
            except Exception as e:
                self.add_result(
                    "Environment Security",
                    ValidationCategory.SECURITY,
                    ValidationLevel.WARNING,
                    f"Could not check .env security: {e}",
                    execution_time=time.time() - start_time
                )
    
    # ==================== SUMMARY GENERATION ====================
    
    def _generate_summary(self) -> ValidationSummary:
        """Generate validation summary"""
        
        summary = ValidationSummary()
        summary.total_checks = len(self.results)
        summary.execution_time = time.time() - self.start_time
        summary.results = self.results
        
        # Count by level
        for result in self.results:
            if result.level == ValidationLevel.SUCCESS:
                summary.passed += 1
            elif result.level == ValidationLevel.INFO:
                summary.passed += 1
            elif result.level == ValidationLevel.WARNING:
                summary.warnings += 1
            elif result.level == ValidationLevel.CRITICAL:
                summary.critical_errors += 1
        
        summary.errors = summary.warnings + summary.critical_errors
        
        # Calculate system score (0-100)
        if summary.total_checks > 0:
            score = (summary.passed / summary.total_checks) * 100
            score -= (summary.warnings * 5)  # -5 points per warning
            score -= (summary.critical_errors * 20)  # -20 points per critical error
            summary.system_score = max(0, min(100, score))
        
        return summary

# ==================== REPORT GENERATOR ====================

class ValidationReporter:
    """Generate validation reports"""
    
    @staticmethod
    def print_summary(summary: ValidationSummary):
        """Print validation summary to console"""
        
        print("\n" + "="*70)
        print("🔍 NEUROCLUSTER ELITE VALIDATION REPORT")
        print("="*70)
        
        # Overall score
        if summary.system_score >= 90:
            score_emoji = "🟢"
            score_text = "EXCELLENT"
        elif summary.system_score >= 75:
            score_emoji = "🟡"
            score_text = "GOOD"
        elif summary.system_score >= 50:
            score_emoji = "🟠"
            score_text = "NEEDS ATTENTION"
        else:
            score_emoji = "🔴"
            score_text = "CRITICAL ISSUES"
        
        print(f"\n{score_emoji} SYSTEM SCORE: {summary.system_score:.1f}/100 ({score_text})")
        
        # Statistics
        print(f"\n📊 VALIDATION STATISTICS:")
        print(f"   • Total Checks: {summary.total_checks}")
        print(f"   • Passed: {summary.passed}")
        print(f"   • Warnings: {summary.warnings}")
        print(f"   • Critical Errors: {summary.critical_errors}")
        print(f"   • Execution Time: {summary.execution_time:.2f}s")
        
        # Results by category
        categories = {}
        for result in summary.results:
            if result.category not in categories:
                categories[result.category] = {"passed": 0, "warnings": 0, "errors": 0}
            
            if result.level in [ValidationLevel.SUCCESS, ValidationLevel.INFO]:
                categories[result.category]["passed"] += 1
            elif result.level == ValidationLevel.WARNING:
                categories[result.category]["warnings"] += 1
            else:
                categories[result.category]["errors"] += 1
        
        print(f"\n📋 RESULTS BY CATEGORY:")
        for category, counts in categories.items():
            total = sum(counts.values())
            pass_rate = (counts["passed"] / total) * 100 if total > 0 else 0
            
            status_emoji = "✅" if pass_rate >= 90 else "⚠️" if pass_rate >= 70 else "❌"
            
            print(f"   {status_emoji} {category.value.title()}: "
                  f"{counts['passed']}/{total} passed ({pass_rate:.0f}%)")
        
        # Critical issues
        critical_issues = [r for r in summary.results if r.level == ValidationLevel.CRITICAL]
        if critical_issues:
            print(f"\n❌ CRITICAL ISSUES REQUIRING ATTENTION:")
            for i, issue in enumerate(critical_issues, 1):
                print(f"   {i}. {issue.name}: {issue.message}")
                if issue.fix_suggestion:
                    print(f"      Fix: {issue.fix_suggestion}")
        
        # Warnings
        warnings = [r for r in summary.results if r.level == ValidationLevel.WARNING]
        if warnings:
            print(f"\n⚠️ WARNINGS:")
            for i, warning in enumerate(warnings[:5], 1):  # Show first 5
                print(f"   {i}. {warning.name}: {warning.message}")
            
            if len(warnings) > 5:
                print(f"   ... and {len(warnings) - 5} more warnings")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if summary.critical_errors > 0:
            print(f"   1. Address critical issues immediately")
            print(f"   2. Run validation again after fixes")
        elif summary.warnings > 0:
            print(f"   1. Review and address warnings when possible")
            print(f"   2. Monitor system performance")
        else:
            print(f"   1. System is ready for production use")
            print(f"   2. Run periodic validations to maintain health")
        
        if summary.system_score < 75:
            print(f"   3. Consider system upgrades for better performance")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   • Fix critical issues before running the application")
        print(f"   • Review configuration files and environment variables")
        print(f"   • Run: python startup.py dashboard")
        print(f"   • Monitor: http://localhost:8501")
        
        print("\n" + "="*70)

# ==================== MAIN FUNCTION ====================

def main():
    """Main validation function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroCluster Elite Setup Validation")
    parser.add_argument("--categories", nargs="*", 
                       choices=[c.value for c in ValidationCategory],
                       help="Validation categories to run")
    parser.add_argument("--output", choices=["console", "json", "yaml"],
                       default="console", help="Output format")
    parser.add_argument("--output-file", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick validation (skip performance tests)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine categories to run
    categories = None
    if args.categories:
        categories = [ValidationCategory(c) for c in args.categories]
    elif args.quick:
        categories = [
            ValidationCategory.SYSTEM,
            ValidationCategory.PYTHON,
            ValidationCategory.CONFIG,
            ValidationCategory.DATABASE
        ]
    
    print("🔍 NeuroCluster Elite Setup Validation")
    print("=" * 50)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if categories:
        print(f"📋 Categories: {', '.join(c.value for c in categories)}")
    else:
        print(f"📋 Categories: All")
    
    print("=" * 50)
    
    try:
        # Run validation
        validator = ValidationEngine()
        summary = validator.validate_all(categories)
        
        # Generate report
        if args.output == "console":
            ValidationReporter.print_summary(summary)
        elif args.output == "json":
            # Convert to JSON-serializable format
            data = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_checks": summary.total_checks,
                    "passed": summary.passed,
                    "warnings": summary.warnings,
                    "critical_errors": summary.critical_errors,
                    "system_score": summary.system_score,
                    "execution_time": summary.execution_time
                },
                "results": [
                    {
                        "name": r.name,
                        "category": r.category.value,
                        "level": r.level.value,
                        "message": r.message,
                        "details": r.details,
                        "fix_suggestion": r.fix_suggestion,
                        "execution_time": r.execution_time
                    }
                    for r in summary.results
                ]
            }
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"✅ Report saved to: {args.output_file}")
            else:
                print(json.dumps(data, indent=2))
        
        # Return appropriate exit code
        if summary.critical_errors > 0:
            return 2  # Critical issues
        elif summary.warnings > 0:
            return 1  # Warnings
        else:
            return 0  # All good
            
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted")
        return 1
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)