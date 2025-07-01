#!/usr/bin/env python3
"""
File: startup.py
Path: NeuroCluster-Elite/startup.py
Description: Universal application launcher for NeuroCluster Elite Trading Platform

This script serves as the central entry point for launching all components of the
NeuroCluster Elite trading platform. It provides a unified interface to start
the dashboard, API server, console interface, or all services together with
proper orchestration and monitoring.

Features:
- Unified launcher for all application components
- Service dependency management and startup ordering
- Health monitoring and automatic restart capabilities
- Environment validation and setup
- Configuration validation and auto-correction
- Log aggregation and monitoring
- Graceful shutdown handling
- Multi-platform support (Windows, macOS, Linux)
- Development and production mode support
- Docker integration and container management

Launch Modes:
- dashboard: Streamlit web dashboard only
- api: FastAPI server only
- console: Command-line interface
- full: All services (dashboard + api + monitoring)
- docker: Docker container orchestration

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import os
import sys
import time
import signal
import subprocess
import threading
import queue
import logging
import psutil
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import argparse
import asyncio
import aiohttp
import webbrowser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== ENUMS AND DATA STRUCTURES ====================

class LaunchMode(Enum):
    """Application launch modes"""
    DASHBOARD = "dashboard"
    API = "api"
    CONSOLE = "console"
    FULL = "full"
    DOCKER = "docker"
    TEST = "test"

class ServiceStatus(Enum):
    """Service status enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    command: List[str]
    port: Optional[int] = None
    health_url: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    required: bool = True
    startup_timeout: int = 60
    restart_policy: str = "always"  # always, on-failure, unless-stopped
    max_restarts: int = 3
    environment: Dict[str, str] = field(default_factory=dict)

@dataclass
class ServiceState:
    """Runtime state of a service"""
    config: ServiceConfig
    status: ServiceStatus = ServiceStatus.STOPPED
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    restart_count: int = 0
    last_health_check: Optional[datetime] = None
    health_status: bool = False
    output_queue: queue.Queue = field(default_factory=queue.Queue)

# ==================== SERVICE MANAGER ====================

class ServiceManager:
    """Manages application services and their lifecycle"""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.services: Dict[str, ServiceState] = {}
        self.shutdown_event = threading.Event()
        self.monitor_thread = None
        self.log_monitor_thread = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize service configurations
        self._initialize_service_configs()
    
    def _initialize_service_configs(self):
        """Initialize service configurations"""
        
        # Streamlit Dashboard Service
        dashboard_config = ServiceConfig(
            name="dashboard",
            command=[
                sys.executable, "-m", "streamlit", "run", "main_dashboard.py",
                "--server.address", "0.0.0.0",
                "--server.port", "8501",
                "--server.headless", "true",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
            ],
            port=8501,
            health_url="http://localhost:8501/health",
            required=True,
            startup_timeout=90
        )
        
        # FastAPI Server Service
        api_config = ServiceConfig(
            name="api",
            command=[
                sys.executable, "main_server.py",
                "--host", "0.0.0.0",
                "--port", "8000"
            ],
            port=8000,
            health_url="http://localhost:8000/health",
            required=True,
            startup_timeout=60
        )
        
        # Console Interface Service
        console_config = ServiceConfig(
            name="console",
            command=[sys.executable, "main_console.py", "--daemon"],
            required=False,
            startup_timeout=30
        )
        
        # Redis Cache Service (if available)
        redis_config = ServiceConfig(
            name="redis",
            command=["redis-server", "--port", "6379"],
            port=6379,
            required=False,
            startup_timeout=30
        )
        
        # Initialize service states
        for config in [dashboard_config, api_config, console_config, redis_config]:
            self.services[config.name] = ServiceState(config=config)
    
    def start_service(self, service_name: str) -> bool:
        """Start a specific service"""
        
        if service_name not in self.services:
            logger.error(f"❌ Unknown service: {service_name}")
            return False
        
        service = self.services[service_name]
        
        if service.status == ServiceStatus.RUNNING:
            logger.info(f"✅ Service {service_name} is already running")
            return True
        
        logger.info(f"🚀 Starting service: {service_name}")
        service.status = ServiceStatus.STARTING
        
        try:
            # Check dependencies
            for dependency in service.config.depends_on:
                if not self._is_service_healthy(dependency):
                    logger.error(f"❌ Dependency {dependency} not available for {service_name}")
                    service.status = ServiceStatus.ERROR
                    return False
            
            # Start the process
            env = os.environ.copy()
            env.update(service.config.environment)
            
            service.process = subprocess.Popen(
                service.config.command,
                cwd=self.root_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            service.pid = service.process.pid
            service.start_time = datetime.now()
            service.status = ServiceStatus.RUNNING
            
            # Start log monitoring for this service
            self._start_log_monitoring(service)
            
            # Wait for service to be healthy
            if service.config.health_url:
                if self._wait_for_health(service):
                    logger.info(f"✅ Service {service_name} started successfully (PID: {service.pid})")
                    return True
                else:
                    logger.error(f"❌ Service {service_name} failed health check")
                    self.stop_service(service_name)
                    return False
            else:
                logger.info(f"✅ Service {service_name} started (PID: {service.pid})")
                return True
        
        except Exception as e:
            logger.error(f"❌ Failed to start service {service_name}: {e}")
            service.status = ServiceStatus.ERROR
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service"""
        
        if service_name not in self.services:
            logger.error(f"❌ Unknown service: {service_name}")
            return False
        
        service = self.services[service_name]
        
        if service.status == ServiceStatus.STOPPED:
            logger.info(f"✅ Service {service_name} is already stopped")
            return True
        
        logger.info(f"🛑 Stopping service: {service_name}")
        service.status = ServiceStatus.STOPPING
        
        try:
            if service.process:
                # Try graceful shutdown first
                service.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    service.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    logger.warning(f"⚠️ Force killing service {service_name}")
                    service.process.kill()
                    service.process.wait()
                
                service.process = None
                service.pid = None
            
            service.status = ServiceStatus.STOPPED
            service.health_status = False
            
            logger.info(f"✅ Service {service_name} stopped")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to stop service {service_name}: {e}")
            service.status = ServiceStatus.ERROR
            return False
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a specific service"""
        
        logger.info(f"🔄 Restarting service: {service_name}")
        
        if not self.stop_service(service_name):
            return False
        
        # Wait a moment before restarting
        time.sleep(2)
        
        return self.start_service(service_name)
    
    def start_mode(self, mode: LaunchMode) -> bool:
        """Start services based on launch mode"""
        
        logger.info(f"🚀 Starting NeuroCluster Elite in {mode.value} mode")
        
        services_to_start = []
        
        if mode == LaunchMode.DASHBOARD:
            services_to_start = ["dashboard"]
        elif mode == LaunchMode.API:
            services_to_start = ["api"]
        elif mode == LaunchMode.CONSOLE:
            services_to_start = ["console"]
        elif mode == LaunchMode.FULL:
            services_to_start = ["api", "dashboard"]
        elif mode == LaunchMode.DOCKER:
            return self._start_docker_services()
        elif mode == LaunchMode.TEST:
            return self._run_tests()
        
        # Start services in dependency order
        success = True
        for service_name in services_to_start:
            if not self.start_service(service_name):
                success = False
                if self.services[service_name].config.required:
                    logger.error(f"❌ Required service {service_name} failed to start")
                    return False
        
        if success:
            logger.info("✅ All services started successfully")
            
            # Open browser for dashboard mode
            if mode in [LaunchMode.DASHBOARD, LaunchMode.FULL]:
                self._open_browser()
            
            # Start monitoring
            self._start_monitoring()
        
        return success
    
    def stop_all(self) -> bool:
        """Stop all running services"""
        
        logger.info("🛑 Stopping all services...")
        
        # Stop services in reverse dependency order
        service_names = list(self.services.keys())
        service_names.reverse()
        
        success = True
        for service_name in service_names:
            if not self.stop_service(service_name):
                success = False
        
        # Stop monitoring
        self.shutdown_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        if self.log_monitor_thread and self.log_monitor_thread.is_alive():
            self.log_monitor_thread.join(timeout=5)
        
        return success
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "system": self._get_system_info()
        }
        
        for name, service in self.services.items():
            status["services"][name] = {
                "status": service.status.value,
                "pid": service.pid,
                "start_time": service.start_time.isoformat() if service.start_time else None,
                "restart_count": service.restart_count,
                "health_status": service.health_status,
                "port": service.config.port
            }
        
        return status
    
    def _wait_for_health(self, service: ServiceState) -> bool:
        """Wait for service to become healthy"""
        
        if not service.config.health_url:
            return True
        
        logger.info(f"🔍 Waiting for {service.config.name} health check...")
        
        start_time = time.time()
        while time.time() - start_time < service.config.startup_timeout:
            try:
                import requests
                response = requests.get(service.config.health_url, timeout=5)
                if response.status_code == 200:
                    service.health_status = True
                    service.last_health_check = datetime.now()
                    return True
            except:
                pass
            
            time.sleep(2)
        
        return False
    
    def _is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        
        if service.status != ServiceStatus.RUNNING:
            return False
        
        if not service.config.health_url:
            return service.process and service.process.poll() is None
        
        try:
            import requests
            response = requests.get(service.config.health_url, timeout=5)
            healthy = response.status_code == 200
            service.health_status = healthy
            service.last_health_check = datetime.now()
            return healthy
        except:
            service.health_status = False
            return False
    
    def _start_monitoring(self):
        """Start service monitoring thread"""
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.monitor_thread = threading.Thread(target=self._monitor_services, daemon=True)
        self.monitor_thread.start()
        
        logger.info("📊 Started service monitoring")
    
    def _monitor_services(self):
        """Monitor services and restart if needed"""
        
        while not self.shutdown_event.is_set():
            try:
                for name, service in self.services.items():
                    if service.status == ServiceStatus.RUNNING:
                        # Check if process is still alive
                        if service.process and service.process.poll() is not None:
                            logger.warning(f"⚠️ Service {name} died (exit code: {service.process.returncode})")
                            service.status = ServiceStatus.ERROR
                            
                            # Restart if policy allows
                            if (service.config.restart_policy == "always" and 
                                service.restart_count < service.config.max_restarts):
                                
                                logger.info(f"🔄 Auto-restarting service {name}")
                                service.restart_count += 1
                                self.start_service(name)
                        
                        # Health check
                        elif service.config.health_url:
                            if not self._is_service_healthy(name):
                                logger.warning(f"⚠️ Service {name} failed health check")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in service monitoring: {e}")
                time.sleep(30)
    
    def _start_log_monitoring(self, service: ServiceState):
        """Start log monitoring for a service"""
        
        if not service.process:
            return
        
        def monitor_logs():
            try:
                for line in service.process.stdout:
                    if line:
                        # Add to queue for processing
                        service.output_queue.put((datetime.now(), line.strip()))
                        
                        # Log based on service
                        service_logger = logging.getLogger(f"service.{service.config.name}")
                        service_logger.info(line.strip())
            except:
                pass
        
        log_thread = threading.Thread(target=monitor_logs, daemon=True)
        log_thread.start()
    
    def _start_docker_services(self) -> bool:
        """Start services using Docker Compose"""
        
        logger.info("🐳 Starting Docker services...")
        
        compose_file = self.root_path / "docker-compose.yml"
        if not compose_file.exists():
            logger.error("❌ docker-compose.yml not found")
            return False
        
        try:
            # Start Docker Compose
            subprocess.run([
                "docker-compose", "up", "-d"
            ], cwd=self.root_path, check=True)
            
            logger.info("✅ Docker services started")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to start Docker services: {e}")
            return False
        except FileNotFoundError:
            logger.error("❌ Docker Compose not found. Please install Docker.")
            return False
    
    def _run_tests(self) -> bool:
        """Run test suite"""
        
        logger.info("🧪 Running test suite...")
        
        try:
            # Run pytest
            result = subprocess.run([
                sys.executable, "-m", "pytest", "src/tests/", "-v"
            ], cwd=self.root_path)
            
            if result.returncode == 0:
                logger.info("✅ All tests passed")
                return True
            else:
                logger.error("❌ Some tests failed")
                return False
                
        except FileNotFoundError:
            logger.error("❌ pytest not found. Install with: pip install pytest")
            return False
    
    def _open_browser(self):
        """Open browser to dashboard"""
        
        def open_delayed():
            time.sleep(5)  # Wait for dashboard to be ready
            try:
                webbrowser.open("http://localhost:8501")
                logger.info("🌐 Opened browser to dashboard")
            except:
                logger.info("🌐 Dashboard available at: http://localhost:8501")
        
        browser_thread = threading.Thread(target=open_delayed, daemon=True)
        browser_thread.start()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                "uptime": time.time() - psutil.boot_time()
            }
        except:
            return {}
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.stop_all()
        sys.exit(0)

# ==================== CONFIGURATION VALIDATOR ====================

class ConfigValidator:
    """Validates application configuration before startup"""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """Validate configuration and environment"""
        
        logger.info("🔍 Validating configuration...")
        
        # Check required files
        required_files = [
            "main_dashboard.py",
            "main_server.py",
            "requirements.txt",
            "src/core/neurocluster_elite.py"
        ]
        
        for file_path in required_files:
            if not (self.root_path / file_path).exists():
                self.errors.append(f"Missing required file: {file_path}")
        
        # Check configuration files
        config_files = [
            "config/default_config.yaml",
            "config/trading_config.yaml",
            ".env"
        ]
        
        for config_file in config_files:
            config_path = self.root_path / config_file
            if not config_path.exists():
                self.warnings.append(f"Missing config file: {config_file}")
            else:
                self._validate_config_file(config_path)
        
        # Check Python dependencies
        self._validate_dependencies()
        
        # Check ports
        self._validate_ports()
        
        # Check directories
        self._validate_directories()
        
        # Print results
        if self.errors:
            logger.error("❌ Configuration validation failed:")
            for error in self.errors:
                logger.error(f"   • {error}")
            return False
        
        if self.warnings:
            logger.warning("⚠️ Configuration warnings:")
            for warning in self.warnings:
                logger.warning(f"   • {warning}")
        
        logger.info("✅ Configuration validation passed")
        return True
    
    def _validate_config_file(self, config_path: Path):
        """Validate a specific configuration file"""
        
        try:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                with open(config_path, 'r') as f:
                    yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    json.load(f)
            elif config_path.name == '.env':
                # Basic .env validation
                with open(config_path, 'r') as f:
                    content = f.read()
                    if 'PAPER_TRADING=true' not in content:
                        self.warnings.append("PAPER_TRADING not set to true in .env")
        except Exception as e:
            self.errors.append(f"Invalid config file {config_path}: {e}")
    
    def _validate_dependencies(self):
        """Validate Python dependencies"""
        
        try:
            import streamlit
            import plotly
            import pandas
            import numpy
        except ImportError as e:
            self.errors.append(f"Missing dependency: {e}")
    
    def _validate_ports(self):
        """Validate that required ports are available"""
        
        import socket
        
        ports_to_check = [8501, 8000]
        
        for port in ports_to_check:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                except OSError:
                    self.warnings.append(f"Port {port} is already in use")
    
    def _validate_directories(self):
        """Validate required directories exist"""
        
        required_dirs = [
            "src",
            "config",
            "data",
            "logs"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.root_path / dir_name
            if not dir_path.exists():
                self.warnings.append(f"Missing directory: {dir_name}")

# ==================== MAIN LAUNCHER ====================

class NeuroClusterLauncher:
    """Main launcher class"""
    
    def __init__(self):
        self.root_path = Path.cwd()
        self.service_manager = ServiceManager(self.root_path)
        self.validator = ConfigValidator(self.root_path)
    
    def launch(self, mode: LaunchMode, validate: bool = True, background: bool = False) -> bool:
        """Launch the application"""
        
        print("🚀 NeuroCluster Elite Launcher")
        print("=" * 50)
        print(f"📍 Project Path: {self.root_path}")
        print(f"🎯 Launch Mode: {mode.value}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        try:
            # Validate configuration
            if validate and not self.validator.validate():
                logger.error("❌ Configuration validation failed")
                return False
            
            # Start services
            if not self.service_manager.start_mode(mode):
                logger.error("❌ Failed to start services")
                return False
            
            if background:
                logger.info("✅ Services started in background")
                return True
            
            # Interactive mode - wait for user input
            self._interactive_loop()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested")
            return True
        except Exception as e:
            logger.error(f"❌ Launch failed: {e}")
            return False
        finally:
            self.service_manager.stop_all()
    
    def _interactive_loop(self):
        """Interactive command loop"""
        
        print("\n🎛️ NeuroCluster Elite Control Panel")
        print("Commands: status, restart <service>, stop <service>, logs <service>, quit")
        print("Press Ctrl+C or type 'quit' to exit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "status":
                    self._show_status()
                elif command.startswith("restart "):
                    service_name = command.split(" ", 1)[1]
                    self.service_manager.restart_service(service_name)
                elif command.startswith("stop "):
                    service_name = command.split(" ", 1)[1]
                    self.service_manager.stop_service(service_name)
                elif command.startswith("logs "):
                    service_name = command.split(" ", 1)[1]
                    self._show_logs(service_name)
                elif command == "help":
                    self._show_help()
                elif command == "":
                    continue
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    def _show_status(self):
        """Show service status"""
        
        status = self.service_manager.get_status()
        
        print("\n📊 Service Status:")
        print("-" * 60)
        
        for name, service_info in status["services"].items():
            status_emoji = {
                "running": "✅",
                "stopped": "⭕",
                "starting": "🔄",
                "stopping": "🛑",
                "error": "❌"
            }.get(service_info["status"], "❓")
            
            print(f"{status_emoji} {name:<12} {service_info['status']:<10}", end="")
            
            if service_info["pid"]:
                print(f" PID: {service_info['pid']:<8}", end="")
            
            if service_info["port"]:
                print(f" Port: {service_info['port']}", end="")
            
            print()
        
        # System info
        system = status["system"]
        if system:
            print(f"\n💻 System: CPU {system.get('cpu_percent', 0):.1f}% | "
                  f"Memory {system.get('memory_percent', 0):.1f}% | "
                  f"Disk {system.get('disk_percent', 0):.1f}%")
    
    def _show_logs(self, service_name: str):
        """Show recent logs for a service"""
        
        if service_name not in self.service_manager.services:
            print(f"❌ Unknown service: {service_name}")
            return
        
        service = self.service_manager.services[service_name]
        
        print(f"\n📋 Recent logs for {service_name}:")
        print("-" * 60)
        
        # Show last 20 log entries
        logs = []
        try:
            while len(logs) < 20:
                timestamp, line = service.output_queue.get_nowait()
                logs.append((timestamp, line))
        except queue.Empty:
            pass
        
        for timestamp, line in logs[-20:]:
            print(f"{timestamp.strftime('%H:%M:%S')} | {line}")
        
        if not logs:
            print("No recent logs available")
    
    def _show_help(self):
        """Show help information"""
        
        print("\n📚 Available Commands:")
        print("-" * 40)
        print("status                 - Show service status")
        print("restart <service>      - Restart a service")
        print("stop <service>         - Stop a service")
        print("logs <service>         - Show recent logs")
        print("help                   - Show this help")
        print("quit/exit              - Exit the launcher")
        print("\nServices: dashboard, api, console, redis")
        print("\n🌐 URLs:")
        print("Dashboard: http://localhost:8501")
        print("API:       http://localhost:8000")
        print("API Docs:  http://localhost:8000/docs")

# ==================== MAIN FUNCTION ====================

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="NeuroCluster Elite Application Launcher")
    parser.add_argument("mode", nargs="?", default="dashboard", 
                       choices=[mode.value for mode in LaunchMode],
                       help="Launch mode")
    parser.add_argument("--no-validate", action="store_true", 
                       help="Skip configuration validation")
    parser.add_argument("--background", "-b", action="store_true",
                       help="Run in background mode")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        launcher = NeuroClusterLauncher()
        mode = LaunchMode(args.mode)
        
        success = launcher.launch(
            mode=mode,
            validate=not args.no_validate,
            background=args.background
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Launcher interrupted")
        return 0
    except Exception as e:
        logger.error(f"❌ Launcher failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)