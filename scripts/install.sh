#!/bin/bash

# ============================================================================
# NeuroCluster Elite - Automated Installation Script
# ============================================================================
# 
# File: install.sh
# Path: NeuroCluster-Elite/scripts/install.sh
# Description: Comprehensive installation script for NeuroCluster Elite
#
# This script handles:
# - System requirements verification
# - Python environment setup
# - Dependency installation  
# - Directory structure creation
# - Configuration initialization
# - Database setup
# - Service configuration
# - Security setup
# - Docker environment (optional)
#
# Usage:
#   chmod +x scripts/install.sh
#   ./scripts/install.sh [options]
#
# Options:
#   --dev          Install development dependencies
#   --docker       Setup Docker environment
#   --minimal      Minimal installation (core only)
#   --full         Full installation with all features
#   --production   Production-ready installation
#   --unattended   Non-interactive installation
#   --help         Show this help message
#
# Author: NeuroCluster Elite Team
# Created: 2025-06-30
# Version: 1.0.0
# License: MIT
# ============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ==================== CONFIGURATION ====================

# Script configuration
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="NeuroCluster Elite Installer"
MINIMUM_PYTHON_VERSION="3.8"
REQUIRED_DISK_SPACE_GB=2
REQUIRED_RAM_MB=1024

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Installation modes
INSTALL_MODE="standard"
INTERACTIVE=true
INSTALL_DOCKER=false
INSTALL_DEV=false
CREATE_VENV=true
SETUP_SYSTEMD=false

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/neurocluster-env"
LOG_FILE="$PROJECT_ROOT/install.log"
CONFIG_DIR="$PROJECT_ROOT/config"
DATA_DIR="$PROJECT_ROOT/data"

# System detection
OS_TYPE=""
ARCH=""
PACKAGE_MANAGER=""
PYTHON_CMD=""

# ==================== HELPER FUNCTIONS ====================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $*" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "${PURPLE}"
    echo "============================================================================"
    echo "  🧠 NeuroCluster Elite - Advanced AI Trading Platform"
    echo "============================================================================"
    echo -e "${NC}"
    echo -e "${CYAN}Version:${NC} $SCRIPT_VERSION"
    echo -e "${CYAN}Mode:${NC} $INSTALL_MODE"
    echo -e "${CYAN}Project Root:${NC} $PROJECT_ROOT"
    echo -e "${CYAN}Log File:${NC} $LOG_FILE"
    echo ""
}

print_help() {
    echo -e "${WHITE}$SCRIPT_NAME v$SCRIPT_VERSION${NC}"
    echo ""
    echo "USAGE:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --dev          Install development dependencies"
    echo "  --docker       Setup Docker environment"
    echo "  --minimal      Minimal installation (core only)"
    echo "  --full         Full installation with all features"
    echo "  --production   Production-ready installation"
    echo "  --unattended   Non-interactive installation"
    echo "  --no-venv      Skip virtual environment creation"
    echo "  --systemd      Setup systemd services"
    echo "  --help         Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                     # Standard installation"
    echo "  $0 --full --docker    # Full installation with Docker"
    echo "  $0 --dev              # Development setup"
    echo "  $0 --production       # Production deployment"
    echo ""
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

version_gt() {
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"
}

confirm() {
    if [[ "$INTERACTIVE" == "false" ]]; then
        return 0
    fi
    
    local prompt="$1"
    local default="${2:-y}"
    
    if [[ "$default" == "y" ]]; then
        prompt="$prompt [Y/n]: "
    else
        prompt="$prompt [y/N]: "
    fi
    
    while true; do
        read -p "$prompt" response
        case "$response" in
            [Yy]* ) return 0 ;;
            [Nn]* ) return 1 ;;
            "" ) [[ "$default" == "y" ]] && return 0 || return 1 ;;
            * ) echo "Please answer yes or no." ;;
        esac
    done
}

# ==================== SYSTEM DETECTION ====================

detect_system() {
    log "🔍 Detecting system configuration..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="linux"
        if [[ -f /etc/debian_version ]]; then
            PACKAGE_MANAGER="apt"
        elif [[ -f /etc/redhat-release ]]; then
            PACKAGE_MANAGER="yum"
        elif [[ -f /etc/arch-release ]]; then
            PACKAGE_MANAGER="pacman"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        PACKAGE_MANAGER="brew"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS_TYPE="windows"
        PACKAGE_MANAGER="choco"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Detect architecture
    ARCH=$(uname -m)
    
    # Find Python
    for py_cmd in python3.11 python3.10 python3.9 python3.8 python3 python; do
        if check_command "$py_cmd"; then
            PYTHON_CMD="$py_cmd"
            break
        fi
    done
    
    log_info "OS: $OS_TYPE ($ARCH)"
    log_info "Package Manager: $PACKAGE_MANAGER"
    log_info "Python Command: $PYTHON_CMD"
}

# ==================== SYSTEM REQUIREMENTS ====================

check_requirements() {
    log "📋 Checking system requirements..."
    
    local requirements_met=true
    
    # Check Python version
    if [[ -n "$PYTHON_CMD" ]]; then
        local python_version
        python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
        log_info "Python version: $python_version"
        
        if version_gt "$MINIMUM_PYTHON_VERSION" "$python_version"; then
            log_error "Python $MINIMUM_PYTHON_VERSION or higher required (found $python_version)"
            requirements_met=false
        fi
    else
        log_error "Python not found. Please install Python $MINIMUM_PYTHON_VERSION or higher"
        requirements_met=false
    fi
    
    # Check pip
    if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        log_error "pip not found. Please install pip"
        requirements_met=false
    fi
    
    # Check disk space
    local available_space
    if [[ "$OS_TYPE" == "macos" ]]; then
        available_space=$(df -g . | tail -1 | awk '{print $4}')
    else
        available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    fi
    
    if [[ "$available_space" -lt "$REQUIRED_DISK_SPACE_GB" ]]; then
        log_error "Insufficient disk space. Required: ${REQUIRED_DISK_SPACE_GB}GB, Available: ${available_space}GB"
        requirements_met=false
    fi
    
    # Check RAM
    local total_ram
    if [[ "$OS_TYPE" == "linux" ]]; then
        total_ram=$(free -m | grep '^Mem:' | awk '{print $2}')
    elif [[ "$OS_TYPE" == "macos" ]]; then
        total_ram=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024)}')
    fi
    
    if [[ -n "$total_ram" && "$total_ram" -lt "$REQUIRED_RAM_MB" ]]; then
        log_warn "Low RAM detected. Recommended: ${REQUIRED_RAM_MB}MB, Available: ${total_ram}MB"
    fi
    
    # Check required commands
    local required_commands=("git" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! check_command "$cmd"; then
            log_error "Required command not found: $cmd"
            requirements_met=false
        fi
    done
    
    if [[ "$requirements_met" == "false" ]]; then
        log_error "System requirements not met. Please address the issues above."
        exit 1
    fi
    
    log "✅ System requirements satisfied"
}

# ==================== DEPENDENCY INSTALLATION ====================

install_system_dependencies() {
    log "📦 Installing system dependencies..."
    
    case "$PACKAGE_MANAGER" in
        "apt")
            sudo apt update
            sudo apt install -y python3-pip python3-venv python3-dev \
                build-essential libssl-dev libffi-dev git curl wget \
                sqlite3 libsqlite3-dev portaudio19-dev python3-pyaudio
            ;;
        "yum")
            sudo yum update -y
            sudo yum install -y python3-pip python3-devel gcc openssl-devel \
                libffi-devel git curl wget sqlite-devel portaudio-devel
            ;;
        "brew")
            brew update
            brew install python3 git curl wget sqlite portaudio
            ;;
        "pacman")
            sudo pacman -Syu --noconfirm python-pip python git curl wget \
                sqlite base-devel portaudio
            ;;
        *)
            log_warn "Unsupported package manager: $PACKAGE_MANAGER"
            log_warn "Please manually install: python3-pip, python3-dev, git, curl, sqlite3"
            ;;
    esac
    
    log "✅ System dependencies installed"
}

create_virtual_environment() {
    if [[ "$CREATE_VENV" == "false" ]]; then
        log "⏭️ Skipping virtual environment creation"
        return
    fi
    
    log "🐍 Creating Python virtual environment..."
    
    if [[ -d "$VENV_PATH" ]]; then
        if confirm "Virtual environment already exists. Recreate it?"; then
            rm -rf "$VENV_PATH"
        else
            log "Using existing virtual environment"
            return
        fi
    fi
    
    $PYTHON_CMD -m venv "$VENV_PATH"
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log "✅ Virtual environment created: $VENV_PATH"
}

install_python_dependencies() {
    log "📚 Installing Python dependencies..."
    
    # Activate virtual environment if it exists
    if [[ -d "$VENV_PATH" && "$CREATE_VENV" == "true" ]]; then
        source "$VENV_PATH/bin/activate"
    fi
    
    # Install based on mode
    case "$INSTALL_MODE" in
        "minimal")
            pip install -e "."
            ;;
        "development")
            pip install -e ".[dev]"
            ;;
        "full")
            pip install -e ".[full]"
            ;;
        "production")
            pip install -e ".[full]"
            pip install gunicorn supervisor
            ;;
        *)
            pip install -e ".[trading]"
            ;;
    esac
    
    log "✅ Python dependencies installed"
}

# ==================== PROJECT SETUP ====================

create_directory_structure() {
    log "📁 Creating directory structure..."
    
    # Run the directory creation script
    if [[ -f "$PROJECT_ROOT/create_directories.py" ]]; then
        $PYTHON_CMD "$PROJECT_ROOT/create_directories.py"
    else
        # Create essential directories manually
        local directories=(
            "$DATA_DIR"
            "$DATA_DIR/cache"
            "$DATA_DIR/logs" 
            "$DATA_DIR/exports"
            "$PROJECT_ROOT/logs"
        )
        
        for dir in "${directories[@]}"; do
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        done
    fi
    
    log "✅ Directory structure created"
}

setup_configuration() {
    log "⚙️ Setting up configuration..."
    
    # Copy environment file
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
            cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
            log_info "Created .env from template"
        else
            log_warn ".env.example not found, creating basic .env"
            cat > "$PROJECT_ROOT/.env" << EOF
# NeuroCluster Elite Configuration
# Created by installer on $(date)

PAPER_TRADING=true
INITIAL_CAPITAL=100000
DEFAULT_STOCKS=AAPL,GOOGL,MSFT
RISK_LEVEL=moderate
LOG_LEVEL=INFO
EOF
        fi
    else
        log_info ".env already exists, skipping"
    fi
    
    # Set appropriate permissions
    chmod 600 "$PROJECT_ROOT/.env"
    
    log "✅ Configuration setup complete"
}

initialize_database() {
    log "🗄️ Initializing database..."
    
    # Activate virtual environment if it exists
    if [[ -d "$VENV_PATH" && "$CREATE_VENV" == "true" ]]; then
        source "$VENV_PATH/bin/activate"
    fi
    
    # Create database directory
    mkdir -p "$DATA_DIR"
    
    # Run database initialization if script exists
    if [[ -f "$PROJECT_ROOT/src/utils/database.py" ]]; then
        $PYTHON_CMD -c "
from src.utils.database import DatabaseManager
db = DatabaseManager()
db.initialize_database()
print('Database initialized successfully')
"
    fi
    
    log "✅ Database initialized"
}

# ==================== DOCKER SETUP ====================

setup_docker() {
    if [[ "$INSTALL_DOCKER" == "false" ]]; then
        return
    fi
    
    log "🐳 Setting up Docker environment..."
    
    # Check if Docker is installed
    if ! check_command "docker"; then
        log_error "Docker not found. Please install Docker first."
        return 1
    fi
    
    if ! check_command "docker-compose"; then
        log_error "Docker Compose not found. Please install Docker Compose first."
        return 1
    fi
    
    # Build Docker images
    if [[ -f "$PROJECT_ROOT/docker/docker-compose.yml" ]]; then
        cd "$PROJECT_ROOT"
        docker-compose build
        log "✅ Docker images built"
    else
        log_warn "Docker Compose file not found"
    fi
}

# ==================== SERVICE SETUP ====================

setup_systemd_services() {
    if [[ "$SETUP_SYSTEMD" == "false" || "$OS_TYPE" != "linux" ]]; then
        return
    fi
    
    log "🔧 Setting up systemd services..."
    
    # Create systemd service file
    local service_file="/etc/systemd/system/neurocluster-elite.service"
    local user_name=$(whoami)
    
    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=NeuroCluster Elite Trading Platform
After=network.target

[Service]
Type=simple
User=$user_name
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$VENV_PATH/bin
ExecStart=$VENV_PATH/bin/python main_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable neurocluster-elite.service
    
    log "✅ Systemd service configured"
    log_info "Use 'sudo systemctl start neurocluster-elite' to start the service"
}

# ==================== TESTING ====================

run_tests() {
    log "🧪 Running installation tests..."
    
    # Activate virtual environment if it exists
    if [[ -d "$VENV_PATH" && "$CREATE_VENV" == "true" ]]; then
        source "$VENV_PATH/bin/activate"
    fi
    
    # Test imports
    $PYTHON_CMD -c "
try:
    from src.core.neurocluster_elite import NeuroClusterElite
    from src.trading.trading_engine import AdvancedTradingEngine
    from src.data.multi_asset_manager import MultiAssetDataManager
    print('✅ Core modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"
    
    # Test configuration
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        log_info "Configuration file exists"
    else
        log_warn "Configuration file missing"
    fi
    
    # Test database connectivity
    if [[ -f "$DATA_DIR/database.db" ]]; then
        log_info "Database file exists"
    fi
    
    log "✅ Installation tests completed"
}

# ==================== POST-INSTALLATION ====================

print_completion_message() {
    echo ""
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}  🎉 NeuroCluster Elite Installation Complete!${NC}"
    echo -e "${GREEN}============================================================================${NC}"
    echo ""
    echo -e "${CYAN}📍 Installation Path:${NC} $PROJECT_ROOT"
    if [[ "$CREATE_VENV" == "true" ]]; then
        echo -e "${CYAN}🐍 Virtual Environment:${NC} $VENV_PATH"
    fi
    echo -e "${CYAN}📊 Configuration:${NC} $PROJECT_ROOT/.env"
    echo -e "${CYAN}📝 Logs:${NC} $LOG_FILE"
    echo ""
    echo -e "${YELLOW}🚀 Quick Start Commands:${NC}"
    echo ""
    
    if [[ "$CREATE_VENV" == "true" ]]; then
        echo -e "${WHITE}# Activate virtual environment:${NC}"
        echo "source $VENV_PATH/bin/activate"
        echo ""
    fi
    
    echo -e "${WHITE}# Launch Streamlit Dashboard:${NC}"
    echo "streamlit run main_dashboard.py"
    echo ""
    echo -e "${WHITE}# Launch Console Interface:${NC}"
    echo "python main_console.py"
    echo ""
    echo -e "${WHITE}# Launch API Server:${NC}"
    echo "python main_server.py"
    echo ""
    
    if [[ "$INSTALL_DOCKER" == "true" ]]; then
        echo -e "${WHITE}# Launch with Docker:${NC}"
        echo "docker-compose up -d"
        echo ""
    fi
    
    echo -e "${CYAN}📖 Documentation:${NC}"
    echo "  - API Reference: docs/API_REFERENCE.md"
    echo "  - Strategy Guide: docs/STRATEGY_GUIDE.md"
    echo "  - Voice Commands: docs/VOICE_COMMANDS.md"
    echo ""
    echo -e "${CYAN}🌐 Web Interfaces:${NC}"
    echo "  - Dashboard: http://localhost:8501"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo ""
    echo -e "${GREEN}Happy Trading! 📈${NC}"
    echo ""
}

# ==================== ARGUMENT PARSING ====================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev|--development)
                INSTALL_MODE="development"
                INSTALL_DEV=true
                shift
                ;;
            --docker)
                INSTALL_DOCKER=true
                shift
                ;;
            --minimal)
                INSTALL_MODE="minimal"
                shift
                ;;
            --full)
                INSTALL_MODE="full"
                shift
                ;;
            --production)
                INSTALL_MODE="production"
                SETUP_SYSTEMD=true
                shift
                ;;
            --unattended)
                INTERACTIVE=false
                shift
                ;;
            --no-venv)
                CREATE_VENV=false
                shift
                ;;
            --systemd)
                SETUP_SYSTEMD=true
                shift
                ;;
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_help
                exit 1
                ;;
        esac
    done
}

# ==================== MAIN INSTALLATION FLOW ====================

main() {
    # Initialize log file
    echo "NeuroCluster Elite Installation Log - $(date)" > "$LOG_FILE"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Print header
    print_header
    
    # Detect system
    detect_system
    
    # Check requirements
    check_requirements
    
    # Confirm installation
    if [[ "$INTERACTIVE" == "true" ]]; then
        echo -e "${YELLOW}Installation Configuration:${NC}"
        echo "  Mode: $INSTALL_MODE"
        echo "  Docker: $INSTALL_DOCKER"
        echo "  Virtual Environment: $CREATE_VENV"
        echo "  Systemd Services: $SETUP_SYSTEMD"
        echo ""
        
        if ! confirm "Proceed with installation?"; then
            log "Installation cancelled by user"
            exit 0
        fi
    fi
    
    # Start installation
    log "🚀 Starting NeuroCluster Elite installation..."
    
    # Install system dependencies
    install_system_dependencies
    
    # Create virtual environment
    create_virtual_environment
    
    # Install Python dependencies
    install_python_dependencies
    
    # Create directory structure
    create_directory_structure
    
    # Setup configuration
    setup_configuration
    
    # Initialize database
    initialize_database
    
    # Setup Docker (if requested)
    setup_docker
    
    # Setup systemd services (if requested)
    setup_systemd_services
    
    # Run tests
    run_tests
    
    # Print completion message
    print_completion_message
    
    log "🎉 NeuroCluster Elite installation completed successfully!"
}

# ==================== SCRIPT EXECUTION ====================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi