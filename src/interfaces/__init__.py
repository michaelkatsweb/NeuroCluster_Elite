#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/interfaces/__init__.py
Description: Interfaces package initialization

This module initializes the user interface components including Streamlit dashboard,
console interface, mobile API, voice commands, and UI components.

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

# Import main interface components
try:
    from .streamlit_dashboard import NeuroClusterDashboard, DashboardConfig
    from .console_interface import ConsoleInterface, ConsoleCommand
    from .mobile_api import MobileAPI, APIEndpoint
    from .voice_commands import VoiceCommandSystem, VoiceCommand
    
    # Import UI components
    from .components.charts import (
        create_price_chart, create_volume_chart, create_sentiment_chart,
        create_portfolio_chart, create_performance_chart
    )
    from .components.widgets import (
        create_metric_widget, create_alert_widget, create_status_widget,
        create_trade_widget, create_scanner_widget
    )
    from .components.layouts import (
        create_main_layout, create_trading_layout, create_analysis_layout,
        create_portfolio_layout, create_settings_layout
    )
    
    __all__ = [
        # Main interfaces
        'NeuroClusterDashboard',
        'DashboardConfig',
        'ConsoleInterface', 
        'ConsoleCommand',
        'MobileAPI',
        'APIEndpoint',
        'VoiceCommandSystem',
        'VoiceCommand',
        
        # Chart components
        'create_price_chart',
        'create_volume_chart',
        'create_sentiment_chart',
        'create_portfolio_chart',
        'create_performance_chart',
        
        # Widget components
        'create_metric_widget',
        'create_alert_widget',
        'create_status_widget',
        'create_trade_widget',
        'create_scanner_widget',
        
        # Layout components
        'create_main_layout',
        'create_trading_layout',
        'create_analysis_layout',
        'create_portfolio_layout',
        'create_settings_layout'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some interface components could not be imported: {e}")
    __all__ = []

# Interface module constants
SUPPORTED_INTERFACES = ['streamlit', 'console', 'mobile', 'voice']
DEFAULT_THEME = 'dark'
DEFAULT_UPDATE_INTERVAL = 5  # seconds
MAX_CONCURRENT_USERS = 100

# UI Configuration
UI_CONFIG = {
    'streamlit': {
        'page_title': 'NeuroCluster Elite',
        'page_icon': '🧠',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    },
    'console': {
        'prompt': 'neurocluster> ',
        'history_size': 1000,
        'auto_complete': True
    },
    'mobile': {
        'api_version': 'v1',
        'rate_limit': 1000,  # requests per hour
        'authentication': True
    },
    'voice': {
        'wake_word': 'neurocluster',
        'language': 'en-US',
        'confidence_threshold': 0.7
    }
}

# Color schemes
COLOR_SCHEMES = {
    'dark': {
        'background': '#0E1117',
        'secondary_background': '#262730',
        'text': '#FAFAFA',
        'primary': '#FF6B6B',
        'success': '#4ECDC4',
        'warning': '#FFD93D',
        'error': '#FF6B6B',
        'info': '#6BCF7F'
    },
    'light': {
        'background': '#FFFFFF',
        'secondary_background': '#F0F2F6',
        'text': '#262730',
        'primary': '#FF4B4B',
        'success': '#09AB3B',
        'warning': '#FFBD45',
        'error': '#FF4B4B',
        'info': '#1f77b4'
    }
}

# Chart configurations
CHART_CONFIG = {
    'default_height': 400,
    'default_width': 800,
    'animation_duration': 300,
    'grid_color': 'rgba(128, 128, 128, 0.2)',
    'font_family': 'Arial, sans-serif',
    'font_size': 12
}

# Widget configurations
WIDGET_CONFIG = {
    'metric': {
        'decimal_places': 2,
        'show_delta': True,
        'delta_color': 'normal'
    },
    'alert': {
        'auto_dismiss': True,
        'dismiss_time': 5000,  # milliseconds
        'max_alerts': 10
    },
    'status': {
        'update_interval': 1000,  # milliseconds
        'show_timestamp': True
    }
}

# Layout configurations
LAYOUT_CONFIG = {
    'sidebar_width': 300,
    'main_content_padding': 20,
    'widget_spacing': 15,
    'section_spacing': 30
}

def get_interface_info():
    """Get interface module information"""
    return {
        'supported_interfaces': len(SUPPORTED_INTERFACES),
        'available_components': len(__all__),
        'color_schemes': list(COLOR_SCHEMES.keys()),
        'default_theme': DEFAULT_THEME,
        'update_interval': f"{DEFAULT_UPDATE_INTERVAL}s",
        'max_users': MAX_CONCURRENT_USERS
    }

def get_ui_config(interface_type: str = 'streamlit') -> dict:
    """Get UI configuration for specific interface type"""
    return UI_CONFIG.get(interface_type, {})

def get_color_scheme(theme: str = DEFAULT_THEME) -> dict:
    """Get color scheme for specified theme"""
    return COLOR_SCHEMES.get(theme, COLOR_SCHEMES[DEFAULT_THEME])

def initialize_interface_logging():
    """Initialize logging for interface components"""
    import logging
    
    # Create interface logger
    interface_logger = logging.getLogger('neurocluster.interfaces')
    interface_logger.setLevel(logging.INFO)
    
    # Create handler if it doesn't exist
    if not interface_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        interface_logger.addHandler(handler)
    
    return interface_logger

# Initialize logging
logger = initialize_interface_logging()
logger.info("🖥️ NeuroCluster Elite interfaces initialized")

# Interface validation
def validate_interface_dependencies():
    """Validate that required dependencies are available"""
    missing_deps = []
    
    # Check Streamlit
    try:
        import streamlit
    except ImportError:
        missing_deps.append('streamlit')
    
    # Check FastAPI for mobile API
    try:
        import fastapi
    except ImportError:
        missing_deps.append('fastapi')
    
    # Check speech recognition for voice
    try:
        import speech_recognition
    except ImportError:
        missing_deps.append('speech_recognition')
    
    if missing_deps:
        logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
        logger.info("Install missing dependencies to enable all interface features")
    
    return len(missing_deps) == 0

# Run validation on import
validate_interface_dependencies()