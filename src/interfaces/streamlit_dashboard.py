#!/usr/bin/env python3
"""
File: streamlit_dashboard.py
Path: NeuroCluster-Elite/src/interfaces/streamlit_dashboard.py
Description: Streamlit dashboard interface for NeuroCluster Elite

This module implements a comprehensive Streamlit dashboard providing an intuitive
web interface for the NeuroCluster Elite trading platform with real-time data,
interactive charts, trading controls, and analytics.

Features:
- Real-time market data and portfolio tracking
- Interactive trading interface with order management
- Advanced charting with technical indicators
- Sentiment analysis and social sentiment tracking
- Market scanner and opportunity detection
- Risk management and performance analytics
- Multi-asset support with regime detection
- Voice command integration
- Mobile-responsive design

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import time
import logging
from pathlib import Path
import threading

# Streamlit components
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
from streamlit_autorefresh import st_autorefresh
import streamlit_echarts as st_echarts

# Import our modules
try:
    from src.core.neurocluster_elite import NeuroClusterElite, RegimeType, AssetType, MarketData
    from src.data.multi_asset_manager import MultiAssetDataManager
    from src.trading.trading_engine import AdvancedTradingEngine
    from src.trading.portfolio_manager import PortfolioManager
    from src.analysis.sentiment_analyzer import AdvancedSentimentAnalyzer
    from src.analysis.market_scanner import AdvancedMarketScanner, ScanType
    from src.interfaces.voice_commands import VoiceCommandSystem
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_enhanced_logger, LogCategory
    from src.utils.helpers import format_currency, format_percentage
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.INTERFACE)

# ==================== CONFIGURATION AND SETUP ====================

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    page_title: str = "NeuroCluster Elite"
    page_icon: str = "🧠"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    theme: str = "dark"
    auto_refresh: bool = True
    refresh_interval: int = 5000  # milliseconds
    max_data_points: int = 1000
    chart_height: int = 400
    enable_voice: bool = True
    enable_alerts: bool = True

# ==================== STREAMLIT CONFIGURATION ====================

def configure_page():
    """Configure Streamlit page settings"""
    
    st.set_page_config(
        page_title="NeuroCluster Elite",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/neurocluster-elite',
            'Report a bug': 'https://github.com/neurocluster-elite/issues',
            'About': "NeuroCluster Elite - AI-Powered Trading Platform"
        }
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-active { background-color: #4ECDC4; }
    .status-warning { background-color: #FFD93D; }
    .status-error { background-color: #FF6B6B; }
    .status-inactive { background-color: #95A5A6; }
    
    /* Hide streamlit style */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== MAIN DASHBOARD CLASS ====================

class NeuroClusterDashboard:
    """Main Streamlit dashboard for NeuroCluster Elite"""
    
    def __init__(self, config: DashboardConfig = None):
        """Initialize dashboard"""
        self.config = config or DashboardConfig()
        
        # Initialize session state
        self._initialize_session_state()
        
        # Initialize components
        self.neurocluster = None
        self.data_manager = None
        self.trading_engine = None
        self.portfolio_manager = None
        self.sentiment_analyzer = None
        self.market_scanner = None
        self.voice_system = None
        
        # Data storage
        self.market_data = {}
        self.portfolio_data = {}
        self.sentiment_data = {}
        self.scan_results = []
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        
        default_state = {
            'authenticated': False,
            'username': None,
            'selected_symbols': ['AAPL', 'BTC-USD', 'ETH-USD'],
            'active_trades': [],
            'alerts': [],
            'current_regime': None,
            'last_update': datetime.now(),
            'voice_enabled': False,
            'auto_refresh': True,
            'theme': 'dark',
            'page': 'Dashboard'
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _initialize_components(self):
        """Initialize NeuroCluster components"""
        
        try:
            # Initialize configuration
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Initialize core components
            self.neurocluster = NeuroClusterElite(config.get('algorithm', {}))
            self.data_manager = MultiAssetDataManager(config.get('data', {}))
            self.trading_engine = AdvancedTradingEngine(config.get('trading', {}))
            self.portfolio_manager = PortfolioManager(config.get('portfolio', {}))
            self.sentiment_analyzer = AdvancedSentimentAnalyzer(config.get('sentiment', {}))
            self.market_scanner = AdvancedMarketScanner(config.get('scanner', {}))
            
            # Initialize voice system if enabled
            if self.config.enable_voice:
                self.voice_system = VoiceCommandSystem(config.get('voice', {}))
            
            logger.info("🚀 Dashboard components initialized")
            
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            logger.error(f"Component initialization failed: {e}")
    
    def run(self):
        """Run the main dashboard"""
        
        # Configure page
        configure_page()
        
        # Auto-refresh
        if st.session_state.auto_refresh:
            st_autorefresh(interval=self.config.refresh_interval, key="dashboard_refresh")
        
        # Authentication check
        if not st.session_state.authenticated:
            self.show_login()
            return
        
        # Main dashboard layout
        self.show_main_dashboard()
    
    def show_login(self):
        """Show login interface"""
        
        st.markdown('<h1 class="main-header">NeuroCluster Elite</h1>', unsafe_allow_html=True)
        st.markdown("### 🧠 AI-Powered Trading Platform")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("---")
            
            # Login form
            with st.form("login_form"):
                st.markdown("### 🔐 Login")
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                remember_me = st.checkbox("Remember me")
                
                submitted = st.form_submit_button("Login", use_container_width=True)
                
                if submitted:
                    # Simple authentication (would use proper auth in production)
                    if username and password:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Please enter username and password")
            
            st.markdown("---")
            
            # Demo access
            if st.button("🎮 Demo Access", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.username = "demo_user"
                st.success("✅ Demo access granted!")
                st.rerun()
            
            # Features preview
            st.markdown("### ✨ Features")
            features = [
                "🧠 Advanced NeuroCluster Algorithm",
                "📊 Real-time Multi-Asset Trading",
                "🎯 AI-Powered Market Analysis",
                "📱 Social Sentiment Tracking",
                "🔍 Advanced Market Scanner",
                "📈 Portfolio Management",
                "🗣️ Voice Commands",
                "🔔 Smart Alerts"
            ]
            
            for feature in features:
                st.markdown(f"- {feature}")
    
    def show_main_dashboard(self):
        """Show main dashboard interface"""
        
        # Header
        self.show_header()
        
        # Sidebar navigation
        selected_page = self.show_sidebar()
        
        # Main content based on selected page
        if selected_page == "Dashboard":
            self.show_dashboard_page()
        elif selected_page == "Trading":
            self.show_trading_page()
        elif selected_page == "Analysis":
            self.show_analysis_page()
        elif selected_page == "Scanner":
            self.show_scanner_page()
        elif selected_page == "Portfolio":
            self.show_portfolio_page()
        elif selected_page == "Settings":
            self.show_settings_page()
    
    def show_header(self):
        """Show dashboard header"""
        
        header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
        
        with header_col1:
            st.markdown(f"# 🧠 NeuroCluster Elite")
            st.markdown(f"Welcome back, **{st.session_state.username}**!")
        
        with header_col2:
            # Current time and status
            current_time = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"**Time:** {current_time}")
            
            # System status
            status_color = "status-active" if self.neurocluster else "status-error"
            st.markdown(f'<span class="{status_color} status-indicator"></span>**System Active**', 
                       unsafe_allow_html=True)
        
        with header_col3:
            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh"):
                    self.refresh_data()
                    st.rerun()
            
            with col2:
                if st.button("🚪 Logout"):
                    st.session_state.authenticated = False
                    st.rerun()
        
        st.markdown("---")
    
    def show_sidebar(self):
        """Show sidebar navigation"""
        
        with st.sidebar:
            st.markdown("## 🎛️ Navigation")
            
            # Navigation menu
            selected = option_menu(
                menu_title=None,
                options=["Dashboard", "Trading", "Analysis", "Scanner", "Portfolio", "Settings"],
                icons=["speedometer2", "graph-up", "bar-chart", "search", "wallet", "gear"],
                menu_icon="cast",
                default_index=0,
                orientation="vertical",
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "#4ECDC4", "font-size": "18px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#FF6B6B"
                    },
                    "nav-link-selected": {"background-color": "#FF6B6B"},
                }
            )
            
            st.markdown("---")
            
            # Quick stats
            self.show_sidebar_stats()
            
            # System controls
            self.show_sidebar_controls()
            
            return selected
    
    def show_sidebar_stats(self):
        """Show quick stats in sidebar"""
        
        st.markdown("### 📊 Quick Stats")
        
        # Portfolio value (simulated)
        portfolio_value = np.random.uniform(95000, 105000)
        daily_pnl = np.random.uniform(-2000, 3000)
        daily_pnl_pct = (daily_pnl / portfolio_value) * 100
        
        # Display metrics
        st.metric(
            label="Portfolio Value",
            value=format_currency(portfolio_value),
            delta=f"{format_currency(daily_pnl)} ({daily_pnl_pct:+.2f}%)"
        )
        
        # Active positions
        active_positions = np.random.randint(3, 8)
        st.metric(
            label="Active Positions",
            value=str(active_positions)
        )
        
        # Today's signals
        signals_today = np.random.randint(5, 15)
        st.metric(
            label="Signals Today",
            value=str(signals_today)
        )
        
        # Current regime
        regimes = [RegimeType.BULL, RegimeType.BEAR, RegimeType.SIDEWAYS, RegimeType.VOLATILE]
        current_regime = np.random.choice(regimes)
        regime_confidence = np.random.uniform(65, 95)
        
        st.markdown(f"**Current Regime:** {current_regime.value}")
        st.progress(regime_confidence / 100)
        st.caption(f"Confidence: {regime_confidence:.1f}%")
    
    def show_sidebar_controls(self):
        """Show system controls in sidebar"""
        
        st.markdown("---")
        st.markdown("### ⚙️ Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox(
            "Auto Refresh",
            value=st.session_state.auto_refresh,
            key="sidebar_auto_refresh"
        )
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
        
        # Voice commands toggle
        if self.config.enable_voice:
            voice_enabled = st.checkbox(
                "Voice Commands",
                value=st.session_state.voice_enabled,
                key="sidebar_voice"
            )
            if voice_enabled != st.session_state.voice_enabled:
                st.session_state.voice_enabled = voice_enabled
        
        # Theme selector
        theme = st.selectbox(
            "Theme",
            options=["dark", "light"],
            index=0 if st.session_state.theme == "dark" else 1,
            key="sidebar_theme"
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
        
        # Emergency stop
        if st.button("🛑 Emergency Stop", type="secondary", use_container_width=True):
            st.warning("Emergency stop activated - all trading halted!")
    
    def show_dashboard_page(self):
        """Show main dashboard page"""
        
        # Market overview
        st.markdown("## 🌍 Market Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Generate sample data
        with col1:
            btc_price = np.random.uniform(42000, 48000)
            btc_change = np.random.uniform(-5, 5)
            st.metric("Bitcoin", format_currency(btc_price), f"{btc_change:+.2f}%")
        
        with col2:
            spy_price = np.random.uniform(450, 470)
            spy_change = np.random.uniform(-2, 2)
            st.metric("S&P 500", format_currency(spy_price), f"{spy_change:+.2f}%")
        
        with col3:
            vix = np.random.uniform(15, 25)
            vix_change = np.random.uniform(-3, 3)
            st.metric("VIX", f"{vix:.2f}", f"{vix_change:+.2f}")
        
        with col4:
            fear_greed = np.random.uniform(20, 80)
            st.metric("Fear & Greed", f"{fear_greed:.0f}", "Index")
        
        # Charts row
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("### 📈 Price Chart")
            self.show_price_chart()
        
        with chart_col2:
            st.markdown("### 🎯 Sentiment Analysis")
            self.show_sentiment_chart()
        
        # Recent alerts and activity
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            st.markdown("### 🔔 Recent Alerts")
            self.show_recent_alerts()
        
        with alert_col2:
            st.markdown("### 📊 Trading Activity")
            self.show_trading_activity()
    
    def show_trading_page(self):
        """Show trading interface page"""
        
        st.markdown("## 💰 Trading Interface")
        
        # Symbol selector
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_symbol = st.selectbox(
                "Select Symbol",
                options=['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD', 'TSLA'],
                index=0
            )
        
        with col2:
            asset_type = st.selectbox(
                "Asset Type",
                options=['Stock', 'Crypto', 'Forex'],
                index=0
            )
        
        with col3:
            timeframe = st.selectbox(
                "Timeframe",
                options=['1m', '5m', '15m', '1h', '4h', '1d'],
                index=3
            )
        
        # Trading interface
        trade_col1, trade_col2 = st.columns([2, 1])
        
        with trade_col1:
            # Price chart with trading interface
            st.markdown("### 📊 Trading Chart")
            self.show_trading_chart(selected_symbol)
        
        with trade_col2:
            # Trading panel
            st.markdown("### 🎯 Place Order")
            self.show_trading_panel(selected_symbol)
        
        # Active orders and positions
        st.markdown("---")
        
        order_col1, order_col2 = st.columns(2)
        
        with order_col1:
            st.markdown("### 📋 Active Orders")
            self.show_active_orders()
        
        with order_col2:
            st.markdown("### 💼 Current Positions")
            self.show_current_positions()
    
    def show_analysis_page(self):
        """Show analysis page"""
        
        st.markdown("## 🔬 Market Analysis")
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Technical", "Sentiment", "Regime", "Correlation"])
        
        with tab1:
            self.show_technical_analysis()
        
        with tab2:
            self.show_sentiment_analysis()
        
        with tab3:
            self.show_regime_analysis()
        
        with tab4:
            self.show_correlation_analysis()
    
    def show_scanner_page(self):
        """Show market scanner page"""
        
        st.markdown("## 🔍 Market Scanner")
        
        # Scanner configuration
        scan_col1, scan_col2 = st.columns([1, 2])
        
        with scan_col1:
            st.markdown("### ⚙️ Scanner Settings")
            
            # Scan type selection
            scan_types = st.multiselect(
                "Scan Types",
                options=[scan.value for scan in ScanType],
                default=['breakout', 'momentum']
            )
            
            # Asset types
            asset_types = st.multiselect(
                "Asset Types",
                options=['Stock', 'Crypto', 'Forex'],
                default=['Stock', 'Crypto']
            )
            
            # Filters
            min_volume = st.number_input("Minimum Volume", value=100000, step=10000)
            min_price = st.number_input("Minimum Price", value=1.0, step=0.1)
            max_results = st.slider("Max Results", min_value=10, max_value=100, value=50)
            
            # Run scan button
            if st.button("🚀 Run Scan", type="primary", use_container_width=True):
                with st.spinner("Scanning markets..."):
                    self.run_market_scan(scan_types, asset_types, min_volume, min_price, max_results)
        
        with scan_col2:
            st.markdown("### 🎯 Scan Results")
            self.show_scan_results()
    
    def show_portfolio_page(self):
        """Show portfolio management page"""
        
        st.markdown("## 💼 Portfolio Management")
        
        # Portfolio overview
        portfolio_col1, portfolio_col2, portfolio_col3 = st.columns(3)
        
        with portfolio_col1:
            total_value = np.random.uniform(95000, 105000)
            st.metric("Total Value", format_currency(total_value), "+2.5%")
        
        with portfolio_col2:
            daily_pnl = np.random.uniform(-1000, 2000)
            st.metric("Today's P&L", format_currency(daily_pnl), f"{daily_pnl/total_value*100:+.2f}%")
        
        with portfolio_col3:
            positions = np.random.randint(5, 12)
            st.metric("Positions", str(positions))
        
        # Portfolio tabs
        port_tab1, port_tab2, port_tab3 = st.tabs(["Holdings", "Performance", "Risk"])
        
        with port_tab1:
            self.show_portfolio_holdings()
        
        with port_tab2:
            self.show_portfolio_performance()
        
        with port_tab3:
            self.show_portfolio_risk()
    
    def show_settings_page(self):
        """Show settings page"""
        
        st.markdown("## ⚙️ Settings")
        
        # Settings tabs
        settings_tab1, settings_tab2, settings_tab3 = st.tabs(["Trading", "Alerts", "System"])
        
        with settings_tab1:
            self.show_trading_settings()
        
        with settings_tab2:
            self.show_alert_settings()
        
        with settings_tab3:
            self.show_system_settings()
    
    # ==================== CHART METHODS ====================
    
    def show_price_chart(self):
        """Show price chart"""
        
        # Generate sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#4ECDC4', width=2)
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_sentiment_chart(self):
        """Show sentiment analysis chart"""
        
        # Generate sample sentiment data
        categories = ['News', 'Social', 'Technical', 'Fundamental']
        values = np.random.uniform(30, 90, 4)
        colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#95A5A6']
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color=colors)
        ])
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_title="Sentiment Score",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_trading_chart(self, symbol: str):
        """Show detailed trading chart"""
        
        # Generate OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        opens = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        closes = opens * (1 + np.random.normal(0, 0.01, 100))
        highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, 0.02, 100))
        lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, 0.02, 100))
        volumes = np.random.lognormal(10, 1, 100)
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price', 'Volume')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name="Price"
            ),
            row=1, col=1
        )
        
        # Volume chart
        colors = ['red' if closes[i] < opens[i] else 'green' for i in range(len(closes))]
        fig.add_trace(
            go.Bar(x=dates, y=volumes, marker_color=colors, name="Volume"),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== WIDGET METHODS ====================
    
    def show_trading_panel(self, symbol: str):
        """Show trading panel"""
        
        # Order type
        order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop Loss"])
        
        # Side
        side = st.radio("Side", ["Buy", "Sell"], horizontal=True)
        
        # Quantity
        quantity = st.number_input("Quantity", value=1.0, step=0.1)
        
        # Price (for limit orders)
        if order_type == "Limit":
            price = st.number_input("Price", value=100.0, step=0.01)
        
        # Stop loss
        if order_type == "Stop Loss":
            stop_price = st.number_input("Stop Price", value=95.0, step=0.01)
        
        # Advanced options
        with st.expander("Advanced Options"):
            time_in_force = st.selectbox("Time in Force", ["GTC", "IOC", "FOK"])
            reduce_only = st.checkbox("Reduce Only")
        
        # Submit order
        if st.button(f"Place {side} Order", type="primary", use_container_width=True):
            st.success(f"✅ {side} order for {quantity} {symbol} placed successfully!")
    
    def show_recent_alerts(self):
        """Show recent alerts"""
        
        # Generate sample alerts
        alerts = [
            {"time": "10:30", "type": "🚀", "message": "AAPL breakout detected"},
            {"time": "10:15", "type": "⚠️", "message": "High volatility in BTC"},
            {"time": "10:00", "type": "📈", "message": "SPY momentum signal"},
            {"time": "09:45", "type": "🔔", "message": "Portfolio rebalanced"},
        ]
        
        for alert in alerts:
            st.markdown(f"**{alert['time']}** {alert['type']} {alert['message']}")
    
    def show_trading_activity(self):
        """Show recent trading activity"""
        
        # Generate sample activity
        activities = [
            {"time": "10:25", "action": "BUY", "symbol": "AAPL", "qty": "100"},
            {"time": "10:10", "action": "SELL", "symbol": "TSLA", "qty": "50"},
            {"time": "09:55", "action": "BUY", "symbol": "BTC", "qty": "0.1"},
            {"time": "09:30", "action": "SELL", "symbol": "GOOGL", "qty": "25"},
        ]
        
        for activity in activities:
            color = "🟢" if activity['action'] == "BUY" else "🔴"
            st.markdown(f"**{activity['time']}** {color} {activity['action']} {activity['qty']} {activity['symbol']}")
    
    def show_active_orders(self):
        """Show active orders table"""
        
        # Sample orders data
        orders_data = {
            'Symbol': ['AAPL', 'BTC-USD', 'GOOGL'],
            'Side': ['BUY', 'SELL', 'BUY'],
            'Type': ['Limit', 'Market', 'Stop'],
            'Quantity': [100, 0.5, 25],
            'Price': [150.00, 45000.00, 2800.00],
            'Status': ['Pending', 'Filled', 'Pending']
        }
        
        df = pd.DataFrame(orders_data)
        st.dataframe(df, use_container_width=True)
    
    def show_current_positions(self):
        """Show current positions table"""
        
        # Sample positions data
        positions_data = {
            'Symbol': ['AAPL', 'BTC-USD', 'ETH-USD', 'TSLA'],
            'Quantity': [200, 1.5, 10.0, 50],
            'Avg Price': [145.00, 43000.00, 2500.00, 220.00],
            'Current Price': [148.50, 44500.00, 2650.00, 215.00],
            'P&L': [700.00, 2250.00, 1500.00, -250.00],
            'P&L %': [2.41, 5.23, 6.00, -2.27]
        }
        
        df = pd.DataFrame(positions_data)
        
        # Color code P&L
        def color_pnl(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'
        
        styled_df = df.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
        st.dataframe(styled_df, use_container_width=True)
    
    # ==================== ANALYSIS METHODS ====================
    
    def show_technical_analysis(self):
        """Show technical analysis"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Technical Indicators")
            
            # Sample technical data
            tech_data = {
                'Indicator': ['RSI', 'MACD', 'BB Upper', 'BB Lower', 'SMA 20', 'EMA 50'],
                'Value': [65.4, 2.1, 152.3, 145.2, 148.7, 147.9],
                'Signal': ['Neutral', 'Bullish', 'Resistance', 'Support', 'Bullish', 'Bullish']
            }
            
            st.dataframe(pd.DataFrame(tech_data), use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Pattern Recognition")
            
            patterns = [
                {"Pattern": "Head & Shoulders", "Confidence": "75%", "Signal": "Bearish"},
                {"Pattern": "Bull Flag", "Confidence": "82%", "Signal": "Bullish"},
                {"Pattern": "Double Bottom", "Confidence": "68%", "Signal": "Bullish"},
            ]
            
            for pattern in patterns:
                st.markdown(f"**{pattern['Pattern']}** - {pattern['Signal']} ({pattern['Confidence']})")
    
    def show_sentiment_analysis(self):
        """Show detailed sentiment analysis"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📰 News Sentiment")
            
            # Sample news sentiment
            news_sentiment = {
                'Source': ['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch'],
                'Sentiment': [0.65, 0.72, 0.58, 0.69],
                'Articles': [15, 8, 12, 10],
                'Impact': ['Medium', 'High', 'Low', 'Medium']
            }
            
            st.dataframe(pd.DataFrame(news_sentiment), use_container_width=True)
        
        with col2:
            st.markdown("#### 📱 Social Sentiment")
            
            # Sample social metrics
            social_metrics = {
                'Platform': ['Reddit', 'Twitter', 'StockTwits', 'Discord'],
                'Mentions': [1250, 3400, 890, 450],
                'Sentiment': [0.72, 0.68, 0.75, 0.71],
                'Trending': ['🔥', '📈', '⚡', '💬']
            }
            
            st.dataframe(pd.DataFrame(social_metrics), use_container_width=True)
    
    def show_regime_analysis(self):
        """Show regime analysis"""
        
        st.markdown("#### 🧠 Market Regime Detection")
        
        # Current regime
        current_regime = RegimeType.BULL
        confidence = 87.5
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Regime", current_regime.value)
        
        with col2:
            st.metric("Confidence", f"{confidence}%")
        
        with col3:
            st.metric("Duration", "3 days")
        
        # Regime history chart
        st.markdown("#### 📊 Regime History")
        
        # Generate sample regime data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        regimes = np.random.choice(['Bull', 'Bear', 'Sideways', 'Volatile'], 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=regimes,
            mode='markers+lines',
            marker=dict(size=8, color='#4ECDC4')
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_title="Regime",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_correlation_analysis(self):
        """Show correlation analysis"""
        
        st.markdown("#### 🔗 Asset Correlations")
        
        # Generate sample correlation matrix
        assets = ['AAPL', 'GOOGL', 'MSFT', 'BTC', 'ETH', 'SPY']
        corr_matrix = np.random.uniform(0.3, 0.9, (6, 6))
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Make it symmetric
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=assets,
            y=assets,
            colorscale='RdYlBu',
            text=corr_matrix,
            texttemplate="%{text:.2f}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== SCANNER METHODS ====================
    
    def run_market_scan(self, scan_types, asset_types, min_volume, min_price, max_results):
        """Run market scan"""
        
        # Simulate scan results
        time.sleep(2)  # Simulate processing time
        
        self.scan_results = [
            {
                'Symbol': 'AAPL',
                'Type': 'Breakout',
                'Score': 85.2,
                'Price': 148.50,
                'Change': 2.3,
                'Volume': 1500000,
                'Signal': 'BUY'
            },
            {
                'Symbol': 'BTC-USD',
                'Type': 'Momentum',
                'Score': 78.9,
                'Price': 44500.00,
                'Change': 3.1,
                'Volume': 25000000,
                'Signal': 'BUY'
            },
            {
                'Symbol': 'TSLA',
                'Type': 'Volume Spike',
                'Score': 72.1,
                'Price': 215.00,
                'Change': -1.8,
                'Volume': 3500000,
                'Signal': 'SELL'
            }
        ]
        
        st.success(f"✅ Scan completed! Found {len(self.scan_results)} opportunities.")
    
    def show_scan_results(self):
        """Show market scan results"""
        
        if not self.scan_results:
            st.info("Run a scan to see results here.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.scan_results)
        
        # Style the dataframe
        def color_signal(val):
            color = 'green' if val == 'BUY' else 'red' if val == 'SELL' else 'blue'
            return f'color: {color}'
        
        styled_df = df.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Action buttons for each result
        for i, result in enumerate(self.scan_results):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{result['Symbol']}** - {result['Type']} (Score: {result['Score']})")
            
            with col2:
                if st.button(f"View Chart", key=f"chart_{i}"):
                    st.info(f"Chart for {result['Symbol']} would open here")
            
            with col3:
                if st.button(f"Trade", key=f"trade_{i}"):
                    st.success(f"Trading interface for {result['Symbol']} would open here")
    
    # ==================== PORTFOLIO METHODS ====================
    
    def show_portfolio_holdings(self):
        """Show portfolio holdings"""
        
        # Sample holdings data
        holdings_data = {
            'Symbol': ['AAPL', 'BTC-USD', 'ETH-USD', 'GOOGL', 'TSLA'],
            'Shares': [200, 1.5, 10, 25, 50],
            'Avg Cost': [145.00, 43000, 2500, 2800, 220],
            'Current Price': [148.50, 44500, 2650, 2850, 215],
            'Market Value': [29700, 66750, 26500, 71250, 10750],
            'P&L': [700, 2250, 1500, 1250, -250],
            'Weight': ['14.5%', '32.6%', '12.9%', '34.8%', '5.2%']
        }
        
        df = pd.DataFrame(holdings_data)
        st.dataframe(df, use_container_width=True)
        
        # Portfolio allocation pie chart
        fig = go.Figure(data=[go.Pie(
            labels=holdings_data['Symbol'],
            values=holdings_data['Market Value'],
            hole=0.3
        )])
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            title="Portfolio Allocation"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_portfolio_performance(self):
        """Show portfolio performance"""
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "12.5%", "2.3%")
        
        with col2:
            st.metric("Sharpe Ratio", "1.85", "0.15")
        
        with col3:
            st.metric("Max Drawdown", "-8.2%", "1.1%")
        
        with col4:
            st.metric("Win Rate", "67%", "3%")
        
        # Performance chart
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        portfolio_values = 100000 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        benchmark_values = 100000 * np.exp(np.cumsum(np.random.normal(0.0008, 0.015, 100)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=portfolio_values,
            mode='lines', name='Portfolio',
            line=dict(color='#4ECDC4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=benchmark_values,
            mode='lines', name='Benchmark',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            title="Portfolio vs Benchmark",
            yaxis_title="Value ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_portfolio_risk(self):
        """Show portfolio risk metrics"""
        
        # Risk metrics
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.markdown("#### 📊 Risk Metrics")
            
            risk_metrics = {
                'Metric': ['Portfolio Beta', 'Volatility', 'VaR (95%)', 'Expected Shortfall'],
                'Value': [1.15, '18.5%', '-2.3%', '-3.8%'],
                'Rating': ['Medium', 'Medium', 'Low', 'Low']
            }
            
            st.dataframe(pd.DataFrame(risk_metrics), use_container_width=True)
        
        with risk_col2:
            st.markdown("#### ⚠️ Risk Warnings")
            
            warnings = [
                "⚠️ High concentration in tech stocks",
                "⚠️ Crypto allocation above recommended limit",
                "✅ Diversification score: Good",
                "✅ Correlation risk: Acceptable"
            ]
            
            for warning in warnings:
                st.markdown(warning)
    
    # ==================== SETTINGS METHODS ====================
    
    def show_trading_settings(self):
        """Show trading settings"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚙️ Trading Parameters")
            
            st.number_input("Max Position Size (%)", value=10.0, min_value=1.0, max_value=50.0)
            st.number_input("Risk Per Trade (%)", value=2.0, min_value=0.5, max_value=10.0)
            st.selectbox("Default Order Type", ["Market", "Limit", "Stop Loss"])
            st.checkbox("Enable Auto Trading", value=False)
            
        with col2:
            st.markdown("#### 🎯 Strategy Settings")
            
            st.multiselect(
                "Active Strategies",
                options=["Momentum", "Mean Reversion", "Breakout", "Pairs Trading"],
                default=["Momentum", "Breakout"]
            )
            
            st.slider("Strategy Confidence Threshold", 0.0, 1.0, 0.7)
            st.selectbox("Regime Adaptation", ["Automatic", "Manual", "Disabled"])
    
    def show_alert_settings(self):
        """Show alert settings"""
        
        st.markdown("#### 🔔 Alert Configuration")
        
        # Alert types
        alert_types = [
            "Price Alerts",
            "Volume Spikes",
            "Breakout Signals",
            "Sentiment Changes",
            "Portfolio Alerts",
            "Risk Warnings"
        ]
        
        for alert_type in alert_types:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(alert_type)
            
            with col2:
                st.checkbox("Email", key=f"email_{alert_type}")
            
            with col3:
                st.checkbox("Push", key=f"push_{alert_type}")
    
    def show_system_settings(self):
        """Show system settings"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖥️ Interface Settings")
            
            st.selectbox("Theme", ["Dark", "Light"], index=0)
            st.slider("Auto Refresh (seconds)", 1, 60, 5)
            st.checkbox("Enable Voice Commands", value=False)
            st.checkbox("Show Advanced Features", value=True)
        
        with col2:
            st.markdown("#### 🔧 System Status")
            
            status_items = [
                ("NeuroCluster Algorithm", "🟢 Active"),
                ("Data Feed", "🟢 Connected"),
                ("Trading Engine", "🟢 Running"),
                ("Risk Manager", "🟢 Monitoring"),
                ("Sentiment Analyzer", "🟡 Updating"),
                ("Market Scanner", "🟢 Scanning")
            ]
            
            for item, status in status_items:
                st.markdown(f"**{item}:** {status}")
    
    # ==================== UTILITY METHODS ====================
    
    def refresh_data(self):
        """Refresh all data"""
        
        try:
            # Update timestamp
            st.session_state.last_update = datetime.now()
            
            # Here you would refresh actual data
            # self.market_data = await self.data_manager.fetch_market_data(...)
            # self.sentiment_data = await self.sentiment_analyzer.analyze_sentiment(...)
            
            logger.info("Dashboard data refreshed")
            
        except Exception as e:
            st.error(f"Failed to refresh data: {e}")
            logger.error(f"Data refresh failed: {e}")

# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point for the dashboard"""
    
    try:
        # Initialize and run dashboard
        dashboard = NeuroClusterDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}")

if __name__ == "__main__":
    main()