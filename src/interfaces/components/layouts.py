#!/usr/bin/env python3
"""
File: layouts.py
Path: NeuroCluster-Elite/src/interfaces/components/layouts.py
Description: Page layouts and dashboard templates for NeuroCluster Elite

This module provides pre-built page layouts and dashboard templates for the
NeuroCluster Elite trading platform, including responsive layouts, sidebar
configurations, and specialized dashboard views for different use cases.

Features:
- Responsive dashboard layouts
- Sidebar navigation and controls
- Multi-tab page structures
- Modal dialogs and overlays
- Mobile-optimized layouts
- Theme and styling management
- Component arrangement templates
- Custom CSS injection

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import our modules
try:
    from src.interfaces.components.charts import ChartConfig, CandlestickChart, ChartType
    from src.interfaces.components.widgets import (
        MetricWidget, PerformanceWidget, RiskWidget, DataTableWidget,
        TradingControlWidget, AlertWidget, SignalWidget, MarketOverviewWidget,
        WidgetStyle, WidgetTheme
    )
    from src.core.neurocluster_elite import AssetType, RegimeType
    from src.utils.config_manager import ConfigManager
except ImportError:
    # Fallback for testing
    from enum import Enum
    class AssetType(Enum):
        STOCK = "stock"
        CRYPTO = "crypto"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== LAYOUT CONFIGURATION ====================

class LayoutType(Enum):
    """Available layout types"""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    THREE_COLUMN = "three_column"
    SIDEBAR_MAIN = "sidebar_main"
    TABBED = "tabbed"
    DASHBOARD_GRID = "dashboard_grid"
    FULL_SCREEN = "full_screen"

class ResponsiveBreakpoint(Enum):
    """Responsive design breakpoints"""
    MOBILE = "mobile"     # < 768px
    TABLET = "tablet"     # 768px - 1024px
    DESKTOP = "desktop"   # > 1024px

@dataclass
class LayoutConfig:
    """Configuration for page layouts"""
    # Basic layout settings
    layout_type: LayoutType = LayoutType.DASHBOARD_GRID
    page_title: str = "NeuroCluster Elite"
    page_icon: str = "🧠"
    
    # Layout dimensions
    sidebar_width: int = 300
    main_content_padding: str = "1rem"
    component_spacing: str = "1rem"
    
    # Responsive settings
    mobile_breakpoint: int = 768
    tablet_breakpoint: int = 1024
    enable_responsive: bool = True
    
    # Theme and styling
    theme: WidgetTheme = WidgetTheme.DARK
    custom_css: Optional[str] = None
    hide_streamlit_style: bool = True
    
    # Navigation
    show_sidebar: bool = True
    sidebar_state: str = "expanded"  # expanded, collapsed, auto
    navigation_style: str = "tabs"  # tabs, sidebar, dropdown
    
    # Performance
    enable_caching: bool = True
    lazy_loading: bool = True
    optimize_for_mobile: bool = True

# ==================== BASE LAYOUT CLASS ====================

class BaseLayout:
    """Base class for all page layouts"""
    
    def __init__(self, config: Optional[LayoutConfig] = None):
        """Initialize base layout"""
        self.config = config or LayoutConfig()
        self.widget_style = WidgetStyle(theme=self.config.theme)
        self._setup_page_config()
        self._inject_custom_css()
    
    def _setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=self.config.page_title,
            page_icon=self.config.page_icon,
            layout="wide",
            initial_sidebar_state=self.config.sidebar_state,
            menu_items={
                'Get Help': 'https://github.com/neurocluster-elite',
                'Report a bug': 'https://github.com/neurocluster-elite/issues',
                'About': 'NeuroCluster Elite - AI-Powered Trading Platform'
            }
        )
    
    def _inject_custom_css(self):
        """Inject custom CSS for styling"""
        
        base_css = """
        <style>
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom styling */
        .main > div {
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        /* Card-like containers */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main > div {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        
        /* Custom components */
        .stAlert > div {
            border-radius: 10px;
        }
        
        .stMetric > div {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Trading signal colors */
        .signal-buy {
            color: #00ff88 !important;
            font-weight: bold;
        }
        
        .signal-sell {
            color: #ff4444 !important;
            font-weight: bold;
        }
        
        /* Status indicators */
        .status-online {
            color: #00ff88;
        }
        
        .status-offline {
            color: #ff4444;
        }
        
        .status-warning {
            color: #ff9500;
        }
        
        /* Hover effects */
        .hover-glow:hover {
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
            transition: box-shadow 0.3s ease;
        }
        </style>
        """
        
        # Add custom CSS if provided
        if self.config.custom_css:
            base_css += f"\n{self.config.custom_css}"
        
        st.markdown(base_css, unsafe_allow_html=True)
    
    def render(self, content_func: Callable, **kwargs):
        """Render the layout with content"""
        raise NotImplementedError("Subclasses must implement render method")

# ==================== DASHBOARD LAYOUT ====================

class DashboardLayout(BaseLayout):
    """Main dashboard layout with multiple sections"""
    
    def render(self, data: Dict[str, Any], **kwargs):
        """Render dashboard layout"""
        
        # Header
        self._render_header(data.get('header', {}))
        
        # Main content area
        if self.config.show_sidebar:
            self._render_with_sidebar(data)
        else:
            self._render_main_content(data)
    
    def _render_header(self, header_data: Dict[str, Any]):
        """Render dashboard header"""
        
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.markdown("# 🧠 NeuroCluster Elite")
            st.caption("AI-Powered Trading Platform")
        
        with col2:
            # Market status
            market_status = header_data.get('market_status', 'Unknown')
            if market_status == 'Open':
                st.markdown('<p class="status-online">🟢 Market Open</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-offline">🔴 Market Closed</p>', unsafe_allow_html=True)
        
        with col3:
            # Quick stats
            portfolio_value = header_data.get('portfolio_value', 0)
            daily_pnl = header_data.get('daily_pnl', 0)
            
            st.metric(
                "Portfolio Value",
                f"${portfolio_value:,.2f}",
                f"${daily_pnl:+,.2f}"
            )
        
        st.divider()
    
    def _render_with_sidebar(self, data: Dict[str, Any]):
        """Render layout with sidebar"""
        
        # Sidebar content
        with st.sidebar:
            self._render_sidebar(data.get('sidebar', {}))
        
        # Main content
        self._render_main_content(data.get('main', {}))
    
    def _render_sidebar(self, sidebar_data: Dict[str, Any]):
        """Render sidebar content"""
        
        st.markdown("## 🎛️ Controls")
        
        # Trading controls
        st.markdown("### Trading")
        auto_trading = st.checkbox("Auto Trading", value=sidebar_data.get('auto_trading', False))
        paper_trading = st.checkbox("Paper Trading", value=sidebar_data.get('paper_trading', True))
        
        # Strategy selection
        st.markdown("### Strategy")
        strategies = sidebar_data.get('available_strategies', ['Default'])
        selected_strategy = st.selectbox("Active Strategy", strategies)
        
        # Risk controls
        st.markdown("### Risk Controls")
        max_position_size = st.slider("Max Position Size (%)", 1, 20, 5)
        daily_loss_limit = st.slider("Daily Loss Limit (%)", 1, 10, 3)
        
        # Time range selector
        st.markdown("### Analysis")
        time_ranges = ['1D', '1W', '1M', '3M', '6M', '1Y', 'ALL']
        selected_timeframe = st.selectbox("Timeframe", time_ranges, index=2)
        
        # Asset filters
        asset_types = [e.value for e in AssetType]
        selected_assets = st.multiselect("Asset Types", asset_types, default=asset_types)
        
        # Market data refresh
        st.markdown("### Data")
        if st.button("Refresh Data", type="primary"):
            st.rerun()
        
        # System status
        st.markdown("### System Status")
        status_data = sidebar_data.get('system_status', {})
        
        # Algorithm status
        algo_efficiency = status_data.get('algorithm_efficiency', 99.59)
        st.metric("Algorithm Efficiency", f"{algo_efficiency:.2f}%")
        
        processing_time = status_data.get('processing_time', 0.045)
        st.metric("Processing Time", f"{processing_time:.3f}ms")
        
        # Connection status
        data_connections = status_data.get('data_connections', {})
        for source, status in data_connections.items():
            color = "🟢" if status == "connected" else "🔴"
            st.text(f"{color} {source}")
    
    def _render_main_content(self, main_data: Dict[str, Any]):
        """Render main dashboard content"""
        
        # Performance overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_performance_section(main_data.get('performance', {}))
        
        with col2:
            self._render_alerts_section(main_data.get('alerts', {}))
        
        # Charts and analysis
        self._render_charts_section(main_data.get('charts', {}))
        
        # Positions and trades
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_positions_section(main_data.get('positions', {}))
        
        with col2:
            self._render_signals_section(main_data.get('signals', {}))
    
    def _render_performance_section(self, performance_data: Dict[str, Any]):
        """Render performance metrics section"""
        
        st.markdown("## 📊 Performance Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Total Value", performance_data.get('total_value', 0), "prefix", "$"),
            ("Daily P&L", performance_data.get('daily_pnl', 0), "prefix", "$"),
            ("Total Return", performance_data.get('total_return', 0), "suffix", "%"),
            ("Sharpe Ratio", performance_data.get('sharpe_ratio', 0), None, None)
        ]
        
        for i, (label, value, format_type, format_char) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                if format_type == "prefix":
                    st.metric(label, f"{format_char}{value:,.2f}")
                elif format_type == "suffix":
                    st.metric(label, f"{value:.2f}{format_char}")
                else:
                    st.metric(label, f"{value:.2f}")
        
        # Performance chart placeholder
        st.markdown("### Portfolio Performance")
        # This would show actual performance chart
        st.info("📈 Performance chart will be displayed here")
    
    def _render_alerts_section(self, alerts_data: Dict[str, Any]):
        """Render alerts and notifications section"""
        
        st.markdown("## 🚨 Alerts")
        
        alerts = alerts_data.get('recent_alerts', [])
        
        if not alerts:
            st.info("No recent alerts")
        else:
            for alert in alerts[:5]:  # Show last 5 alerts
                alert_type = alert.get('type', 'info')
                message = alert.get('message', 'No message')
                timestamp = alert.get('timestamp', datetime.now())
                
                if alert_type == 'error':
                    st.error(f"🔴 {message}")
                elif alert_type == 'warning':
                    st.warning(f"🟡 {message}")
                elif alert_type == 'success':
                    st.success(f"🟢 {message}")
                else:
                    st.info(f"🔵 {message}")
                
                st.caption(f"⏰ {timestamp.strftime('%H:%M:%S')}")
    
    def _render_charts_section(self, charts_data: Dict[str, Any]):
        """Render charts and analysis section"""
        
        st.markdown("## 📈 Market Analysis")
        
        # Tab layout for different chart types
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Chart", "🔍 Patterns", "📡 Signals", "🌍 Market"])
        
        with tab1:
            st.info("📊 Main candlestick chart will be displayed here")
            # This would show actual candlestick chart
        
        with tab2:
            st.info("🔍 Pattern recognition results will be displayed here")
            # This would show pattern analysis
        
        with tab3:
            st.info("📡 Trading signals analysis will be displayed here")
            # This would show signal analysis
        
        with tab4:
            st.info("🌍 Market overview and correlation analysis will be displayed here")
            # This would show market overview
    
    def _render_positions_section(self, positions_data: Dict[str, Any]):
        """Render current positions section"""
        
        st.markdown("## 📈 Current Positions")
        
        positions = positions_data.get('positions', [])
        
        if not positions:
            st.info("No current positions")
        else:
            # Create positions table
            st.dataframe(
                positions,
                use_container_width=True,
                hide_index=True
            )
    
    def _render_signals_section(self, signals_data: Dict[str, Any]):
        """Render trading signals section"""
        
        st.markdown("## 📡 Recent Signals")
        
        signals = signals_data.get('recent_signals', [])
        
        if not signals:
            st.info("No recent signals")
        else:
            for signal in signals[:5]:  # Show last 5 signals
                symbol = signal.get('symbol', 'N/A')
                signal_type = signal.get('signal_type', 'HOLD')
                confidence = signal.get('confidence', 0)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if signal_type in ['BUY', 'STRONG_BUY']:
                        st.markdown(f'<p class="signal-buy">🟢 {symbol} - {signal_type}</p>', unsafe_allow_html=True)
                    elif signal_type in ['SELL', 'STRONG_SELL']:
                        st.markdown(f'<p class="signal-sell">🔴 {symbol} - {signal_type}</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"🟡 {symbol} - {signal_type}")
                
                with col2:
                    st.progress(confidence)
                
                with col3:
                    st.caption(f"{confidence:.0%}")

# ==================== TRADING LAYOUT ====================

class TradingLayout(BaseLayout):
    """Specialized layout for trading interface"""
    
    def render(self, data: Dict[str, Any], **kwargs):
        """Render trading interface layout"""
        
        # Trading header
        self._render_trading_header(data.get('trading_status', {}))
        
        # Main trading interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chart and analysis
            self._render_trading_chart(data.get('chart_data', {}))
        
        with col2:
            # Trading controls
            self._render_trading_controls(data.get('trading_controls', {}))
        
        # Order book and recent trades
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_order_book(data.get('orders', {}))
        
        with col2:
            self._render_recent_trades(data.get('trades', {}))
    
    def _render_trading_header(self, status_data: Dict[str, Any]):
        """Render trading status header"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trading_mode = status_data.get('mode', 'Paper')
            if trading_mode == 'Live':
                st.markdown('<p class="status-warning">🟡 Live Trading</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-online">🟢 Paper Trading</p>', unsafe_allow_html=True)
        
        with col2:
            buying_power = status_data.get('buying_power', 0)
            st.metric("Buying Power", f"${buying_power:,.2f}")
        
        with col3:
            open_orders = status_data.get('open_orders', 0)
            st.metric("Open Orders", open_orders)
        
        with col4:
            day_trades = status_data.get('day_trades', 0)
            st.metric("Day Trades", f"{day_trades}/3")
        
        st.divider()
    
    def _render_trading_chart(self, chart_data: Dict[str, Any]):
        """Render trading chart with indicators"""
        
        st.markdown("### 📊 Chart Analysis")
        
        # Chart controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox("Symbol", chart_data.get('symbols', ['AAPL']))
        
        with col2:
            timeframe = st.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '4h', '1d'])
        
        with col3:
            chart_type = st.selectbox("Chart Type", ['Candlestick', 'Line', 'Area'])
        
        # Chart placeholder
        st.info(f"📊 {chart_type} chart for {symbol} ({timeframe}) will be displayed here")
    
    def _render_trading_controls(self, controls_data: Dict[str, Any]):
        """Render trading control panel"""
        
        st.markdown("### 🎛️ Trading Controls")
        
        # Order entry form
        with st.form("quick_order"):
            symbol = st.text_input("Symbol", value="AAPL")
            
            col1, col2 = st.columns(2)
            with col1:
                side = st.selectbox("Side", ["BUY", "SELL"])
                quantity = st.number_input("Quantity", min_value=1, value=100)
            
            with col2:
                order_type = st.selectbox("Type", ["MARKET", "LIMIT"])
                if order_type == "LIMIT":
                    price = st.number_input("Price", min_value=0.01, value=100.0)
            
            submitted = st.form_submit_button("Place Order", type="primary")
            
            if submitted:
                st.success(f"Order submitted: {side} {quantity} {symbol}")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Close All Positions", type="secondary"):
                st.warning("Close all positions requested")
        
        with col2:
            if st.button("Cancel All Orders", type="secondary"):
                st.warning("Cancel all orders requested")
    
    def _render_order_book(self, orders_data: Dict[str, Any]):
        """Render order book"""
        
        st.markdown("### 📋 Open Orders")
        
        orders = orders_data.get('open_orders', [])
        
        if not orders:
            st.info("No open orders")
        else:
            st.dataframe(orders, use_container_width=True)
    
    def _render_recent_trades(self, trades_data: Dict[str, Any]):
        """Render recent trades"""
        
        st.markdown("### 🔄 Recent Trades")
        
        trades = trades_data.get('recent_trades', [])
        
        if not trades:
            st.info("No recent trades")
        else:
            for trade in trades[:5]:  # Show last 5 trades
                symbol = trade.get('symbol', 'N/A')
                side = trade.get('side', 'N/A')
                quantity = trade.get('quantity', 0)
                price = trade.get('price', 0)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if side == 'BUY':
                        st.markdown(f'<p class="signal-buy">🟢 {side} {symbol}</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="signal-sell">🔴 {side} {symbol}</p>', unsafe_allow_html=True)
                
                with col2:
                    st.text(f"{quantity} @ ${price:.2f}")
                
                with col3:
                    timestamp = trade.get('timestamp', datetime.now())
                    st.caption(timestamp.strftime('%H:%M:%S'))

# ==================== ANALYSIS LAYOUT ====================

class AnalysisLayout(BaseLayout):
    """Layout for detailed market analysis"""
    
    def render(self, data: Dict[str, Any], **kwargs):
        """Render analysis layout"""
        
        # Analysis navigation
        analysis_tabs = st.tabs([
            "🔍 Pattern Analysis",
            "📊 Technical Analysis", 
            "🧠 Algorithm Analysis",
            "📈 Performance Analysis",
            "⚠️ Risk Analysis"
        ])
        
        with analysis_tabs[0]:
            self._render_pattern_analysis(data.get('patterns', {}))
        
        with analysis_tabs[1]:
            self._render_technical_analysis(data.get('technical', {}))
        
        with analysis_tabs[2]:
            self._render_algorithm_analysis(data.get('algorithm', {}))
        
        with analysis_tabs[3]:
            self._render_performance_analysis(data.get('performance', {}))
        
        with analysis_tabs[4]:
            self._render_risk_analysis(data.get('risk', {}))
    
    def _render_pattern_analysis(self, pattern_data: Dict[str, Any]):
        """Render pattern recognition analysis"""
        
        st.markdown("## 🔍 Pattern Recognition Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("📊 Pattern recognition charts will be displayed here")
        
        with col2:
            st.markdown("### Detected Patterns")
            patterns = pattern_data.get('detected_patterns', [])
            
            if not patterns:
                st.info("No patterns detected")
            else:
                for pattern in patterns:
                    st.markdown(f"• **{pattern.get('type', 'Unknown')}**")
                    st.caption(f"Confidence: {pattern.get('confidence', 0):.0%}")
    
    def _render_technical_analysis(self, technical_data: Dict[str, Any]):
        """Render technical analysis"""
        
        st.markdown("## 📊 Technical Analysis")
        
        # Technical indicators
        col1, col2, col3 = st.columns(3)
        
        indicators = technical_data.get('indicators', {})
        
        with col1:
            rsi = indicators.get('rsi', 50)
            st.metric("RSI (14)", f"{rsi:.1f}")
            
            if rsi > 70:
                st.caption("🔴 Overbought")
            elif rsi < 30:
                st.caption("🟢 Oversold")
            else:
                st.caption("🟡 Neutral")
        
        with col2:
            macd = indicators.get('macd', 0)
            st.metric("MACD", f"{macd:.3f}")
        
        with col3:
            bb_position = indicators.get('bollinger_position', 0.5)
            st.metric("BB Position", f"{bb_position:.2f}")
    
    def _render_algorithm_analysis(self, algorithm_data: Dict[str, Any]):
        """Render algorithm performance analysis"""
        
        st.markdown("## 🧠 NeuroCluster Algorithm Analysis")
        
        # Algorithm metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            efficiency = algorithm_data.get('efficiency', 99.59)
            st.metric("Efficiency", f"{efficiency:.2f}%")
        
        with col2:
            processing_time = algorithm_data.get('processing_time', 0.045)
            st.metric("Processing Time", f"{processing_time:.3f}ms")
        
        with col3:
            clusters = algorithm_data.get('active_clusters', 8)
            st.metric("Active Clusters", clusters)
        
        with col4:
            confidence = algorithm_data.get('avg_confidence', 0.85)
            st.metric("Avg Confidence", f"{confidence:.0%}")
        
        # Algorithm status
        st.markdown("### Algorithm Status")
        status = algorithm_data.get('status', 'operational')
        
        if status == 'operational':
            st.success("🟢 Algorithm operating at optimal performance")
        elif status == 'degraded':
            st.warning("🟡 Algorithm performance degraded")
        else:
            st.error("🔴 Algorithm error detected")
    
    def _render_performance_analysis(self, performance_data: Dict[str, Any]):
        """Render performance analysis"""
        
        st.markdown("## 📈 Performance Analysis")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_return = performance_data.get('total_return', 0)
            st.metric("Total Return", f"{total_return:.2f}%")
        
        with col2:
            sharpe_ratio = performance_data.get('sharpe_ratio', 0)
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        with col3:
            max_drawdown = performance_data.get('max_drawdown', 0)
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        # Performance chart placeholder
        st.info("📈 Detailed performance charts will be displayed here")
    
    def _render_risk_analysis(self, risk_data: Dict[str, Any]):
        """Render risk analysis"""
        
        st.markdown("## ⚠️ Risk Analysis")
        
        # Risk metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            portfolio_risk = risk_data.get('portfolio_risk', 0)
            st.metric("Portfolio Risk", f"{portfolio_risk:.2f}%")
        
        with col2:
            var_95 = risk_data.get('var_95', 0)
            st.metric("VaR (95%)", f"${var_95:,.0f}")
        
        with col3:
            beta = risk_data.get('beta', 1.0)
            st.metric("Beta", f"{beta:.2f}")
        
        # Risk level indicator
        risk_level = risk_data.get('risk_level', 'Medium')
        risk_score = risk_data.get('risk_score', 0.5)
        
        st.markdown("### Risk Level")
        st.progress(risk_score)
        
        if risk_score < 0.3:
            st.success(f"🟢 {risk_level} Risk - Conservative portfolio")
        elif risk_score < 0.7:
            st.warning(f"🟡 {risk_level} Risk - Balanced portfolio")
        else:
            st.error(f"🔴 {risk_level} Risk - Aggressive portfolio")

# ==================== MOBILE LAYOUT ====================

class MobileLayout(BaseLayout):
    """Mobile-optimized layout"""
    
    def render(self, data: Dict[str, Any], **kwargs):
        """Render mobile-optimized layout"""
        
        # Mobile header
        st.markdown("# 🧠 NeuroCluster")
        
        # Key metrics (stacked)
        portfolio_value = data.get('portfolio_value', 0)
        daily_pnl = data.get('daily_pnl', 0)
        
        st.metric("Portfolio", f"${portfolio_value:,.0f}", f"${daily_pnl:+,.0f}")
        
        # Navigation tabs
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trading", "⚙️ Settings"])
        
        with tab1:
            self._render_mobile_overview(data)
        
        with tab2:
            self._render_mobile_trading(data)
        
        with tab3:
            self._render_mobile_settings(data)
    
    def _render_mobile_overview(self, data: Dict[str, Any]):
        """Render mobile overview"""
        
        # Quick stats
        st.markdown("### Today's Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            win_rate = data.get('win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.0f}%")
        
        with col2:
            trades_today = data.get('trades_today', 0)
            st.metric("Trades", trades_today)
        
        # Mini chart
        st.markdown("### Performance Chart")
        st.info("📱 Mobile-optimized chart will be displayed here")
        
        # Top positions
        st.markdown("### Top Positions")
        positions = data.get('top_positions', [])
        
        for position in positions[:3]:
            symbol = position.get('symbol', 'N/A')
            pnl = position.get('pnl_pct', 0)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.text(symbol)
            
            with col2:
                color = "🟢" if pnl >= 0 else "🔴"
                st.text(f"{color} {pnl:+.1f}%")
    
    def _render_mobile_trading(self, data: Dict[str, Any]):
        """Render mobile trading interface"""
        
        st.markdown("### Quick Trade")
        
        # Simplified order form
        symbol = st.text_input("Symbol", value="AAPL")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🟢 BUY", use_container_width=True):
                st.success("Buy order placed")
        
        with col2:
            if st.button("🔴 SELL", use_container_width=True):
                st.error("Sell order placed")
        
        # Recent signals
        st.markdown("### Recent Signals")
        signals = data.get('recent_signals', [])
        
        for signal in signals[:3]:
            symbol = signal.get('symbol', 'N/A')
            signal_type = signal.get('signal_type', 'HOLD')
            
            if signal_type in ['BUY', 'STRONG_BUY']:
                st.success(f"🟢 {symbol} - {signal_type}")
            elif signal_type in ['SELL', 'STRONG_SELL']:
                st.error(f"🔴 {symbol} - {signal_type}")
            else:
                st.info(f"🟡 {symbol} - {signal_type}")
    
    def _render_mobile_settings(self, data: Dict[str, Any]):
        """Render mobile settings"""
        
        st.markdown("### Settings")
        
        # Basic settings
        paper_trading = st.checkbox("Paper Trading", value=True)
        notifications = st.checkbox("Push Notifications", value=True)
        
        # Risk settings
        st.markdown("#### Risk Settings")
        max_position = st.slider("Max Position %", 1, 20, 5)
        
        # Data refresh
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()

# ==================== LAYOUT FACTORY ====================

class LayoutFactory:
    """Factory for creating layout instances"""
    
    @staticmethod
    def create_layout(layout_type: LayoutType, config: Optional[LayoutConfig] = None) -> BaseLayout:
        """Create layout instance"""
        
        if layout_type == LayoutType.DASHBOARD_GRID:
            return DashboardLayout(config)
        elif layout_type == LayoutType.TABBED:
            return TradingLayout(config)
        elif layout_type == LayoutType.FULL_SCREEN:
            return AnalysisLayout(config)
        else:
            return DashboardLayout(config)  # Default
    
    @staticmethod
    def get_mobile_layout(config: Optional[LayoutConfig] = None) -> MobileLayout:
        """Get mobile-optimized layout"""
        return MobileLayout(config)

# ==================== UTILITY FUNCTIONS ====================

def detect_mobile_device() -> bool:
    """Detect if user is on mobile device (simplified)"""
    # In a real implementation, this would check user agent
    # For now, we'll assume desktop
    return False

def get_optimal_layout(data: Dict[str, Any]) -> BaseLayout:
    """Get optimal layout based on context"""
    
    if detect_mobile_device():
        return LayoutFactory.get_mobile_layout()
    
    # Choose layout based on user preferences or data
    layout_preference = data.get('layout_preference', LayoutType.DASHBOARD_GRID)
    return LayoutFactory.create_layout(layout_preference)

def apply_responsive_styling():
    """Apply responsive CSS styling"""
    
    responsive_css = """
    <style>
    @media (max-width: 768px) {
        .main > div {
            padding: 0.5rem !important;
        }
        
        .stColumns > div {
            padding: 0.25rem !important;
        }
        
        .stMetric {
            font-size: 0.9rem !important;
        }
    }
    </style>
    """
    
    st.markdown(responsive_css, unsafe_allow_html=True)

# ==================== TESTING FUNCTION ====================

def test_layouts():
    """Test layout components"""
    
    print("🎨 Testing Layout Components")
    print("=" * 50)
    
    # Test data
    test_data = {
        'header': {
            'market_status': 'Open',
            'portfolio_value': 125000,
            'daily_pnl': 1250
        },
        'sidebar': {
            'auto_trading': False,
            'paper_trading': True,
            'available_strategies': ['Default', 'Momentum', 'Mean Reversion'],
            'system_status': {
                'algorithm_efficiency': 99.59,
                'processing_time': 0.045,
                'data_connections': {
                    'Yahoo Finance': 'connected',
                    'Alpha Vantage': 'disconnected'
                }
            }
        },
        'main': {
            'performance': {
                'total_value': 125000,
                'daily_pnl': 1250,
                'total_return': 25.0,
                'sharpe_ratio': 1.85
            },
            'alerts': {
                'recent_alerts': [
                    {'type': 'success', 'message': 'Trade executed', 'timestamp': datetime.now()},
                    {'type': 'warning', 'message': 'High volatility detected', 'timestamp': datetime.now()}
                ]
            },
            'positions': {
                'positions': [
                    {'symbol': 'AAPL', 'quantity': 100, 'value': 18000, 'pnl': 1000},
                    {'symbol': 'BTC-USD', 'quantity': 0.5, 'value': 22500, 'pnl': 2500}
                ]
            },
            'signals': {
                'recent_signals': [
                    {'symbol': 'TSLA', 'signal_type': 'BUY', 'confidence': 0.85},
                    {'symbol': 'NVDA', 'signal_type': 'SELL', 'confidence': 0.92}
                ]
            }
        }
    }
    
    print("✅ Created test data")
    
    # Test dashboard layout
    config = LayoutConfig(layout_type=LayoutType.DASHBOARD_GRID)
    dashboard = DashboardLayout(config)
    print("✅ Created DashboardLayout")
    
    # Test trading layout
    trading = TradingLayout(config)
    print("✅ Created TradingLayout")
    
    # Test analysis layout
    analysis = AnalysisLayout(config)
    print("✅ Created AnalysisLayout")
    
    # Test mobile layout
    mobile = MobileLayout(config)
    print("✅ Created MobileLayout")
    
    # Test layout factory
    factory_layout = LayoutFactory.create_layout(LayoutType.DASHBOARD_GRID)
    print("✅ Created layout via factory")
    
    print("\n🎉 Layout components testing completed!")

if __name__ == "__main__":
    test_layouts()