#!/usr/bin/env python3
"""
File: widgets.py
Path: NeuroCluster-Elite/src/interfaces/components/widgets.py
Description: Reusable UI widgets for NeuroCluster Elite dashboard

This module provides a comprehensive collection of reusable UI widgets
designed specifically for the NeuroCluster Elite trading platform,
including metrics displays, controls, data tables, and interactive elements.

Features:
- Financial metrics display widgets
- Portfolio overview components
- Trading controls and forms
- Data tables with sorting and filtering
- Alert and notification widgets
- Performance indicators
- Risk metrics displays
- Interactive controls for strategies

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData, RegimeType
    from src.trading.strategies.base_strategy import TradingSignal, SignalType, StrategyMetrics
    from src.trading.portfolio_manager import Portfolio
    from src.utils.helpers import format_currency, format_percentage, calculate_sharpe_ratio
except ImportError:
    # Fallback for testing
    from enum import Enum
    class AssetType(Enum):
        STOCK = "stock"
        CRYPTO = "crypto"
        FOREX = "forex"
        COMMODITY = "commodity"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== WIDGET STYLES ====================

class WidgetTheme(Enum):
    """Widget themes"""
    DARK = "dark"
    LIGHT = "light"
    PROFESSIONAL = "professional"

@dataclass
class WidgetStyle:
    """Widget styling configuration"""
    theme: WidgetTheme = WidgetTheme.DARK
    primary_color: str = "#00ff88"
    secondary_color: str = "#00bfff"
    danger_color: str = "#ff4444"
    warning_color: str = "#ff9500"
    success_color: str = "#00ff88"
    background_color: str = "rgba(0,0,0,0.1)"
    border_radius: str = "10px"
    box_shadow: str = "0 2px 4px rgba(0,0,0,0.1)"

# ==================== METRIC WIDGETS ====================

class MetricWidget:
    """Reusable metric display widget"""
    
    def __init__(self, style: Optional[WidgetStyle] = None):
        self.style = style or WidgetStyle()
    
    def display_metric(self, label: str, value: Union[float, int, str], 
                      delta: Optional[Union[float, str]] = None,
                      delta_color: Optional[str] = None,
                      help_text: Optional[str] = None,
                      prefix: str = "", suffix: str = ""):
        """Display a single metric with optional delta"""
        
        # Format value
        if isinstance(value, float):
            if abs(value) >= 1e9:
                formatted_value = f"{prefix}{value/1e9:.2f}B{suffix}"
            elif abs(value) >= 1e6:
                formatted_value = f"{prefix}{value/1e6:.2f}M{suffix}"
            elif abs(value) >= 1e3:
                formatted_value = f"{prefix}{value/1e3:.2f}K{suffix}"
            else:
                formatted_value = f"{prefix}{value:.2f}{suffix}"
        else:
            formatted_value = f"{prefix}{value}{suffix}"
        
        # Format delta
        if delta is not None:
            if isinstance(delta, float):
                delta_formatted = f"{delta:+.2f}"
                if delta_color is None:
                    delta_color = self.style.success_color if delta >= 0 else self.style.danger_color
            else:
                delta_formatted = str(delta)
                delta_color = delta_color or self.style.secondary_color
        else:
            delta_formatted = None
        
        # Display metric
        st.metric(
            label=label,
            value=formatted_value,
            delta=delta_formatted,
            help=help_text
        )
    
    def display_metric_grid(self, metrics: List[Dict[str, Any]], columns: int = 3):
        """Display multiple metrics in a grid layout"""
        
        cols = st.columns(columns)
        
        for i, metric in enumerate(metrics):
            with cols[i % columns]:
                self.display_metric(**metric)

class PerformanceWidget:
    """Performance metrics widget"""
    
    def __init__(self, style: Optional[WidgetStyle] = None):
        self.style = style or WidgetStyle()
        self.metric_widget = MetricWidget(style)
    
    def display_portfolio_performance(self, portfolio_data: Dict[str, Any]):
        """Display portfolio performance metrics"""
        
        st.subheader("📊 Portfolio Performance")
        
        # Main performance metrics
        performance_metrics = [
            {
                "label": "Total Value",
                "value": portfolio_data.get('total_value', 0),
                "prefix": "$",
                "delta": portfolio_data.get('daily_change', 0),
                "help_text": "Current total portfolio value"
            },
            {
                "label": "Daily P&L",
                "value": portfolio_data.get('daily_pnl', 0),
                "prefix": "$",
                "delta": portfolio_data.get('daily_pnl_pct', 0),
                "suffix": "%",
                "help_text": "Today's profit and loss"
            },
            {
                "label": "Total Return",
                "value": portfolio_data.get('total_return_pct', 0),
                "suffix": "%",
                "help_text": "Total return since inception"
            }
        ]
        
        self.metric_widget.display_metric_grid(performance_metrics, columns=3)
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.metric_widget.display_metric(
                "Sharpe Ratio",
                portfolio_data.get('sharpe_ratio', 0),
                help_text="Risk-adjusted return measure"
            )
        
        with col2:
            self.metric_widget.display_metric(
                "Max Drawdown",
                portfolio_data.get('max_drawdown', 0),
                suffix="%",
                help_text="Maximum peak-to-trough decline"
            )
        
        with col3:
            self.metric_widget.display_metric(
                "Win Rate",
                portfolio_data.get('win_rate', 0),
                suffix="%",
                help_text="Percentage of profitable trades"
            )

class RiskWidget:
    """Risk metrics display widget"""
    
    def __init__(self, style: Optional[WidgetStyle] = None):
        self.style = style or WidgetStyle()
        self.metric_widget = MetricWidget(style)
    
    def display_risk_metrics(self, risk_data: Dict[str, Any]):
        """Display risk management metrics"""
        
        st.subheader("⚠️ Risk Metrics")
        
        # Risk level indicator
        risk_level = risk_data.get('risk_level', 'Medium')
        risk_score = risk_data.get('risk_score', 0.5)
        
        # Color code risk level
        if risk_score < 0.3:
            risk_color = self.style.success_color
        elif risk_score < 0.7:
            risk_color = self.style.warning_color
        else:
            risk_color = self.style.danger_color
        
        # Display risk level with progress bar
        st.markdown(f"**Risk Level:** :{'red' if risk_score > 0.7 else 'orange' if risk_score > 0.3 else 'green'}[{risk_level}]")
        st.progress(risk_score)
        
        # Risk metrics grid
        risk_metrics = [
            {
                "label": "Portfolio Risk",
                "value": risk_data.get('portfolio_risk', 0),
                "suffix": "%",
                "help_text": "Current portfolio risk level"
            },
            {
                "label": "VaR (95%)",
                "value": risk_data.get('var_95', 0),
                "prefix": "$",
                "help_text": "Value at Risk (95% confidence)"
            },
            {
                "label": "Beta",
                "value": risk_data.get('beta', 1.0),
                "help_text": "Portfolio beta vs benchmark"
            }
        ]
        
        self.metric_widget.display_metric_grid(risk_metrics, columns=3)

# ==================== DATA TABLE WIDGETS ====================

class DataTableWidget:
    """Enhanced data table widget with filtering and sorting"""
    
    def __init__(self, style: Optional[WidgetStyle] = None):
        self.style = style or WidgetStyle()
    
    def display_positions_table(self, positions: List[Dict[str, Any]]):
        """Display current positions table"""
        
        if not positions:
            st.info("No positions found")
            return
        
        st.subheader("📈 Current Positions")
        
        # Convert to DataFrame
        df = pd.DataFrame(positions)
        
        # Format DataFrame for display
        display_df = df.copy()
        
        # Format currency columns
        currency_columns = ['market_value', 'cost_basis', 'unrealized_pnl']
        for col in currency_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
        
        # Format percentage columns
        pct_columns = ['unrealized_pnl_pct', 'allocation_pct']
        for col in pct_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
        
        # Add color coding for P&L
        def highlight_pnl(val):
            if 'unrealized_pnl' in val.name and val.name != 'unrealized_pnl_pct':
                try:
                    num_val = float(val.replace('$', '').replace(',', ''))
                    color = 'color: green' if num_val >= 0 else 'color: red'
                    return color
                except:
                    return ''
            return ''
        
        # Display table with styling
        styled_df = display_df.style.applymap(highlight_pnl)
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary metrics
        if len(df) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_value = df['market_value'].sum()
                st.metric("Total Positions Value", f"${total_value:,.2f}")
            
            with col2:
                total_pnl = df['unrealized_pnl'].sum()
                st.metric("Total Unrealized P&L", f"${total_pnl:,.2f}")
            
            with col3:
                avg_return = df['unrealized_pnl_pct'].mean()
                st.metric("Average Return", f"{avg_return:.2f}%")
    
    def display_trades_table(self, trades: List[Dict[str, Any]], max_rows: int = 10):
        """Display recent trades table"""
        
        if not trades:
            st.info("No trades found")
            return
        
        st.subheader("🔄 Recent Trades")
        
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(trades)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False)
        
        # Limit rows
        df = df.head(max_rows)
        
        # Format for display
        display_df = df.copy()
        
        # Format currency columns
        if 'price' in display_df.columns:
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
        
        if 'total_value' in display_df.columns:
            display_df['total_value'] = display_df['total_value'].apply(lambda x: f"${x:,.2f}")
        
        # Format timestamp
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Color code by side
        def highlight_side(val):
            if val == 'BUY':
                return 'color: green'
            elif val == 'SELL':
                return 'color: red'
            return ''
        
        if 'side' in display_df.columns:
            styled_df = display_df.style.applymap(highlight_side, subset=['side'])
        else:
            styled_df = display_df
        
        st.dataframe(styled_df, use_container_width=True)

# ==================== CONTROL WIDGETS ====================

class TradingControlWidget:
    """Trading control and order entry widget"""
    
    def __init__(self, style: Optional[WidgetStyle] = None):
        self.style = style or WidgetStyle()
    
    def display_order_form(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """Display order entry form"""
        
        st.subheader("📋 Place Order")
        
        with st.form("order_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                symbol = st.selectbox("Symbol", symbols)
                side = st.selectbox("Side", ["BUY", "SELL"])
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "STOP"])
            
            with col2:
                quantity = st.number_input("Quantity", min_value=1, value=100)
                if order_type in ["LIMIT", "STOP"]:
                    price = st.number_input("Price", min_value=0.01, value=100.0, step=0.01)
                else:
                    price = None
            
            # Advanced options
            with st.expander("Advanced Options"):
                time_in_force = st.selectbox("Time in Force", ["DAY", "GTC", "IOC"])
                stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.01)
                take_profit = st.number_input("Take Profit", min_value=0.0, value=0.0, step=0.01)
            
            # Submit button
            submitted = st.form_submit_button("Place Order", type="primary")
            
            if submitted:
                order_data = {
                    "symbol": symbol,
                    "side": side,
                    "order_type": order_type,
                    "quantity": quantity,
                    "price": price,
                    "time_in_force": time_in_force,
                    "stop_loss": stop_loss if stop_loss > 0 else None,
                    "take_profit": take_profit if take_profit > 0 else None,
                    "timestamp": datetime.now()
                }
                return order_data
        
        return None
    
    def display_strategy_controls(self, available_strategies: List[str]) -> Dict[str, Any]:
        """Display strategy control panel"""
        
        st.subheader("🎯 Strategy Controls")
        
        # Strategy selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_strategy = st.selectbox("Select Strategy", available_strategies)
            auto_trading = st.checkbox("Auto Trading", value=False)
        
        with col2:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
            max_positions = st.number_input("Max Positions", min_value=1, max_value=50, value=10)
        
        # Risk controls
        with st.expander("Risk Controls"):
            col1, col2 = st.columns(2)
            
            with col1:
                max_position_size = st.slider("Max Position Size (%)", 1, 20, 5)
                daily_loss_limit = st.slider("Daily Loss Limit (%)", 1, 10, 3)
            
            with col2:
                portfolio_risk_limit = st.slider("Portfolio Risk Limit (%)", 1, 20, 10)
                correlation_limit = st.slider("Correlation Limit", 0.0, 1.0, 0.7, 0.05)
        
        return {
            "strategy": selected_strategy,
            "auto_trading": auto_trading,
            "confidence_threshold": confidence_threshold,
            "max_positions": max_positions,
            "max_position_size": max_position_size / 100,
            "daily_loss_limit": daily_loss_limit / 100,
            "portfolio_risk_limit": portfolio_risk_limit / 100,
            "correlation_limit": correlation_limit
        }

# ==================== ALERT WIDGETS ====================

class AlertWidget:
    """Alert and notification display widget"""
    
    def __init__(self, style: Optional[WidgetStyle] = None):
        self.style = style or WidgetStyle()
    
    def display_alerts(self, alerts: List[Dict[str, Any]], max_alerts: int = 5):
        """Display recent alerts"""
        
        if not alerts:
            return
        
        st.subheader("🚨 Recent Alerts")
        
        # Sort alerts by timestamp (most recent first)
        sorted_alerts = sorted(alerts, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        for alert in sorted_alerts[:max_alerts]:
            alert_type = alert.get('type', 'info')
            message = alert.get('message', 'No message')
            timestamp = alert.get('timestamp', datetime.now())
            
            # Determine alert style
            if alert_type == 'error':
                st.error(f"🔴 {message}")
            elif alert_type == 'warning':
                st.warning(f"🟡 {message}")
            elif alert_type == 'success':
                st.success(f"🟢 {message}")
            else:
                st.info(f"🔵 {message}")
            
            # Show timestamp in small text
            st.caption(f"⏰ {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def display_system_status(self, status_data: Dict[str, Any]):
        """Display system status indicators"""
        
        st.subheader("💾 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Data feed status
            data_feed_status = status_data.get('data_feed', 'unknown')
            if data_feed_status == 'connected':
                st.success("🟢 Data Feed")
            else:
                st.error("🔴 Data Feed")
        
        with col2:
            # Trading engine status
            trading_status = status_data.get('trading_engine', 'unknown')
            if trading_status == 'active':
                st.success("🟢 Trading")
            else:
                st.warning("🟡 Trading")
        
        with col3:
            # Algorithm status
            algo_status = status_data.get('algorithm', 'unknown')
            if algo_status == 'operational':
                st.success("🟢 Algorithm")
            else:
                st.error("🔴 Algorithm")
        
        with col4:
            # Risk management status
            risk_status = status_data.get('risk_management', 'unknown')
            if risk_status == 'active':
                st.success("🟢 Risk Mgmt")
            else:
                st.error("🔴 Risk Mgmt")
        
        # Performance metrics
        performance = status_data.get('performance', {})
        if performance:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_usage = performance.get('cpu_usage', 0)
                st.metric("CPU Usage", f"{cpu_usage:.1f}%")
            
            with col2:
                memory_usage = performance.get('memory_usage', 0)
                st.metric("Memory Usage", f"{memory_usage:.1f}%")
            
            with col3:
                algorithm_speed = performance.get('algorithm_speed', 0)
                st.metric("Algorithm Speed", f"{algorithm_speed:.2f}ms")

# ==================== SIGNAL WIDGETS ====================

class SignalWidget:
    """Trading signal display widget"""
    
    def __init__(self, style: Optional[WidgetStyle] = None):
        self.style = style or WidgetStyle()
    
    def display_recent_signals(self, signals: List[Dict[str, Any]], max_signals: int = 10):
        """Display recent trading signals"""
        
        if not signals:
            st.info("No recent signals")
            return
        
        st.subheader("📡 Recent Signals")
        
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        for signal in sorted_signals[:max_signals]:
            symbol = signal.get('symbol', 'N/A')
            signal_type = signal.get('signal_type', 'HOLD')
            confidence = signal.get('confidence', 0)
            price = signal.get('price', 0)
            timestamp = signal.get('timestamp', datetime.now())
            
            # Create signal display
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
            
            with col1:
                # Signal type with color
                if signal_type in ['BUY', 'STRONG_BUY']:
                    st.markdown(f"🟢 **{symbol}** - {signal_type}")
                elif signal_type in ['SELL', 'STRONG_SELL']:
                    st.markdown(f"🔴 **{symbol}** - {signal_type}")
                else:
                    st.markdown(f"🟡 **{symbol}** - {signal_type}")
            
            with col2:
                st.text(f"${price:.2f}")
            
            with col3:
                # Confidence with progress bar
                st.progress(confidence)
                st.caption(f"{confidence:.0%}")
            
            with col4:
                st.caption(timestamp.strftime('%H:%M:%S'))
            
            st.divider()
    
    def display_signal_summary(self, signal_stats: Dict[str, Any]):
        """Display signal summary statistics"""
        
        st.subheader("📊 Signal Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_signals = signal_stats.get('total_signals', 0)
            st.metric("Total Signals", total_signals)
        
        with col2:
            success_rate = signal_stats.get('success_rate', 0)
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col3:
            avg_confidence = signal_stats.get('avg_confidence', 0)
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        with col4:
            signals_today = signal_stats.get('signals_today', 0)
            st.metric("Signals Today", signals_today)

# ==================== MARKET OVERVIEW WIDGET ====================

class MarketOverviewWidget:
    """Market overview and watchlist widget"""
    
    def __init__(self, style: Optional[WidgetStyle] = None):
        self.style = style or WidgetStyle()
    
    def display_market_overview(self, market_data: Dict[str, Any]):
        """Display market overview"""
        
        st.subheader("🌍 Market Overview")
        
        # Market indices
        indices = market_data.get('indices', {})
        if indices:
            col1, col2, col3 = st.columns(3)
            
            for i, (index, data) in enumerate(indices.items()):
                with [col1, col2, col3][i % 3]:
                    price = data.get('price', 0)
                    change = data.get('change', 0)
                    change_pct = data.get('change_pct', 0)
                    
                    delta_color = "normal" if change >= 0 else "inverse"
                    st.metric(
                        index,
                        f"${price:,.2f}",
                        f"{change:+.2f} ({change_pct:+.2f}%)",
                        delta_color=delta_color
                    )
        
        # Market sentiment
        sentiment = market_data.get('sentiment', {})
        if sentiment:
            st.markdown("**Market Sentiment**")
            col1, col2 = st.columns(2)
            
            with col1:
                fear_greed = sentiment.get('fear_greed', 50)
                st.metric("Fear & Greed Index", fear_greed)
                
                # Add interpretation
                if fear_greed < 20:
                    st.caption("🔴 Extreme Fear")
                elif fear_greed < 40:
                    st.caption("🟠 Fear")
                elif fear_greed < 60:
                    st.caption("🟡 Neutral")
                elif fear_greed < 80:
                    st.caption("🟢 Greed")
                else:
                    st.caption("🔴 Extreme Greed")
            
            with col2:
                vix = sentiment.get('vix', 20)
                st.metric("VIX", f"{vix:.2f}")
                
                # Add interpretation
                if vix < 15:
                    st.caption("🟢 Low Volatility")
                elif vix < 25:
                    st.caption("🟡 Normal Volatility")
                elif vix < 35:
                    st.caption("🟠 High Volatility")
                else:
                    st.caption("🔴 Extreme Volatility")
    
    def display_watchlist(self, watchlist: List[Dict[str, Any]]):
        """Display stock watchlist"""
        
        st.subheader("👀 Watchlist")
        
        if not watchlist:
            st.info("No items in watchlist")
            return
        
        # Create watchlist table
        for item in watchlist:
            symbol = item.get('symbol', 'N/A')
            price = item.get('price', 0)
            change = item.get('change', 0)
            change_pct = item.get('change_pct', 0)
            volume = item.get('volume', 0)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                st.markdown(f"**{symbol}**")
            
            with col2:
                st.text(f"${price:.2f}")
            
            with col3:
                color = "🟢" if change >= 0 else "🔴"
                st.text(f"{color} {change:+.2f} ({change_pct:+.2f}%)")
            
            with col4:
                if volume >= 1e6:
                    st.caption(f"{volume/1e6:.1f}M")
                else:
                    st.caption(f"{volume/1e3:.0f}K")
        
        st.divider()

# ==================== UTILITY FUNCTIONS ====================

def create_widget_container(title: str, content_func: Callable, **kwargs):
    """Create a styled container for widgets"""
    
    with st.container():
        st.markdown(f"### {title}")
        content_func(**kwargs)
        st.markdown("---")

def display_loading_spinner(text: str = "Loading..."):
    """Display loading spinner"""
    with st.spinner(text):
        return st.empty()

def display_success_message(message: str, duration: int = 3):
    """Display temporary success message"""
    placeholder = st.empty()
    placeholder.success(message)
    # Note: In a real implementation, you'd use time.sleep(duration) 
    # and then placeholder.empty(), but that would block the UI

def display_error_message(message: str):
    """Display error message"""
    st.error(f"❌ {message}")

def display_warning_message(message: str):
    """Display warning message"""
    st.warning(f"⚠️ {message}")

def display_info_message(message: str):
    """Display info message"""
    st.info(f"ℹ️ {message}")

# ==================== TESTING FUNCTION ====================

def test_widgets():
    """Test widget functionality"""
    
    print("🎛️ Testing Widget Components")
    print("=" * 50)
    
    # Test data
    portfolio_data = {
        'total_value': 125000.50,
        'daily_change': 1250.75,
        'daily_pnl': 1250.75,
        'daily_pnl_pct': 1.01,
        'total_return_pct': 25.0,
        'sharpe_ratio': 1.85,
        'max_drawdown': -8.5,
        'win_rate': 68.5
    }
    
    risk_data = {
        'risk_level': 'Medium',
        'risk_score': 0.45,
        'portfolio_risk': 12.5,
        'var_95': -2500,
        'beta': 1.15
    }
    
    positions = [
        {'symbol': 'AAPL', 'quantity': 100, 'market_value': 18000, 'cost_basis': 17000, 
         'unrealized_pnl': 1000, 'unrealized_pnl_pct': 5.88, 'allocation_pct': 14.4},
        {'symbol': 'BTC-USD', 'quantity': 0.5, 'market_value': 22500, 'cost_basis': 20000,
         'unrealized_pnl': 2500, 'unrealized_pnl_pct': 12.5, 'allocation_pct': 18.0}
    ]
    
    signals = [
        {'symbol': 'TSLA', 'signal_type': 'BUY', 'confidence': 0.85, 'price': 245.50,
         'timestamp': datetime.now() - timedelta(minutes=5)},
        {'symbol': 'NVDA', 'signal_type': 'STRONG_SELL', 'confidence': 0.92, 'price': 875.25,
         'timestamp': datetime.now() - timedelta(minutes=15)}
    ]
    
    print("✅ Created test data")
    
    # Test metric widget
    metric_widget = MetricWidget()
    print("✅ Created MetricWidget")
    
    # Test performance widget
    perf_widget = PerformanceWidget()
    print("✅ Created PerformanceWidget")
    
    # Test risk widget
    risk_widget = RiskWidget()
    print("✅ Created RiskWidget")
    
    # Test data table widget
    table_widget = DataTableWidget()
    print("✅ Created DataTableWidget")
    
    # Test trading control widget
    control_widget = TradingControlWidget()
    print("✅ Created TradingControlWidget")
    
    # Test alert widget
    alert_widget = AlertWidget()
    print("✅ Created AlertWidget")
    
    # Test signal widget
    signal_widget = SignalWidget()
    print("✅ Created SignalWidget")
    
    # Test market overview widget
    market_widget = MarketOverviewWidget()
    print("✅ Created MarketOverviewWidget")
    
    print("\n🎉 Widget components testing completed!")

if __name__ == "__main__":
    test_widgets()