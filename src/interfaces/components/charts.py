#!/usr/bin/env python3
"""
File: charts.py
Path: NeuroCluster-Elite/src/interfaces/components/charts.py
Description: Advanced charting components for NeuroCluster Elite dashboard

This module provides sophisticated charting components optimized for financial
data visualization, including candlestick charts, technical indicators,
pattern overlays, and interactive features for the Streamlit dashboard.

Features:
- Professional candlestick and OHLC charts
- Technical indicator overlays (RSI, MACD, Bollinger Bands, etc.)
- Pattern recognition visualization
- Multi-timeframe analysis
- Interactive features with Plotly
- Performance optimization for real-time data
- Mobile-responsive design
- Export capabilities

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.core.pattern_recognition import RecognizedPattern, PatternType
    from src.trading.strategies.base_strategy import TradingSignal, SignalType
    from src.utils.helpers import format_currency, format_percentage, calculate_returns
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

# ==================== CHART CONFIGURATION ====================

class ChartType(Enum):
    """Supported chart types"""
    CANDLESTICK = "candlestick"
    OHLC = "ohlc"
    LINE = "line"
    AREA = "area"
    VOLUME = "volume"
    HEATMAP = "heatmap"
    SCATTER = "scatter"

class ChartTheme(Enum):
    """Chart themes"""
    DARK = "plotly_dark"
    LIGHT = "plotly_white"
    PROFESSIONAL = "plotly"
    MINIMAL = "simple_white"

@dataclass
class ChartConfig:
    """Configuration for chart appearance and behavior"""
    # Basic settings
    theme: ChartTheme = ChartTheme.DARK
    height: int = 600
    width: Optional[int] = None
    
    # Colors
    bullish_color: str = "#00ff88"
    bearish_color: str = "#ff4444"
    volume_color: str = "#888888"
    ma_colors: List[str] = field(default_factory=lambda: ["#ff9500", "#00bfff", "#ff00ff"])
    
    # Technical indicators
    show_volume: bool = True
    show_ma: bool = True
    ma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    show_bollinger: bool = False
    show_rsi: bool = False
    show_macd: bool = False
    
    # Interactive features
    enable_crossfilter: bool = True
    enable_range_selector: bool = True
    enable_zoom: bool = True
    enable_pan: bool = True
    
    # Performance
    max_candles: int = 1000
    downsample_threshold: int = 5000
    update_interval_ms: int = 1000

# ==================== BASE CHART CLASS ====================

class BaseChart:
    """Base class for all chart components"""
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """Initialize base chart"""
        self.config = config or ChartConfig()
        self.fig = None
        
    def _create_base_figure(self, rows: int = 1, shared_xaxes: bool = True) -> go.Figure:
        """Create base figure with subplots"""
        
        if rows == 1:
            fig = go.Figure()
        else:
            subplot_titles = self._get_subplot_titles(rows)
            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=shared_xaxes,
                vertical_spacing=0.03,
                subplot_titles=subplot_titles,
                row_heights=self._get_row_heights(rows)
            )
        
        # Apply theme and basic layout
        fig.update_layout(
            template=self.config.theme.value,
            height=self.config.height,
            width=self.config.width,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='x unified'
        )
        
        return fig
    
    def _get_subplot_titles(self, rows: int) -> List[str]:
        """Get subplot titles based on number of rows"""
        titles = ["Price"]
        if rows > 1:
            titles.append("Volume")
        if rows > 2:
            titles.append("RSI")
        if rows > 3:
            titles.append("MACD")
        return titles[:rows]
    
    def _get_row_heights(self, rows: int) -> List[float]:
        """Get row heights for subplots"""
        if rows == 1:
            return [1.0]
        elif rows == 2:
            return [0.7, 0.3]
        elif rows == 3:
            return [0.6, 0.2, 0.2]
        elif rows == 4:
            return [0.5, 0.2, 0.15, 0.15]
        else:
            # Distribute remaining space evenly
            main_height = 0.5
            remaining = 1.0 - main_height
            sub_height = remaining / (rows - 1)
            return [main_height] + [sub_height] * (rows - 1)
    
    def _add_range_selector(self, fig: go.Figure):
        """Add range selector buttons"""
        if not self.config.enable_range_selector:
            return
            
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=30, label="30D", step="day", stepmode="backward"),
                        dict(count=90, label="3M", step="day", stepmode="backward"),
                        dict(count=365, label="1Y", step="day", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="rgba(255,255,255,0.2)",
                    borderwidth=1
                ),
                rangeslider=dict(visible=False),
                type="date"
            )
        )
    
    def _optimize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data for performance"""
        if len(data) <= self.config.max_candles:
            return data
        
        # Downsample if too many data points
        if len(data) > self.config.downsample_threshold:
            # Keep recent data at full resolution, downsample older data
            recent_days = 30
            recent_data = data.tail(recent_days * 24)  # Last 30 days at full resolution
            older_data = data.head(len(data) - len(recent_data))
            
            # Downsample older data to hourly
            if len(older_data) > 0:
                older_data = older_data.resample('1H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            
            return pd.concat([older_data, recent_data])
        
        # Simple tail if not too much data
        return data.tail(self.config.max_candles)

# ==================== CANDLESTICK CHART ====================

class CandlestickChart(BaseChart):
    """Professional candlestick chart component"""
    
    def create_chart(self, data: pd.DataFrame, symbol: str, 
                    patterns: Optional[List[RecognizedPattern]] = None,
                    signals: Optional[List[TradingSignal]] = None) -> go.Figure:
        """Create candlestick chart with indicators and overlays"""
        
        try:
            # Optimize data for performance
            data = self._optimize_data(data)
            
            # Determine number of subplots needed
            rows = 1
            if self.config.show_volume:
                rows += 1
            if self.config.show_rsi:
                rows += 1
            if self.config.show_macd:
                rows += 1
            
            # Create base figure
            fig = self._create_base_figure(rows=rows, shared_xaxes=True)
            
            # Add main candlestick chart
            self._add_candlesticks(fig, data, row=1)
            
            # Add moving averages
            if self.config.show_ma:
                self._add_moving_averages(fig, data, row=1)
            
            # Add Bollinger Bands
            if self.config.show_bollinger:
                self._add_bollinger_bands(fig, data, row=1)
            
            # Add patterns
            if patterns:
                self._add_patterns(fig, data, patterns, row=1)
            
            # Add trading signals
            if signals:
                self._add_signals(fig, data, signals, row=1)
            
            # Add volume subplot
            current_row = 1
            if self.config.show_volume:
                current_row += 1
                self._add_volume(fig, data, row=current_row)
            
            # Add RSI subplot
            if self.config.show_rsi:
                current_row += 1
                self._add_rsi(fig, data, row=current_row)
            
            # Add MACD subplot
            if self.config.show_macd:
                current_row += 1
                self._add_macd(fig, data, row=current_row)
            
            # Finalize chart
            self._finalize_chart(fig, symbol)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {e}")
            return self._create_error_chart(str(e))
    
    def _add_candlesticks(self, fig: go.Figure, data: pd.DataFrame, row: int):
        """Add candlestick trace"""
        
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color=self.config.bullish_color,
                decreasing_line_color=self.config.bearish_color,
                increasing_fillcolor=self.config.bullish_color,
                decreasing_fillcolor=self.config.bearish_color,
                line=dict(width=1),
                hovertemplate="<b>%{x}</b><br>" +
                           "Open: $%{open:,.2f}<br>" +
                           "High: $%{high:,.2f}<br>" +
                           "Low: $%{low:,.2f}<br>" +
                           "Close: $%{close:,.2f}<extra></extra>"
            ),
            row=row, col=1
        )
    
    def _add_moving_averages(self, fig: go.Figure, data: pd.DataFrame, row: int):
        """Add moving average lines"""
        
        for i, period in enumerate(self.config.ma_periods):
            if len(data) >= period:
                ma = data['close'].rolling(window=period).mean()
                color = self.config.ma_colors[i % len(self.config.ma_colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma,
                        mode='lines',
                        name=f'MA{period}',
                        line=dict(color=color, width=1),
                        opacity=0.8,
                        hovertemplate=f"MA{period}: $%{{y:,.2f}}<extra></extra>"
                    ),
                    row=row, col=1
                )
    
    def _add_bollinger_bands(self, fig: go.Figure, data: pd.DataFrame, row: int):
        """Add Bollinger Bands"""
        
        if len(data) >= 20:
            # Calculate Bollinger Bands
            ma20 = data['close'].rolling(window=20).mean()
            std20 = data['close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            
            # Add upper band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=upper_band,
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                    hovertemplate="BB Upper: $%{y:,.2f}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add lower band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=lower_band,
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.1)',
                    hovertemplate="BB Lower: $%{y:,.2f}<extra></extra>"
                ),
                row=row, col=1
            )
    
    def _add_patterns(self, fig: go.Figure, data: pd.DataFrame, 
                     patterns: List[RecognizedPattern], row: int):
        """Add pattern recognition overlays"""
        
        for pattern in patterns[-10:]:  # Show last 10 patterns
            try:
                # Find the pattern time range in data
                pattern_start = pattern.start_time
                pattern_end = pattern.end_time
                
                # Get data subset for pattern
                pattern_data = data[(data.index >= pattern_start) & (data.index <= pattern_end)]
                
                if len(pattern_data) == 0:
                    continue
                
                # Determine pattern color
                pattern_color = self._get_pattern_color(pattern.pattern_type, pattern.signal)
                
                # Add pattern annotation
                fig.add_annotation(
                    x=pattern_end,
                    y=pattern_data['high'].max(),
                    text=f"{pattern.pattern_type.value.replace('_', ' ').title()}<br>Conf: {pattern.confidence:.0%}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=pattern_color,
                    ax=20,
                    ay=-30,
                    bordercolor=pattern_color,
                    borderwidth=2,
                    bgcolor="rgba(0,0,0,0.8)",
                    font=dict(color="white", size=10),
                    row=row, col=1
                )
                
                # Add pattern shape if possible
                self._add_pattern_shape(fig, pattern, pattern_data, pattern_color, row)
                
            except Exception as e:
                logger.warning(f"Error adding pattern {pattern.pattern_type}: {e}")
    
    def _add_signals(self, fig: go.Figure, data: pd.DataFrame, 
                    signals: List[TradingSignal], row: int):
        """Add trading signal markers"""
        
        for signal in signals[-20:]:  # Show last 20 signals
            try:
                # Find signal timestamp in data
                signal_time = signal.timestamp
                signal_data = data[data.index <= signal_time]
                
                if len(signal_data) == 0:
                    continue
                
                # Get signal price and position
                signal_price = signal.current_price
                signal_idx = signal_data.index[-1]
                
                # Determine signal appearance
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    symbol_char = "▲"
                    color = self.config.bullish_color
                    y_position = signal_price * 0.99  # Below price
                else:
                    symbol_char = "▼"
                    color = self.config.bearish_color
                    y_position = signal_price * 1.01  # Above price
                
                # Add signal marker
                fig.add_trace(
                    go.Scatter(
                        x=[signal_idx],
                        y=[y_position],
                        mode='markers+text',
                        text=[symbol_char],
                        textfont=dict(size=20, color=color),
                        marker=dict(size=1, opacity=0),  # Invisible marker
                        name=f"{signal.signal_type.value} Signal",
                        showlegend=False,
                        hovertemplate=f"<b>{signal.signal_type.value}</b><br>" +
                                    f"Price: ${signal_price:,.2f}<br>" +
                                    f"Confidence: {signal.confidence:.0%}<br>" +
                                    f"Strategy: {signal.strategy_name}<extra></extra>"
                    ),
                    row=row, col=1
                )
                
            except Exception as e:
                logger.warning(f"Error adding signal: {e}")
    
    def _add_volume(self, fig: go.Figure, data: pd.DataFrame, row: int):
        """Add volume subplot"""
        
        # Determine volume colors based on price movement
        colors = []
        for i in range(len(data)):
            if i == 0:
                colors.append(self.config.volume_color)
            else:
                if data['close'].iloc[i] >= data['close'].iloc[i-1]:
                    colors.append(self.config.bullish_color)
                else:
                    colors.append(self.config.bearish_color)
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name="Volume",
                marker=dict(color=colors, opacity=0.7),
                hovertemplate="Volume: %{y:,.0f}<extra></extra>"
            ),
            row=row, col=1
        )
        
        # Update y-axis for volume
        fig.update_yaxes(title_text="Volume", row=row, col=1)
    
    def _add_rsi(self, fig: go.Figure, data: pd.DataFrame, row: int):
        """Add RSI subplot"""
        
        # Calculate RSI
        rsi = self._calculate_rsi(data['close'])
        
        if len(rsi.dropna()) > 0:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='#ffa500', width=2),
                    hovertemplate="RSI: %{y:.1f}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=1)
            
            # Update y-axis for RSI
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)
    
    def _add_macd(self, fig: go.Figure, data: pd.DataFrame, row: int):
        """Add MACD subplot"""
        
        # Calculate MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(data['close'])
        
        if len(macd_line.dropna()) > 0:
            # MACD line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=macd_line,
                    mode='lines',
                    name='MACD',
                    line=dict(color='#00bfff', width=2),
                    hovertemplate="MACD: %{y:.3f}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Signal line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=macd_signal,
                    mode='lines',
                    name='Signal',
                    line=dict(color='#ff4500', width=1),
                    hovertemplate="Signal: %{y:.3f}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Histogram
            histogram_colors = ['red' if x < 0 else 'green' for x in macd_histogram]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=macd_histogram,
                    name='Histogram',
                    marker=dict(color=histogram_colors, opacity=0.7),
                    hovertemplate="Histogram: %{y:.3f}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=1)
            
            # Update y-axis for MACD
            fig.update_yaxes(title_text="MACD", row=row, col=1)
    
    def _get_pattern_color(self, pattern_type: PatternType, signal) -> str:
        """Get color for pattern based on type and signal"""
        
        bullish_patterns = [
            PatternType.INVERSE_HEAD_AND_SHOULDERS,
            PatternType.TRIANGLE_ASCENDING,
            PatternType.FLAG_BULL,
            PatternType.DOUBLE_BOTTOM,
            PatternType.SUPPORT_LEVEL,
            PatternType.BREAKOUT
        ]
        
        if pattern_type in bullish_patterns:
            return self.config.bullish_color
        else:
            return self.config.bearish_color
    
    def _add_pattern_shape(self, fig: go.Figure, pattern: RecognizedPattern, 
                          pattern_data: pd.DataFrame, color: str, row: int):
        """Add pattern shape overlay"""
        
        # Simplified pattern shape - could be enhanced with specific pattern geometry
        if len(pattern.key_points) >= 2:
            x_coords = [point.timestamp for point in pattern.key_points]
            y_coords = [point.price for point in pattern.key_points]
            
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    name=f"{pattern.pattern_type.value}",
                    line=dict(color=color, width=2, dash='dot'),
                    marker=dict(color=color, size=6),
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=f"Pattern: {pattern.pattern_type.value}<br>" +
                                f"Confidence: {pattern.confidence:.0%}<extra></extra>"
                ),
                row=row, col=1
            )
    
    def _finalize_chart(self, fig: go.Figure, symbol: str):
        """Apply final styling and configuration"""
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{symbol}</b> - Advanced Chart Analysis",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_rangeslider_visible=False,
            dragmode='zoom' if self.config.enable_zoom else 'pan'
        )
        
        # Add range selector
        self._add_range_selector(fig)
        
        # Update all x-axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.5)'
        )
        
        # Update all y-axes
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.5)'
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when data loading fails"""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error loading chart data:<br>{error_message}",
            showarrow=False,
            font=dict(size=16, color="red"),
            xref="paper", yref="paper"
        )
        fig.update_layout(
            template=self.config.theme.value,
            height=400,
            showlegend=False
        )
        return fig

# ==================== ADDITIONAL CHART TYPES ====================

class LineChart(BaseChart):
    """Simple line chart for quick visualization"""
    
    def create_chart(self, data: pd.DataFrame, symbol: str, 
                    y_column: str = 'close') -> go.Figure:
        """Create simple line chart"""
        
        fig = self._create_base_figure(rows=1)
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[y_column],
                mode='lines',
                name=symbol,
                line=dict(color=self.config.bullish_color, width=2),
                hovertemplate=f"{symbol}: $%{{y:,.2f}}<extra></extra>"
            )
        )
        
        fig.update_layout(
            title=f"{symbol} - Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)"
        )
        
        return fig

class PerformanceChart(BaseChart):
    """Performance comparison chart"""
    
    def create_chart(self, data: Dict[str, pd.DataFrame], 
                    normalize: bool = True) -> go.Figure:
        """Create performance comparison chart"""
        
        fig = self._create_base_figure(rows=1)
        
        colors = ['#00ff88', '#ff4444', '#00bfff', '#ff9500', '#ff00ff']
        
        for i, (symbol, df) in enumerate(data.items()):
            if 'close' not in df.columns:
                continue
                
            prices = df['close']
            if normalize:
                # Normalize to starting value of 100
                performance = (prices / prices.iloc[0]) * 100
                y_title = "Performance (Base 100)"
            else:
                performance = prices
                y_title = "Price ($)"
            
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=performance,
                    mode='lines',
                    name=symbol,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{symbol}: %{{y:,.2f}}<extra></extra>"
                )
            )
        
        fig.update_layout(
            title="Performance Comparison",
            xaxis_title="Date",
            yaxis_title=y_title
        )
        
        return fig

# ==================== UTILITY FUNCTIONS ====================

def create_chart_component(chart_type: ChartType, **kwargs) -> BaseChart:
    """Factory function to create chart components"""
    
    config = kwargs.get('config', ChartConfig())
    
    if chart_type == ChartType.CANDLESTICK:
        return CandlestickChart(config)
    elif chart_type == ChartType.LINE:
        return LineChart(config)
    else:
        return CandlestickChart(config)  # Default to candlestick

def display_chart(chart: go.Figure, key: str = None):
    """Display chart in Streamlit with optimal settings"""
    
    st.plotly_chart(
        chart,
        use_container_width=True,
        key=key,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian'
            ],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'neurocluster_chart',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
    )

def create_mini_chart(data: pd.DataFrame, symbol: str, height: int = 200) -> go.Figure:
    """Create mini chart for overview display"""
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name=symbol,
            line=dict(color='#00ff88', width=1),
            fill='tonexty',
            fillcolor='rgba(0, 255, 136, 0.1)'
        )
    )
    
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ==================== TESTING FUNCTION ====================

def test_chart_components():
    """Test chart components functionality"""
    
    print("📊 Testing Chart Components")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        daily_return = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
        if i == 0:
            close = base_price
        else:
            close = prices[-1]['close'] * (1 + daily_return)
        
        # Generate OHLC
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        
        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
        
        volumes.append(np.random.randint(500000, 2000000))
    
    # Create DataFrame
    sample_data = pd.DataFrame(prices, index=dates)
    sample_data['volume'] = volumes
    
    print(f"✅ Created sample data with {len(sample_data)} records")
    
    # Test candlestick chart
    config = ChartConfig(show_volume=True, show_rsi=True, show_macd=True)
    chart = CandlestickChart(config)
    
    fig = chart.create_chart(sample_data, "TEST")
    print(f"✅ Created candlestick chart with {len(fig.data)} traces")
    
    # Test line chart
    line_chart = LineChart()
    line_fig = line_chart.create_chart(sample_data, "TEST")
    print(f"✅ Created line chart with {len(line_fig.data)} traces")
    
    # Test performance chart
    perf_chart = PerformanceChart()
    perf_data = {"TEST1": sample_data, "TEST2": sample_data * 1.1}
    perf_fig = perf_chart.create_chart(perf_data)
    print(f"✅ Created performance chart with {len(perf_fig.data)} traces")
    
    print("\n🎉 Chart components testing completed!")

if __name__ == "__main__":
    test_chart_components()