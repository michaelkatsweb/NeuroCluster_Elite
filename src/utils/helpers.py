#!/usr/bin/env python3
"""
File: helpers.py
Path: NeuroCluster-Elite/src/utils/helpers.py
Description: Utility helper functions for NeuroCluster Elite

This module provides common utility functions used throughout the application
including formatting, data conversion, mathematical operations, and display utilities.

Features:
- Data formatting (currency, percentage, numbers)
- Table formatting for console output
- Mathematical utilities
- Date/time helpers
- File operations
- Performance utilities

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import os
import sys
import time
import json
import pickle
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
import math
import statistics
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
import logging

# Configure logging
logger = logging.getLogger(__name__)

# ==================== FORMATTING UTILITIES ====================

def format_currency(amount: float, currency: str = "USD", decimal_places: int = 2) -> str:
    """
    Format currency amount with proper symbol and formatting
    
    Args:
        amount: Amount to format
        currency: Currency code (USD, EUR, etc.)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CAD': 'C$',
        'AUD': 'A$',
        'CHF': 'CHF',
        'BTC': '₿',
        'ETH': 'Ξ'
    }
    
    symbol = currency_symbols.get(currency, currency)
    
    # Handle negative amounts
    if amount < 0:
        return f"-{symbol}{abs(amount):,.{decimal_places}f}"
    else:
        return f"{symbol}{amount:,.{decimal_places}f}"

def format_percentage(value: float, decimal_places: int = 2, show_sign: bool = True) -> str:
    """
    Format percentage value
    
    Args:
        value: Percentage value (e.g., 0.05 for 5% or 5.0 for 5%)
        decimal_places: Number of decimal places
        show_sign: Whether to show + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    
    # Auto-detect if value is already in percentage form
    if abs(value) > 1:
        percentage = value
    else:
        percentage = value * 100
    
    if show_sign and percentage > 0:
        return f"+{percentage:.{decimal_places}f}%"
    else:
        return f"{percentage:.{decimal_places}f}%"

def format_number(value: float, decimal_places: int = 2, use_thousands_separator: bool = True) -> str:
    """
    Format number with thousands separator
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places
        use_thousands_separator: Whether to use thousands separator
        
    Returns:
        Formatted number string
    """
    
    if use_thousands_separator:
        return f"{value:,.{decimal_places}f}"
    else:
        return f"{value:.{decimal_places}f}"

def format_large_number(value: float, precision: int = 1) -> str:
    """
    Format large numbers with K, M, B suffixes
    
    Args:
        value: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number with suffix
    """
    
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value >= 1e12:
        return f"{sign}{abs_value/1e12:.{precision}f}T"
    elif abs_value >= 1e9:
        return f"{sign}{abs_value/1e9:.{precision}f}B"
    elif abs_value >= 1e6:
        return f"{sign}{abs_value/1e6:.{precision}f}M"
    elif abs_value >= 1e3:
        return f"{sign}{abs_value/1e3:.{precision}f}K"
    else:
        return f"{sign}{abs_value:.{precision}f}"

def format_time_duration(seconds: float) -> str:
    """
    Format time duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def format_timestamp(timestamp: Union[datetime, float, str], format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp in various formats
    
    Args:
        timestamp: Timestamp to format
        format_string: Format string
        
    Returns:
        Formatted timestamp string
    """
    
    if isinstance(timestamp, str):
        # Try to parse ISO format
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp
    elif isinstance(timestamp, float):
        dt = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return str(timestamp)
    
    return dt.strftime(format_string)

# ==================== TABLE FORMATTING ====================

def print_table(data: List[List[str]], headers: List[str] = None, 
                title: str = None, border_style: str = "simple") -> None:
    """
    Print formatted table to console
    
    Args:
        data: Table data as list of rows
        headers: Optional column headers
        title: Optional table title
        border_style: Border style (simple, double, minimal)
    """
    
    if not data:
        print("No data to display")
        return
    
    # Determine column widths
    all_rows = []
    if headers:
        all_rows.append(headers)
    all_rows.extend(data)
    
    col_widths = []
    for col_idx in range(len(all_rows[0])):
        max_width = max(len(str(row[col_idx])) for row in all_rows if col_idx < len(row))
        col_widths.append(max_width)
    
    # Border characters
    if border_style == "double":
        h_char, v_char, corner = "═", "║", "╬"
        top_left, top_right = "╔", "╗"
        bottom_left, bottom_right = "╚", "╝"
    elif border_style == "minimal":
        h_char, v_char, corner = "-", "|", "+"
        top_left = top_right = bottom_left = bottom_right = corner
    else:  # simple
        h_char, v_char, corner = "-", "|", "+"
        top_left = top_right = bottom_left = bottom_right = corner
    
    # Calculate total width
    total_width = sum(col_widths) + len(col_widths) * 3 + 1
    
    # Print title if provided
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    
    # Print top border
    border_line = top_left + h_char * (total_width - 2) + top_right
    print(border_line)
    
    # Print headers
    if headers:
        header_line = v_char
        for i, header in enumerate(headers):
            header_line += f" {header:<{col_widths[i]}} {v_char}"
        print(header_line)
        
        # Print header separator
        sep_line = corner
        for width in col_widths:
            sep_line += h_char * (width + 2) + corner
        print(sep_line)
    
    # Print data rows
    for row in data:
        row_line = v_char
        for i, cell in enumerate(row):
            if i < len(col_widths):
                row_line += f" {str(cell):<{col_widths[i]}} {v_char}"
        print(row_line)
    
    # Print bottom border
    bottom_border = bottom_left + h_char * (total_width - 2) + bottom_right
    print(bottom_border)

def print_banner(text: str, width: int = 80, char: str = "=", center: bool = True) -> None:
    """
    Print a banner with text
    
    Args:
        text: Text to display
        width: Banner width
        char: Border character
        center: Whether to center the text
    """
    
    print(char * width)
    
    if center:
        padding = (width - len(text) - 2) // 2
        print(f"{char}{' ' * padding}{text}{' ' * padding}{char}")
    else:
        print(f"{char} {text}")
    
    print(char * width)

def print_progress_bar(current: int, total: int, width: int = 50, 
                      char: str = "█", empty_char: str = "░") -> None:
    """
    Print a progress bar
    
    Args:
        current: Current progress
        total: Total items
        width: Progress bar width
        char: Fill character
        empty_char: Empty character
    """
    
    if total == 0:
        percentage = 0
    else:
        percentage = min(current / total, 1.0)
    
    filled_width = int(width * percentage)
    bar = char * filled_width + empty_char * (width - filled_width)
    
    print(f"\r[{bar}] {percentage*100:.1f}% ({current}/{total})", end="", flush=True)

# ==================== MATHEMATICAL UTILITIES ====================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles division by zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    
    return ((new_value - old_value) / old_value) * 100

def calculate_compound_return(returns: List[float]) -> float:
    """
    Calculate compound return from a list of period returns
    
    Args:
        returns: List of period returns (as decimals, e.g., 0.05 for 5%)
        
    Returns:
        Compound return
    """
    
    if not returns:
        return 0.0
    
    compound = 1.0
    for r in returns:
        compound *= (1 + r)
    
    return compound - 1.0

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    
    if not returns or len(returns) < 2:
        return 0.0
    
    excess_returns = [r - risk_free_rate for r in returns]
    
    try:
        return statistics.mean(excess_returns) / statistics.stdev(excess_returns)
    except statistics.StatisticsError:
        return 0.0

def calculate_max_drawdown(values: List[float]) -> float:
    """
    Calculate maximum drawdown from a series of values
    
    Args:
        values: List of portfolio values
        
    Returns:
        Maximum drawdown as percentage
    """
    
    if not values or len(values) < 2:
        return 0.0
    
    peak = values[0]
    max_drawdown = 0.0
    
    for value in values[1:]:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown * 100  # Return as percentage

def moving_average(data: List[float], window: int) -> List[float]:
    """
    Calculate moving average
    
    Args:
        data: Data series
        window: Window size
        
    Returns:
        Moving average series
    """
    
    if window <= 0 or window > len(data):
        return data.copy()
    
    averages = []
    for i in range(len(data)):
        if i < window - 1:
            # Not enough data points, use available data
            averages.append(sum(data[:i+1]) / (i+1))
        else:
            # Calculate moving average
            window_data = data[i-window+1:i+1]
            averages.append(sum(window_data) / window)
    
    return averages

def exponential_moving_average(data: List[float], alpha: float = 0.1) -> List[float]:
    """
    Calculate exponential moving average
    
    Args:
        data: Data series
        alpha: Smoothing factor (0 < alpha <= 1)
        
    Returns:
        EMA series
    """
    
    if not data:
        return []
    
    ema = [data[0]]  # Initialize with first value
    
    for i in range(1, len(data)):
        ema_value = alpha * data[i] + (1 - alpha) * ema[i-1]
        ema.append(ema_value)
    
    return ema

# ==================== DATE/TIME UTILITIES ====================

def get_market_hours(timezone_name: str = "US/Eastern") -> Dict[str, Any]:
    """
    Get market hours information
    
    Args:
        timezone_name: Timezone name
        
    Returns:
        Market hours information
    """
    
    try:
        import pytz
        tz = pytz.timezone(timezone_name)
        now = datetime.now(tz)
        
        # US market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_open = market_open <= now <= market_close and now.weekday() < 5
        
        return {
            'is_open': is_open,
            'open_time': market_open,
            'close_time': market_close,
            'current_time': now,
            'is_weekend': now.weekday() >= 5
        }
    except ImportError:
        # Fallback without pytz
        now = datetime.now()
        return {
            'is_open': 9 <= now.hour < 16 and now.weekday() < 5,
            'current_time': now,
            'is_weekend': now.weekday() >= 5
        }

def is_trading_day(date: datetime = None) -> bool:
    """
    Check if a date is a trading day (weekday, excluding holidays)
    
    Args:
        date: Date to check (default: today)
        
    Returns:
        True if trading day
    """
    
    if date is None:
        date = datetime.now()
    
    # Weekend check
    if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # TODO: Add holiday checking logic
    # For now, just check weekdays
    return True

def get_business_days_between(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of business days between two dates
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of business days
    """
    
    business_days = 0
    current_date = start_date
    
    while current_date <= end_date:
        if is_trading_day(current_date):
            business_days += 1
        current_date += timedelta(days=1)
    
    return business_days

# ==================== FILE OPERATIONS ====================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def safe_file_write(file_path: Union[str, Path], data: Any, 
                   backup: bool = True, format_type: str = "json") -> bool:
    """
    Safely write data to file with backup
    
    Args:
        file_path: File path
        data: Data to write
        backup: Whether to create backup
        format_type: Format type (json, pickle, text)
        
    Returns:
        True if successful
    """
    
    file_path = Path(file_path)
    
    try:
        # Create backup if requested
        if backup and file_path.exists():
            backup_path = file_path.with_suffix(f".backup{file_path.suffix}")
            file_path.rename(backup_path)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write data based on format
        if format_type == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        elif format_type == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:  # text
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        
        return True
        
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return False

def safe_file_read(file_path: Union[str, Path], format_type: str = "json", 
                  default: Any = None) -> Any:
    """
    Safely read data from file
    
    Args:
        file_path: File path
        format_type: Format type (json, pickle, text)
        default: Default value if file doesn't exist
        
    Returns:
        File data or default
    """
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        return default
    
    try:
        if format_type == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif format_type == "pickle":
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:  # text
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return default

def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of file contents
    
    Args:
        file_path: File path
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hash string
    """
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        return ""
    
    hash_obj = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""

# ==================== PERFORMANCE UTILITIES ====================

class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.duration is not None:
            return self.duration * 1000
        return 0.0

def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time function execution
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, execution_time_ms)
    """
    
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    execution_time_ms = (end_time - start_time) * 1000
    
    return result, execution_time_ms

# ==================== DATA VALIDATION ====================

def validate_numeric_range(value: Union[int, float], min_val: float = None, 
                          max_val: float = None) -> bool:
    """
    Validate numeric value is within range
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        True if valid
    """
    
    try:
        num_value = float(value)
        
        if min_val is not None and num_value < min_val:
            return False
        
        if max_val is not None and num_value > max_val:
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False

def validate_email(email: str) -> bool:
    """
    Basic email validation
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid format
    """
    
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def generate_unique_id(prefix: str = "") -> str:
    """
    Generate unique identifier
    
    Args:
        prefix: Optional prefix
        
    Returns:
        Unique identifier string
    """
    
    unique_part = str(uuid.uuid4()).replace('-', '')
    timestamp = str(int(time.time()))
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_part[:8]}"
    else:
        return f"{timestamp}_{unique_part[:8]}"

# ==================== TESTING ====================

def test_helpers():
    """Test helper functions"""
    
    print("🧪 Testing Helper Functions")
    print("=" * 40)
    
    # Test formatting
    print("💰 Currency formatting:")
    print(f"  {format_currency(1234.56)}")
    print(f"  {format_currency(-1234.56, 'EUR')}")
    
    print("\n📊 Percentage formatting:")
    print(f"  {format_percentage(0.0523)}")
    print(f"  {format_percentage(-0.0234)}")
    
    print("\n🔢 Large number formatting:")
    print(f"  {format_large_number(1234567)}")
    print(f"  {format_large_number(1234567890)}")
    
    # Test table formatting
    print("\n📋 Table formatting:")
    data = [
        ["AAPL", "$150.25", "+2.15%"],
        ["GOOGL", "$2,798.50", "-0.54%"],
        ["MSFT", "$299.80", "+0.25%"]
    ]
    headers = ["Symbol", "Price", "Change"]
    print_table(data, headers, "Stock Prices")
    
    # Test mathematical functions
    print("\n🧮 Mathematical functions:")
    returns = [0.05, -0.02, 0.03, 0.01, -0.01]
    print(f"  Compound return: {format_percentage(calculate_compound_return(returns))}")
    print(f"  Sharpe ratio: {calculate_sharpe_ratio(returns):.3f}")
    
    values = [100, 105, 103, 108, 95, 97, 102]
    print(f"  Max drawdown: {calculate_max_drawdown(values):.2f}%")
    
    # Test timer
    print("\n⏱️  Timer test:")
    with Timer("Test operation") as timer:
        time.sleep(0.1)
    print(f"  Elapsed: {timer.elapsed_ms:.1f}ms")
    
    print("\n🎉 Helper function tests completed!")

if __name__ == "__main__":
    test_helpers()