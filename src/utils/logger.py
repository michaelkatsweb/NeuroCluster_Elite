#!/usr/bin/env python3
"""
File: logger.py
Path: NeuroCluster-Elite/src/utils/logger.py
Description: Advanced logging system for NeuroCluster Elite

This module provides comprehensive logging capabilities including performance tracking,
audit logging, error reporting, and real-time log monitoring.

Features:
- Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Performance logging with timing decorators
- Audit trail for trading operations
- Structured logging with JSON support
- Log rotation and compression
- Real-time log streaming

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import logging
import logging.handlers
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, Union
from functools import wraps
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import traceback

# ==================== ENUMS AND DATA STRUCTURES ====================

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """Log categories for filtering and analysis"""
    SYSTEM = "system"
    ALGORITHM = "algorithm"
    TRADING = "trading"
    DATA = "data"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER = "user"
    API = "api"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    category: LogCategory
    module: str
    message: str
    data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for logging"""
    function_name: str
    execution_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

# ==================== CUSTOM FORMATTERS ====================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'thread_name': record.threadName
        }
        
        # Add extra fields if present
        if hasattr(record, 'category'):
            log_data['category'] = record.category
        
        if hasattr(record, 'data'):
            log_data['data'] = record.data
        
        if hasattr(record, 'execution_time'):
            log_data['execution_time'] = record.execution_time
        
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        if hasattr(record, 'session_id'):
            log_data['session_id'] = record.session_id
        
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, default=str)

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        """Format log record with colors"""
        
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Format the message
        message = record.getMessage()
        
        # Add category if present
        category = getattr(record, 'category', '')
        category_str = f"[{category}] " if category else ""
        
        # Add execution time if present
        exec_time = getattr(record, 'execution_time', None)
        exec_time_str = f" ({exec_time:.3f}ms)" if exec_time else ""
        
        # Combine all parts
        formatted = f"{color}{timestamp} - {record.levelname:8} - {record.name:20} - {category_str}{message}{exec_time_str}{reset}"
        
        # Add exception information if present
        if record.exc_info:
            formatted += f"\n{reset}{traceback.format_exception(*record.exc_info)}"
        
        return formatted

# ==================== ENHANCED LOGGER CLASS ====================

class EnhancedLogger:
    """
    Enhanced logger with performance tracking and structured logging
    """
    
    def __init__(self, name: str, category: LogCategory = LogCategory.SYSTEM):
        self.name = name
        self.category = category
        self.logger = logging.getLogger(name)
        self.performance_metrics = []
        self.session_id = None
        self.user_id = None
    
    def set_context(self, session_id: str = None, user_id: str = None):
        """Set logging context"""
        self.session_id = session_id
        self.user_id = user_id
    
    def _log(self, level: int, message: str, data: Dict[str, Any] = None, 
             execution_time: float = None, correlation_id: str = None):
        """Internal logging method with enhanced data"""
        
        # Create log record
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Add enhanced attributes
        record.category = self.category.value
        if data:
            record.data = data
        if execution_time:
            record.execution_time = execution_time
        if self.session_id:
            record.session_id = self.session_id
        if self.user_id:
            record.user_id = self.user_id
        if correlation_id:
            record.correlation_id = correlation_id
        
        # Handle the record
        self.logger.handle(record)
    
    def debug(self, message: str, data: Dict[str, Any] = None, **kwargs):
        """Debug level logging"""
        self._log(logging.DEBUG, message, data, **kwargs)
    
    def info(self, message: str, data: Dict[str, Any] = None, **kwargs):
        """Info level logging"""
        self._log(logging.INFO, message, data, **kwargs)
    
    def warning(self, message: str, data: Dict[str, Any] = None, **kwargs):
        """Warning level logging"""
        self._log(logging.WARNING, message, data, **kwargs)
    
    def error(self, message: str, data: Dict[str, Any] = None, exc_info: bool = True, **kwargs):
        """Error level logging"""
        if exc_info:
            # Capture current exception info
            import sys
            exc_info = sys.exc_info() if any(sys.exc_info()) else None
        
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.ERROR,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=exc_info
        )
        
        # Add enhanced attributes
        record.category = self.category.value
        if data:
            record.data = data
        if self.session_id:
            record.session_id = self.session_id
        if self.user_id:
            record.user_id = self.user_id
        
        self.logger.handle(record)
    
    def critical(self, message: str, data: Dict[str, Any] = None, **kwargs):
        """Critical level logging"""
        self._log(logging.CRITICAL, message, data, **kwargs)
    
    def performance(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        
        data = asdict(metrics)
        message = f"Performance: {metrics.function_name} took {metrics.execution_time:.3f}ms"
        
        self._log(logging.INFO, message, data, metrics.execution_time)
        self.performance_metrics.append(metrics)
    
    def audit(self, action: str, details: Dict[str, Any] = None, user_id: str = None):
        """Audit logging for security and compliance"""
        
        audit_data = {
            'action': action,
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'user_id': user_id or self.user_id,
            'session_id': self.session_id,
            'details': details or {}
        }
        
        message = f"AUDIT: {action}"
        self._log(logging.INFO, message, audit_data)

# ==================== PERFORMANCE DECORATORS ====================

def log_performance(logger: EnhancedLogger = None, category: LogCategory = LogCategory.PERFORMANCE):
    """
    Decorator to log function performance
    
    Usage:
        @log_performance()
        def my_function():
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error_msg = None
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # Log performance
                if logger:
                    metrics = PerformanceMetrics(
                        function_name=func.__name__,
                        execution_time=execution_time,
                        success=success,
                        error_message=error_msg
                    )
                    logger.performance(metrics)
                else:
                    # Use default logger
                    default_logger = logging.getLogger(func.__module__)
                    default_logger.info(f"Performance: {func.__name__} took {execution_time:.3f}ms")
            
            return result
        
        return wrapper
    return decorator

def log_audit(action: str, logger: EnhancedLogger = None):
    """
    Decorator to log function calls for audit purposes
    
    Usage:
        @log_audit("user_login")
        def login_user(username):
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user context if possible
            user_id = None
            if args and hasattr(args[0], 'user_id'):
                user_id = args[0].user_id
            
            details = {
                'function': func.__name__,
                'module': func.__module__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            # Log audit entry
            if logger:
                logger.audit(action, details, user_id)
            else:
                audit_logger = logging.getLogger('audit')
                audit_logger.info(f"AUDIT: {action} - {func.__name__}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# ==================== LOG SETUP FUNCTIONS ====================

def setup_logging(level: Union[str, int] = logging.INFO, 
                 log_file: str = None,
                 enable_json: bool = False,
                 enable_console_colors: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> Dict[str, logging.Logger]:
    """
    Setup comprehensive logging system
    
    Args:
        level: Logging level
        log_file: Log file path
        enable_json: Enable JSON formatting for files
        enable_console_colors: Enable colored console output
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if enable_console_colors:
        console_formatter = ColoredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)8s - %(name)20s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        
        if enable_json:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)8s - %(name)20s - %(funcName)15s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Create specialized loggers
    loggers = {}
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    if log_file:
        perf_file = log_dir / 'performance.log'
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_file, maxBytes=max_file_size, backupCount=backup_count
        )
        perf_handler.setFormatter(JSONFormatter())
        perf_logger.addHandler(perf_handler)
    loggers['performance'] = perf_logger
    
    # Audit logger
    audit_logger = logging.getLogger('audit')
    if log_file:
        audit_file = log_dir / 'audit.log'
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_file, maxBytes=max_file_size, backupCount=backup_count
        )
        audit_handler.setFormatter(JSONFormatter())
        audit_logger.addHandler(audit_handler)
    loggers['audit'] = audit_logger
    
    # Trading logger
    trading_logger = logging.getLogger('trading')
    if log_file:
        trading_file = log_dir / 'trading.log'
        trading_handler = logging.handlers.RotatingFileHandler(
            trading_file, maxBytes=max_file_size, backupCount=backup_count
        )
        trading_handler.setFormatter(JSONFormatter())
        trading_logger.addHandler(trading_handler)
    loggers['trading'] = trading_logger
    
    # Security logger
    security_logger = logging.getLogger('security')
    if log_file:
        security_file = log_dir / 'security.log'
        security_handler = logging.handlers.RotatingFileHandler(
            security_file, maxBytes=max_file_size, backupCount=backup_count
        )
        security_handler.setFormatter(JSONFormatter())
        security_logger.addHandler(security_handler)
    loggers['security'] = security_logger
    
    logging.info("✅ Logging system initialized")
    
    return loggers

def get_enhanced_logger(name: str, category: LogCategory = LogCategory.SYSTEM) -> EnhancedLogger:
    """
    Get an enhanced logger instance
    
    Args:
        name: Logger name
        category: Log category
        
    Returns:
        Enhanced logger instance
    """
    
    return EnhancedLogger(name, category)

# ==================== LOG ANALYSIS UTILITIES ====================

class LogAnalyzer:
    """Log analysis utilities"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
    
    def analyze_performance(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Analyze performance logs
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Performance analysis results
        """
        
        if not self.log_file.exists():
            return {}
        
        performance_data = []
        cutoff_time = datetime.now().timestamp() - time_window
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        
                        if (log_entry.get('category') == 'performance' and
                            'execution_time' in log_entry and
                            datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00')).timestamp() > cutoff_time):
                            
                            performance_data.append({
                                'function': log_entry.get('data', {}).get('function_name', 'unknown'),
                                'execution_time': log_entry.get('execution_time', 0),
                                'timestamp': log_entry['timestamp']
                            })
                    except (json.JSONDecodeError, ValueError):
                        continue
        
        except Exception as e:
            logging.error(f"Error analyzing performance logs: {e}")
            return {}
        
        if not performance_data:
            return {}
        
        # Calculate statistics
        execution_times = [entry['execution_time'] for entry in performance_data]
        
        analysis = {
            'total_calls': len(performance_data),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'slowest_functions': self._get_slowest_functions(performance_data),
            'time_window': time_window
        }
        
        return analysis
    
    def _get_slowest_functions(self, performance_data: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get the slowest functions"""
        
        function_stats = {}
        
        for entry in performance_data:
            func_name = entry['function']
            exec_time = entry['execution_time']
            
            if func_name not in function_stats:
                function_stats[func_name] = {
                    'name': func_name,
                    'total_time': 0,
                    'call_count': 0,
                    'max_time': 0
                }
            
            stats = function_stats[func_name]
            stats['total_time'] += exec_time
            stats['call_count'] += 1
            stats['max_time'] = max(stats['max_time'], exec_time)
            stats['avg_time'] = stats['total_time'] / stats['call_count']
        
        # Sort by average execution time
        sorted_functions = sorted(
            function_stats.values(),
            key=lambda x: x['avg_time'],
            reverse=True
        )
        
        return sorted_functions[:top_n]

# ==================== REAL-TIME LOG MONITORING ====================

class LogMonitor:
    """Real-time log monitoring"""
    
    def __init__(self, log_file: str, callback: Callable = None):
        self.log_file = Path(log_file)
        self.callback = callback
        self.running = False
        self.thread = None
    
    def start_monitoring(self):
        """Start monitoring log file"""
        
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logging.info(f"Started log monitoring: {self.log_file}")
    
    def stop_monitoring(self):
        """Stop monitoring log file"""
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logging.info("Stopped log monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        
        try:
            with open(self.log_file, 'r') as f:
                # Go to end of file
                f.seek(0, 2)
                
                while self.running:
                    line = f.readline()
                    
                    if line:
                        if self.callback:
                            try:
                                self.callback(line.strip())
                            except Exception as e:
                                logging.error(f"Error in log monitor callback: {e}")
                    else:
                        time.sleep(0.1)  # Small delay when no new data
                        
        except FileNotFoundError:
            logging.warning(f"Log file not found: {self.log_file}")
        except Exception as e:
            logging.error(f"Error in log monitoring: {e}")

# ==================== TESTING ====================

def test_logging_system():
    """Test the logging system"""
    
    print("🧪 Testing Logging System")
    print("=" * 40)
    
    # Setup logging
    loggers = setup_logging(
        level=logging.DEBUG,
        log_file='logs/test.log',
        enable_json=True,
        enable_console_colors=True
    )
    
    # Test enhanced logger
    logger = get_enhanced_logger('test_module', LogCategory.ALGORITHM)
    logger.set_context(session_id='test_session', user_id='test_user')
    
    # Test different log levels
    logger.debug("Debug message", {'key': 'value'})
    logger.info("Info message with performance", execution_time=1.234)
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test performance logging
    @log_performance(logger)
    def test_function():
        time.sleep(0.01)  # Simulate work
        return "success"
    
    result = test_function()
    
    # Test audit logging
    @log_audit("test_action", logger)
    def test_audit_function():
        return "audit_test"
    
    audit_result = test_audit_function()
    
    # Test performance analysis
    analyzer = LogAnalyzer('logs/test.log')
    analysis = analyzer.analyze_performance()
    
    print(f"✅ Performance analysis: {len(analysis)} metrics")
    print("🎉 Logging system tests completed!")

if __name__ == "__main__":
    test_logging_system()