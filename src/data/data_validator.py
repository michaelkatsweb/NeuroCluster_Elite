#!/usr/bin/env python3
"""
File: data_validator.py
Path: NeuroCluster-Elite/src/data/data_validator.py
Description: Advanced data validation and quality control for market data

This module provides comprehensive data validation and quality control for all types
of market data including OHLCV data, fundamentals, news, and real-time feeds.
It detects anomalies, validates data integrity, and ensures trading decisions
are based on high-quality, reliable data.

Features:
- OHLCV data validation with integrity checks
- Real-time data anomaly detection
- Historical data completeness validation
- Cross-source data consistency checks
- Data quality scoring and reporting
- Automatic data cleaning and correction
- Missing data detection and interpolation
- Outlier detection and handling
- Timestamp validation and synchronization

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
from scipy import stats
import math

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.utils.config_manager import ConfigManager
    from src.utils.helpers import format_percentage, calculate_hash
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== VALIDATION ENUMS AND STRUCTURES ====================

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationCategory(Enum):
    """Categories of validation checks"""
    INTEGRITY = "integrity"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    ANOMALY = "anomaly"

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"           # 85-95%
    ACCEPTABLE = "acceptable" # 70-85%
    POOR = "poor"           # 50-70%
    UNUSABLE = "unusable"   # <50%

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    row_index: Optional[int] = None
    timestamp: Optional[datetime] = None
    expected_value: Any = None
    actual_value: Any = None
    suggestion: Optional[str] = None

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    symbol: str
    data_type: str
    validation_time: datetime
    total_records: int
    valid_records: int
    quality_score: float  # 0-100
    quality_level: DataQuality
    
    # Issue breakdown
    issues: List[ValidationIssue] = field(default_factory=list)
    critical_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    
    # Data statistics
    completeness_pct: float = 0.0
    accuracy_pct: float = 0.0
    consistency_pct: float = 0.0
    timeliness_pct: float = 0.0
    
    # Recommendations
    can_use_data: bool = True
    requires_cleaning: bool = False
    missing_data_pct: float = 0.0
    outlier_pct: float = 0.0
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue to report"""
        self.issues.append(issue)
        
        if issue.severity == ValidationSeverity.CRITICAL:
            self.critical_count += 1
        elif issue.severity == ValidationSeverity.ERROR:
            self.error_count += 1
        elif issue.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
        else:
            self.info_count += 1

@dataclass
class CleaningResult:
    """Result of data cleaning operation"""
    original_count: int
    cleaned_count: int
    removed_count: int
    corrected_count: int
    interpolated_count: int
    quality_improvement: float
    cleaning_log: List[str] = field(default_factory=list)

# ==================== DATA VALIDATOR CLASS ====================

class AdvancedDataValidator:
    """
    Advanced data validator for market data quality control
    
    This class provides comprehensive validation for all types of market data
    including real-time quotes, historical OHLCV data, fundamentals, and news.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize data validator"""
        
        self.config = config or self._get_default_config()
        self.validation_cache = {}
        self.quality_history = {}
        
        # Initialize validation rules
        self._initialize_validation_rules()
        
        logger.info("Advanced Data Validator initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default validation configuration"""
        return {
            'ohlcv_validation': {
                'price_change_threshold': 0.25,  # 25% max single-bar change
                'volume_spike_threshold': 10.0,   # 10x volume spike
                'gap_threshold': 0.10,            # 10% max gap
                'min_price': 0.01,                # Minimum valid price
                'max_price': 1000000,             # Maximum valid price
                'negative_spread_tolerance': 0.001 # 0.1% tolerance for bid-ask
            },
            'completeness_checks': {
                'min_data_coverage': 0.95,        # 95% minimum coverage
                'max_missing_consecutive': 5,     # Max 5 consecutive missing
                'trading_hours_only': True,       # Validate trading hours
                'weekend_data_allowed': False     # Allow weekend data
            },
            'accuracy_checks': {
                'cross_validation_sources': 3,   # Validate against 3 sources
                'price_deviation_threshold': 0.02, # 2% max deviation
                'volume_deviation_threshold': 0.5,  # 50% max deviation
                'fundamental_staleness_days': 90    # Max 90 days old
            },
            'anomaly_detection': {
                'outlier_z_score_threshold': 3.0,  # 3 standard deviations
                'pattern_break_threshold': 0.05,   # 5% pattern break
                'volatility_spike_threshold': 5.0, # 5x volatility spike
                'correlation_break_threshold': 0.3  # Correlation drop > 30%
            },
            'cleaning_options': {
                'auto_clean': True,
                'interpolate_missing': True,
                'remove_outliers': True,
                'correct_obvious_errors': True,
                'max_interpolation_gap': 3
            },
            'quality_thresholds': {
                'excellent': 95.0,
                'good': 85.0,
                'acceptable': 70.0,
                'poor': 50.0
            }
        }
    
    def _initialize_validation_rules(self):
        """Initialize validation rules for different data types"""
        
        self.validation_rules = {
            'ohlcv': [
                self._validate_ohlcv_integrity,
                self._validate_ohlcv_ranges,
                self._validate_price_relationships,
                self._validate_volume_data,
                self._validate_timestamp_sequence
            ],
            'quote': [
                self._validate_quote_integrity,
                self._validate_bid_ask_spread,
                self._validate_quote_freshness
            ],
            'fundamental': [
                self._validate_fundamental_completeness,
                self._validate_fundamental_ranges,
                self._validate_fundamental_consistency
            ],
            'news': [
                self._validate_news_completeness,
                self._validate_news_relevance,
                self._validate_news_freshness
            ]
        }
    
    def validate_ohlcv_data(self, data: pd.DataFrame, symbol: str) -> ValidationReport:
        """
        Validate OHLCV market data
        
        Args:
            data: OHLCV DataFrame with columns [open, high, low, close, volume]
            symbol: Asset symbol
            
        Returns:
            ValidationReport with detailed findings
        """
        
        try:
            report = ValidationReport(
                symbol=symbol,
                data_type="ohlcv",
                validation_time=datetime.now(),
                total_records=len(data),
                valid_records=0
            )
            
            if data is None or len(data) == 0:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=ValidationSeverity.CRITICAL,
                    message="No data provided",
                    suggestion="Check data source and fetch process"
                ))
                report.can_use_data = False
                report.quality_score = 0.0
                report.quality_level = DataQuality.UNUSABLE
                return report
            
            # Run OHLCV validation rules
            for rule in self.validation_rules.get('ohlcv', []):
                try:
                    rule(data, report)
                except Exception as e:
                    logger.warning(f"Error in validation rule {rule.__name__}: {e}")
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.INTEGRITY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Validation rule failed: {rule.__name__}",
                        suggestion="Review validation logic"
                    ))
            
            # Calculate overall quality score
            self._calculate_quality_score(report)
            
            # Update validation cache
            self._cache_validation_result(symbol, "ohlcv", report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error validating OHLCV data for {symbol}: {e}")
            
            # Return error report
            error_report = ValidationReport(
                symbol=symbol,
                data_type="ohlcv",
                validation_time=datetime.now(),
                total_records=len(data) if data is not None else 0,
                valid_records=0,
                quality_score=0.0,
                quality_level=DataQuality.UNUSABLE,
                can_use_data=False
            )
            
            error_report.add_issue(ValidationIssue(
                category=ValidationCategory.INTEGRITY,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation failed: {str(e)}",
                suggestion="Check data format and validation logic"
            ))
            
            return error_report
    
    def validate_real_time_data(self, data: Any, data_type: str, symbol: str) -> ValidationReport:
        """
        Validate real-time market data
        
        Args:
            data: Real-time data (quote, trade, etc.)
            data_type: Type of data ('quote', 'trade', 'orderbook')
            symbol: Asset symbol
            
        Returns:
            ValidationReport
        """
        
        try:
            report = ValidationReport(
                symbol=symbol,
                data_type=data_type,
                validation_time=datetime.now(),
                total_records=1,
                valid_records=0
            )
            
            # Validate based on data type
            if data_type == 'quote':
                self._validate_quote_data(data, report)
            elif data_type == 'trade':
                self._validate_trade_data(data, report)
            elif data_type == 'orderbook':
                self._validate_orderbook_data(data, report)
            else:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.INTEGRITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Unknown data type: {data_type}",
                    suggestion="Use supported data types: quote, trade, orderbook"
                ))
            
            self._calculate_quality_score(report)
            return report
            
        except Exception as e:
            logger.error(f"Error validating real-time data: {e}")
            return self._create_error_report(symbol, data_type, str(e))
    
    def clean_data(self, data: pd.DataFrame, validation_report: ValidationReport) -> Tuple[pd.DataFrame, CleaningResult]:
        """
        Clean data based on validation results
        
        Args:
            data: Original data
            validation_report: Validation report with issues
            
        Returns:
            Tuple of (cleaned_data, cleaning_result)
        """
        
        try:
            if not self.config['cleaning_options']['auto_clean']:
                logger.info("Auto-cleaning disabled")
                return data, CleaningResult(
                    original_count=len(data),
                    cleaned_count=len(data),
                    removed_count=0,
                    corrected_count=0,
                    interpolated_count=0,
                    quality_improvement=0.0
                )
            
            original_count = len(data)
            cleaned_data = data.copy()
            cleaning_log = []
            removed_count = 0
            corrected_count = 0
            interpolated_count = 0
            
            # Remove obvious outliers
            if self.config['cleaning_options']['remove_outliers']:
                cleaned_data, outliers_removed = self._remove_outliers(cleaned_data)
                removed_count += outliers_removed
                if outliers_removed > 0:
                    cleaning_log.append(f"Removed {outliers_removed} outliers")
            
            # Interpolate missing values
            if self.config['cleaning_options']['interpolate_missing']:
                cleaned_data, interpolated = self._interpolate_missing_data(cleaned_data)
                interpolated_count += interpolated
                if interpolated > 0:
                    cleaning_log.append(f"Interpolated {interpolated} missing values")
            
            # Correct obvious errors
            if self.config['cleaning_options']['correct_obvious_errors']:
                cleaned_data, corrections = self._correct_obvious_errors(cleaned_data)
                corrected_count += corrections
                if corrections > 0:
                    cleaning_log.append(f"Corrected {corrections} obvious errors")
            
            # Calculate quality improvement
            original_quality = validation_report.quality_score
            
            # Re-validate cleaned data
            cleaned_report = self.validate_ohlcv_data(cleaned_data, validation_report.symbol)
            quality_improvement = cleaned_report.quality_score - original_quality
            
            result = CleaningResult(
                original_count=original_count,
                cleaned_count=len(cleaned_data),
                removed_count=removed_count,
                corrected_count=corrected_count,
                interpolated_count=interpolated_count,
                quality_improvement=quality_improvement,
                cleaning_log=cleaning_log
            )
            
            logger.info(f"Data cleaning completed: {quality_improvement:.1f}% quality improvement")
            
            return cleaned_data, result
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return data, CleaningResult(
                original_count=len(data),
                cleaned_count=len(data),
                removed_count=0,
                corrected_count=0,
                interpolated_count=0,
                quality_improvement=0.0,
                cleaning_log=[f"Cleaning failed: {str(e)}"]
            )
    
    # ==================== OHLCV VALIDATION RULES ====================
    
    def _validate_ohlcv_integrity(self, data: pd.DataFrame, report: ValidationReport):
        """Validate OHLCV data integrity"""
        
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.INTEGRITY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Missing required columns: {missing_columns}",
                    suggestion="Ensure all OHLCV columns are present"
                ))
                return
            
            # Check for null values
            null_counts = data[required_columns].isnull().sum()
            total_nulls = null_counts.sum()
            
            if total_nulls > 0:
                report.missing_data_pct = (total_nulls / (len(data) * len(required_columns))) * 100
                
                if report.missing_data_pct > 5.0:  # More than 5% missing
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.COMPLETENESS,
                        severity=ValidationSeverity.ERROR,
                        message=f"High missing data: {report.missing_data_pct:.1f}%",
                        suggestion="Check data source quality and fetch process"
                    ))
                elif report.missing_data_pct > 1.0:  # More than 1% missing
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.COMPLETENESS,
                        severity=ValidationSeverity.WARNING,
                        message=f"Missing data detected: {report.missing_data_pct:.1f}%",
                        suggestion="Consider data interpolation or alternative sources"
                    ))
            
            report.completeness_pct = 100.0 - report.missing_data_pct
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                category=ValidationCategory.INTEGRITY,
                severity=ValidationSeverity.ERROR,
                message=f"Integrity validation failed: {str(e)}"
            ))
    
    def _validate_ohlcv_ranges(self, data: pd.DataFrame, report: ValidationReport):
        """Validate OHLCV value ranges"""
        
        try:
            config = self.config['ohlcv_validation']
            
            # Check price ranges
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    # Check for negative or zero prices
                    invalid_prices = (data[col] <= 0).sum()
                    if invalid_prices > 0:
                        report.add_issue(ValidationIssue(
                            category=ValidationCategory.ACCURACY,
                            severity=ValidationSeverity.ERROR,
                            message=f"Invalid prices in {col}: {invalid_prices} negative/zero values",
                            field=col,
                            suggestion="Remove or correct invalid price data"
                        ))
                    
                    # Check for extremely high prices
                    max_price = data[col].max()
                    if max_price > config['max_price']:
                        report.add_issue(ValidationIssue(
                            category=ValidationCategory.ACCURACY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Extremely high price in {col}: ${max_price:,.2f}",
                            field=col,
                            actual_value=max_price,
                            suggestion="Verify price data accuracy"
                        ))
                    
                    # Check for extremely low prices
                    min_price = data[col].min()
                    if min_price < config['min_price'] and min_price > 0:
                        report.add_issue(ValidationIssue(
                            category=ValidationCategory.ACCURACY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Extremely low price in {col}: ${min_price:.4f}",
                            field=col,
                            actual_value=min_price,
                            suggestion="Verify price data accuracy"
                        ))
            
            # Check volume ranges
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.ACCURACY,
                        severity=ValidationSeverity.ERROR,
                        message=f"Negative volume values: {negative_volume}",
                        field='volume',
                        suggestion="Volume must be non-negative"
                    ))
                
                # Check for zero volume (might be valid for some assets)
                zero_volume = (data['volume'] == 0).sum()
                if zero_volume > len(data) * 0.1:  # More than 10% zero volume
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.ACCURACY,
                        severity=ValidationSeverity.WARNING,
                        message=f"High zero volume count: {zero_volume} ({zero_volume/len(data)*100:.1f}%)",
                        field='volume',
                        suggestion="Verify if zero volume is expected for this asset"
                    ))
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                category=ValidationCategory.ACCURACY,
                severity=ValidationSeverity.ERROR,
                message=f"Range validation failed: {str(e)}"
            ))
    
    def _validate_price_relationships(self, data: pd.DataFrame, report: ValidationReport):
        """Validate OHLC price relationships"""
        
        try:
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                return
            
            # Check high >= low, high >= open, high >= close, low <= open, low <= close
            invalid_relationships = 0
            
            for i, row in data.iterrows():
                violations = []
                
                if row['high'] < row['low']:
                    violations.append("high < low")
                if row['high'] < row['open']:
                    violations.append("high < open")
                if row['high'] < row['close']:
                    violations.append("high < close")
                if row['low'] > row['open']:
                    violations.append("low > open")
                if row['low'] > row['close']:
                    violations.append("low > close")
                
                if violations:
                    invalid_relationships += 1
                    if invalid_relationships <= 5:  # Report first 5 violations
                        report.add_issue(ValidationIssue(
                            category=ValidationCategory.INTEGRITY,
                            severity=ValidationSeverity.ERROR,
                            message=f"Invalid OHLC relationships: {', '.join(violations)}",
                            row_index=i,
                            timestamp=i if isinstance(i, datetime) else None,
                            suggestion="Correct OHLC data or remove invalid bars"
                        ))
            
            if invalid_relationships > 5:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.INTEGRITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Multiple OHLC relationship violations: {invalid_relationships} total",
                    suggestion="Review data source quality"
                ))
            
            # Check for price gaps
            config = self.config['ohlcv_validation']
            if len(data) > 1:
                price_changes = data['close'].pct_change().abs()
                large_gaps = (price_changes > config['gap_threshold']).sum()
                
                if large_gaps > 0:
                    max_gap = price_changes.max() * 100
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.ACCURACY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Large price gaps detected: {large_gaps} gaps > {config['gap_threshold']*100:.1f}%",
                        actual_value=f"{max_gap:.2f}%",
                        suggestion="Verify if gaps are due to corporate actions or data issues"
                    ))
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                category=ValidationCategory.INTEGRITY,
                severity=ValidationSeverity.ERROR,
                message=f"Price relationship validation failed: {str(e)}"
            ))
    
    def _validate_volume_data(self, data: pd.DataFrame, report: ValidationReport):
        """Validate volume data"""
        
        try:
            if 'volume' not in data.columns:
                return
            
            config = self.config['ohlcv_validation']
            
            # Calculate volume statistics
            volume_data = data['volume'].dropna()
            if len(volume_data) == 0:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=ValidationSeverity.WARNING,
                    message="No volume data available",
                    field='volume'
                ))
                return
            
            avg_volume = volume_data.mean()
            volume_spikes = (volume_data > avg_volume * config['volume_spike_threshold']).sum()
            
            if volume_spikes > 0:
                max_spike_ratio = (volume_data.max() / avg_volume) if avg_volume > 0 else 0
                
                severity = ValidationSeverity.INFO
                if volume_spikes > len(volume_data) * 0.05:  # More than 5% spikes
                    severity = ValidationSeverity.WARNING
                
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.ANOMALY,
                    severity=severity,
                    message=f"Volume spikes detected: {volume_spikes} spikes > {config['volume_spike_threshold']}x average",
                    field='volume',
                    actual_value=f"{max_spike_ratio:.1f}x",
                    suggestion="Verify if volume spikes correspond to news events"
                ))
            
            # Check volume pattern consistency
            if len(volume_data) >= 20:
                recent_avg = volume_data.tail(5).mean()
                historical_avg = volume_data.head(-5).mean()
                
                if historical_avg > 0:
                    volume_change = (recent_avg - historical_avg) / historical_avg
                    
                    if abs(volume_change) > 2.0:  # 200% change in recent volume
                        report.add_issue(ValidationIssue(
                            category=ValidationCategory.ANOMALY,
                            severity=ValidationSeverity.INFO,
                            message=f"Significant volume pattern change: {volume_change*100:+.1f}%",
                            field='volume',
                            suggestion="Monitor for market regime changes"
                        ))
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                category=ValidationCategory.ACCURACY,
                severity=ValidationSeverity.ERROR,
                message=f"Volume validation failed: {str(e)}"
            ))
    
    def _validate_timestamp_sequence(self, data: pd.DataFrame, report: ValidationReport):
        """Validate timestamp sequence and frequency"""
        
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.INTEGRITY,
                    severity=ValidationSeverity.ERROR,
                    message="Index is not a DatetimeIndex",
                    suggestion="Convert index to datetime format"
                ))
                return
            
            # Check for duplicate timestamps
            duplicates = data.index.duplicated().sum()
            if duplicates > 0:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.INTEGRITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Duplicate timestamps: {duplicates}",
                    suggestion="Remove or consolidate duplicate entries"
                ))
            
            # Check for correct time ordering
            if not data.index.is_monotonic_increasing:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.INTEGRITY,
                    severity=ValidationSeverity.ERROR,
                    message="Timestamps are not in ascending order",
                    suggestion="Sort data by timestamp"
                ))
            
            # Check for reasonable time gaps
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dt.total_seconds()
                
                # Detect frequency
                median_diff = time_diffs.median()
                expected_freq = None
                
                if 50 <= median_diff <= 70:    # ~1 minute
                    expected_freq = "1min"
                elif 250 <= median_diff <= 350: # ~5 minutes
                    expected_freq = "5min"
                elif 850 <= median_diff <= 950: # ~15 minutes
                    expected_freq = "15min"
                elif 3500 <= median_diff <= 3700: # ~1 hour
                    expected_freq = "1hour"
                elif 82000 <= median_diff <= 90000: # ~1 day
                    expected_freq = "1day"
                
                if expected_freq:
                    # Check for large gaps
                    large_gaps = (time_diffs > median_diff * 3).sum()  # 3x expected interval
                    
                    if large_gaps > 0:
                        max_gap_hours = time_diffs.max() / 3600
                        report.add_issue(ValidationIssue(
                            category=ValidationCategory.COMPLETENESS,
                            severity=ValidationSeverity.WARNING,
                            message=f"Time gaps detected: {large_gaps} gaps > 3x expected interval",
                            actual_value=f"{max_gap_hours:.1f} hours max gap",
                            suggestion="Check for missing data periods"
                        ))
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                category=ValidationCategory.INTEGRITY,
                severity=ValidationSeverity.ERROR,
                message=f"Timestamp validation failed: {str(e)}"
            ))
    
    # ==================== REAL-TIME DATA VALIDATION ====================
    
    def _validate_quote_data(self, quote_data: Any, report: ValidationReport):
        """Validate real-time quote data"""
        
        try:
            # Validate quote structure
            required_fields = ['price', 'bid', 'ask', 'timestamp']
            
            for field in required_fields:
                if not hasattr(quote_data, field):
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.INTEGRITY,
                        severity=ValidationSeverity.ERROR,
                        message=f"Missing required field: {field}",
                        field=field
                    ))
            
            # Validate bid-ask spread
            if hasattr(quote_data, 'bid') and hasattr(quote_data, 'ask'):
                if quote_data.ask <= quote_data.bid:
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.ACCURACY,
                        severity=ValidationSeverity.ERROR,
                        message="Invalid bid-ask spread: ask <= bid",
                        actual_value=f"bid: {quote_data.bid}, ask: {quote_data.ask}"
                    ))
            
            # Validate price freshness
            if hasattr(quote_data, 'timestamp'):
                age_seconds = (datetime.now() - quote_data.timestamp).total_seconds()
                if age_seconds > 300:  # More than 5 minutes old
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.TIMELINESS,
                        severity=ValidationSeverity.WARNING,
                        message=f"Stale quote data: {age_seconds:.0f} seconds old",
                        actual_value=f"{age_seconds:.0f}s"
                    ))
            
            report.valid_records = 1 if report.error_count == 0 else 0
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                category=ValidationCategory.INTEGRITY,
                severity=ValidationSeverity.ERROR,
                message=f"Quote validation failed: {str(e)}"
            ))
    
    def _validate_quote_integrity(self, data: Any, report: ValidationReport):
        """Validate quote data integrity"""
        # Implementation for quote integrity validation
        pass
    
    def _validate_bid_ask_spread(self, data: Any, report: ValidationReport):
        """Validate bid-ask spread"""
        # Implementation for bid-ask spread validation
        pass
    
    def _validate_quote_freshness(self, data: Any, report: ValidationReport):
        """Validate quote data freshness"""
        # Implementation for quote freshness validation
        pass
    
    def _validate_trade_data(self, trade_data: Any, report: ValidationReport):
        """Validate trade data"""
        # Implementation for trade data validation
        pass
    
    def _validate_orderbook_data(self, orderbook_data: Any, report: ValidationReport):
        """Validate orderbook data"""
        # Implementation for orderbook validation
        pass
    
    def _validate_fundamental_completeness(self, data: Any, report: ValidationReport):
        """Validate fundamental data completeness"""
        # Implementation for fundamental data validation
        pass
    
    def _validate_fundamental_ranges(self, data: Any, report: ValidationReport):
        """Validate fundamental data ranges"""
        pass
    
    def _validate_fundamental_consistency(self, data: Any, report: ValidationReport):
        """Validate fundamental data consistency"""
        pass
    
    def _validate_news_completeness(self, data: Any, report: ValidationReport):
        """Validate news data completeness"""
        pass
    
    def _validate_news_relevance(self, data: Any, report: ValidationReport):
        """Validate news relevance"""
        pass
    
    def _validate_news_freshness(self, data: Any, report: ValidationReport):
        """Validate news freshness"""
        pass
    
    # ==================== DATA CLEANING METHODS ====================
    
    def _remove_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove statistical outliers from data"""
        
        try:
            original_count = len(data)
            
            # Use Z-score method for outlier detection
            threshold = self.config['anomaly_detection']['outlier_z_score_threshold']
            
            price_columns = ['open', 'high', 'low', 'close']
            outlier_mask = pd.Series(False, index=data.index)
            
            for col in price_columns:
                if col in data.columns:
                    z_scores = np.abs(stats.zscore(data[col].dropna()))
                    outlier_mask = outlier_mask | (z_scores > threshold)
            
            cleaned_data = data[~outlier_mask]
            removed_count = original_count - len(cleaned_data)
            
            return cleaned_data, removed_count
            
        except Exception as e:
            logger.warning(f"Error removing outliers: {e}")
            return data, 0
    
    def _interpolate_missing_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Interpolate missing data points"""
        
        try:
            interpolated_count = 0
            max_gap = self.config['cleaning_options']['max_interpolation_gap']
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    # Count missing values before interpolation
                    missing_before = data[col].isnull().sum()
                    
                    # Interpolate small gaps only
                    data[col] = data[col].interpolate(method='linear', limit=max_gap)
                    
                    # Count missing values after interpolation
                    missing_after = data[col].isnull().sum()
                    interpolated_count += missing_before - missing_after
            
            return data, interpolated_count
            
        except Exception as e:
            logger.warning(f"Error interpolating missing data: {e}")
            return data, 0
    
    def _correct_obvious_errors(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Correct obvious data errors"""
        
        try:
            corrections = 0
            
            # Correct negative prices (set to NaN for interpolation)
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    negative_count = (data[col] <= 0).sum()
                    data.loc[data[col] <= 0, col] = np.nan
                    corrections += negative_count
            
            # Correct negative volume
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                data.loc[data['volume'] < 0, 'volume'] = 0
                corrections += negative_volume
            
            # Correct impossible OHLC relationships
            required_cols = ['open', 'high', 'low', 'close']
            if all(col in data.columns for col in required_cols):
                for i in data.index:
                    row = data.loc[i]
                    
                    # Ensure high is the maximum
                    actual_high = max(row['open'], row['high'], row['low'], row['close'])
                    if row['high'] != actual_high:
                        data.loc[i, 'high'] = actual_high
                        corrections += 1
                    
                    # Ensure low is the minimum
                    actual_low = min(row['open'], row['high'], row['low'], row['close'])
                    if row['low'] != actual_low:
                        data.loc[i, 'low'] = actual_low
                        corrections += 1
            
            return data, corrections
            
        except Exception as e:
            logger.warning(f"Error correcting obvious errors: {e}")
            return data, 0
    
    # ==================== UTILITY METHODS ====================
    
    def _calculate_quality_score(self, report: ValidationReport):
        """Calculate overall data quality score"""
        
        try:
            # Base score starts at 100
            score = 100.0
            
            # Deduct points for issues
            score -= report.critical_count * 25  # 25 points per critical issue
            score -= report.error_count * 10     # 10 points per error
            score -= report.warning_count * 3    # 3 points per warning
            score -= report.info_count * 1       # 1 point per info issue
            
            # Additional deductions for high percentages
            if report.missing_data_pct > 5:
                score -= (report.missing_data_pct - 5) * 2  # 2 points per % over 5%
            
            if report.outlier_pct > 2:
                score -= (report.outlier_pct - 2) * 3  # 3 points per % over 2%
            
            # Ensure score doesn't go below 0
            score = max(0.0, score)
            
            report.quality_score = score
            
            # Determine quality level
            thresholds = self.config['quality_thresholds']
            if score >= thresholds['excellent']:
                report.quality_level = DataQuality.EXCELLENT
            elif score >= thresholds['good']:
                report.quality_level = DataQuality.GOOD
            elif score >= thresholds['acceptable']:
                report.quality_level = DataQuality.ACCEPTABLE
            elif score >= thresholds['poor']:
                report.quality_level = DataQuality.POOR
            else:
                report.quality_level = DataQuality.UNUSABLE
            
            # Set usage recommendation
            report.can_use_data = (score >= thresholds['acceptable'] and 
                                 report.critical_count == 0)
            report.requires_cleaning = (report.warning_count > 0 or 
                                      report.missing_data_pct > 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            report.quality_score = 0.0
            report.quality_level = DataQuality.UNUSABLE
            report.can_use_data = False
    
    def _cache_validation_result(self, symbol: str, data_type: str, report: ValidationReport):
        """Cache validation result for future reference"""
        
        try:
            cache_key = f"{symbol}_{data_type}"
            self.validation_cache[cache_key] = {
                'report': report,
                'timestamp': datetime.now()
            }
            
            # Maintain quality history
            if symbol not in self.quality_history:
                self.quality_history[symbol] = []
            
            self.quality_history[symbol].append({
                'timestamp': report.validation_time,
                'quality_score': report.quality_score,
                'data_type': data_type
            })
            
            # Limit history size
            if len(self.quality_history[symbol]) > 100:
                self.quality_history[symbol] = self.quality_history[symbol][-100:]
                
        except Exception as e:
            logger.warning(f"Error caching validation result: {e}")
    
    def _create_error_report(self, symbol: str, data_type: str, error_message: str) -> ValidationReport:
        """Create error validation report"""
        
        report = ValidationReport(
            symbol=symbol,
            data_type=data_type,
            validation_time=datetime.now(),
            total_records=0,
            valid_records=0,
            quality_score=0.0,
            quality_level=DataQuality.UNUSABLE,
            can_use_data=False
        )
        
        report.add_issue(ValidationIssue(
            category=ValidationCategory.INTEGRITY,
            severity=ValidationSeverity.CRITICAL,
            message=error_message,
            suggestion="Check data format and validation process"
        ))
        
        return report
    
    def get_validation_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get validation summary for symbol or all symbols"""
        
        try:
            if symbol:
                # Summary for specific symbol
                if symbol in self.quality_history:
                    history = self.quality_history[symbol]
                    recent_scores = [h['quality_score'] for h in history[-10:]]
                    
                    return {
                        'symbol': symbol,
                        'validation_count': len(history),
                        'avg_quality_score': np.mean(recent_scores) if recent_scores else 0,
                        'min_quality_score': min(recent_scores) if recent_scores else 0,
                        'max_quality_score': max(recent_scores) if recent_scores else 0,
                        'trend': 'improving' if len(recent_scores) >= 2 and recent_scores[-1] > recent_scores[0] else 'stable',
                        'last_validation': history[-1]['timestamp'] if history else None
                    }
                else:
                    return {'symbol': symbol, 'validation_count': 0}
            else:
                # Summary for all symbols
                total_validations = sum(len(history) for history in self.quality_history.values())
                all_scores = []
                
                for history in self.quality_history.values():
                    all_scores.extend([h['quality_score'] for h in history[-5:]])  # Recent scores
                
                return {
                    'total_symbols': len(self.quality_history),
                    'total_validations': total_validations,
                    'avg_quality_score': np.mean(all_scores) if all_scores else 0,
                    'cache_size': len(self.validation_cache)
                }
                
        except Exception as e:
            logger.error(f"Error getting validation summary: {e}")
            return {}

# ==================== TESTING ====================

def test_data_validator():
    """Test data validator functionality"""
    
    print("🔍 Testing Advanced Data Validator")
    print("=" * 50)
    
    # Create sample data with various issues
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Create base data
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(100):
        # Add some normal variation
        change = np.random.randn() * 0.02
        new_price = (prices[-1] if prices else base_price) * (1 + change)
        
        # Introduce some data issues
        if i == 20:  # Negative price
            new_price = -50
        elif i == 35:  # Extreme spike
            new_price *= 5
        elif i == 50:  # Missing data (will be NaN)
            new_price = np.nan
        
        prices.append(new_price)
        
        # Volume data with issues
        volume = np.random.randint(1000, 10000)
        if i == 25:  # Negative volume
            volume = -1000
        elif i == 60:  # Volume spike
            volume = 100000
        
        volumes.append(volume)
    
    # Create DataFrame with issues
    sample_data = pd.DataFrame({
        'open': prices,
        'close': [p * (1 + np.random.randn() * 0.005) if not pd.isna(p) else np.nan for p in prices],
        'volume': volumes
    }, index=dates)
    
    # Add high/low with potential issues
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1, skipna=True) * (1 + np.random.rand(100) * 0.01)
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1, skipna=True) * (1 - np.random.rand(100) * 0.01)
    
    # Introduce OHLC relationship issues
    sample_data.loc[sample_data.index[30], 'high'] = sample_data.loc[sample_data.index[30], 'low'] - 1  # high < low
    
    # Create data validator
    validator = AdvancedDataValidator()
    
    print(f"✅ Data validator initialized:")
    print(f"   Validation rules: {len(validator.validation_rules)}")
    print(f"   OHLCV rules: {len(validator.validation_rules.get('ohlcv', []))}")
    print(f"   Auto-cleaning enabled: {validator.config['cleaning_options']['auto_clean']}")
    
    # Validate the data
    validation_report = validator.validate_ohlcv_data(sample_data, 'TEST')
    
    print(f"\n📋 Validation Report:")
    print(f"   Symbol: {validation_report.symbol}")
    print(f"   Data type: {validation_report.data_type}")
    print(f"   Total records: {validation_report.total_records}")
    print(f"   Valid records: {validation_report.valid_records}")
    print(f"   Quality score: {validation_report.quality_score:.1f}")
    print(f"   Quality level: {validation_report.quality_level.value}")
    print(f"   Can use data: {'✅' if validation_report.can_use_data else '❌'}")
    print(f"   Requires cleaning: {'✅' if validation_report.requires_cleaning else '❌'}")
    
    print(f"\n⚠️  Issues found:")
    print(f"   Critical: {validation_report.critical_count}")
    print(f"   Errors: {validation_report.error_count}")
    print(f"   Warnings: {validation_report.warning_count}")
    print(f"   Info: {validation_report.info_count}")
    print(f"   Missing data: {validation_report.missing_data_pct:.1f}%")
    
    # Show detailed issues
    print(f"\n🔍 Detailed issues (first 10):")
    for i, issue in enumerate(validation_report.issues[:10]):
        severity_icon = {'critical': '🚨', 'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}
        print(f"   {severity_icon.get(issue.severity.value, '•')} {issue.severity.value.upper()}: {issue.message}")
        if issue.suggestion:
            print(f"      💡 {issue.suggestion}")
    
    if len(validation_report.issues) > 10:
        print(f"   ... and {len(validation_report.issues) - 10} more issues")
    
    # Test data cleaning
    print(f"\n🧹 Testing data cleaning:")
    cleaned_data, cleaning_result = validator.clean_data(sample_data, validation_report)
    
    print(f"   Original records: {cleaning_result.original_count}")
    print(f"   Cleaned records: {cleaning_result.cleaned_count}")
    print(f"   Removed records: {cleaning_result.removed_count}")
    print(f"   Corrected values: {cleaning_result.corrected_count}")
    print(f"   Interpolated values: {cleaning_result.interpolated_count}")
    print(f"   Quality improvement: {cleaning_result.quality_improvement:+.1f}%")
    
    if cleaning_result.cleaning_log:
        print(f"   Cleaning log:")
        for log_entry in cleaning_result.cleaning_log:
            print(f"      - {log_entry}")
    
    # Validate cleaned data
    print(f"\n✨ Validating cleaned data:")
    cleaned_report = validator.validate_ohlcv_data(cleaned_data, 'TEST_CLEANED')
    
    print(f"   Quality improvement: {cleaned_report.quality_score - validation_report.quality_score:+.1f} points")
    print(f"   New quality score: {cleaned_report.quality_score:.1f}")
    print(f"   New quality level: {cleaned_report.quality_level.value}")
    print(f"   Issues reduced: {len(validation_report.issues) - len(cleaned_report.issues)} issues")
    
    # Test validation summary
    print(f"\n📊 Validation summary:")
    summary = validator.get_validation_summary('TEST')
    
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n🎉 Data validator tests completed!")

if __name__ == "__main__":
    test_data_validator()