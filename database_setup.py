#!/usr/bin/env python3
"""
File: database_setup.py
Path: NeuroCluster-Elite/database_setup.py
Description: Database initialization and management for NeuroCluster Elite

This script handles all database initialization, schema creation, data migration,
and database management tasks for the NeuroCluster Elite trading platform.
It supports multiple database backends and provides comprehensive data management.

Features:
- Multi-database support (SQLite, PostgreSQL, MySQL)
- Automatic schema creation and migration
- Data seeding and sample data generation
- Database backup and restore functionality
- Performance optimization and indexing
- Data integrity validation
- Migration rollback capabilities
- Database health monitoring
- Connection pool management
- Transaction management

Database Components:
- Trading data (trades, orders, positions)
- Portfolio management (holdings, performance)
- Market data (prices, indicators, signals)
- User data (settings, preferences)
- System data (logs, configuration)
- Analytics data (backtests, metrics)

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import os
import sys
import sqlite3
import logging
import json
import yaml
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import shutil

# Database libraries
try:
    import sqlalchemy
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Index
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text
    from sqlalchemy.exc import SQLAlchemyError
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

try:
    import psycopg2
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False

try:
    import pymysql
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== ENUMS AND DATA STRUCTURES ====================

class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"

class MigrationStatus(Enum):
    """Migration status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: DatabaseType = DatabaseType.SQLITE
    host: str = "localhost"
    port: int = 5432
    database: str = "neurocluster_elite"
    username: str = "neurocluster"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    echo: bool = False
    sqlite_path: str = "data/databases/neurocluster_elite.db"

@dataclass
class Migration:
    """Database migration definition"""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

# ==================== DATABASE MANAGER ====================

class DatabaseManager:
    """Main database manager class"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.engine: Optional[Engine] = None
        self.metadata = MetaData()
        self.root_path = Path.cwd()
        self.migrations_dir = self.root_path / "data" / "migrations"
        self.backups_dir = self.root_path / "data" / "backups"
        
        # Ensure directories exist
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> bool:
        """Initialize database and create all tables"""
        
        logger.info("🗄️ Initializing NeuroCluster Elite database...")
        
        try:
            # Create database engine
            self.engine = self._create_engine()
            
            # Create database if it doesn't exist
            if not self._database_exists():
                self._create_database()
            
            # Create all tables
            self._create_tables()
            
            # Run migrations
            self._run_migrations()
            
            # Seed initial data
            self._seed_initial_data()
            
            # Validate database
            if self._validate_database():
                logger.info("✅ Database initialization completed successfully")
                return True
            else:
                logger.error("❌ Database validation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    def _create_engine(self) -> Engine:
        """Create database engine based on configuration"""
        
        if self.config.db_type == DatabaseType.SQLITE:
            # Ensure SQLite directory exists
            sqlite_path = Path(self.config.sqlite_path)
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            
            connection_string = f"sqlite:///{sqlite_path}"
            
            engine = sqlalchemy.create_engine(
                connection_string,
                echo=self.config.echo,
                pool_timeout=self.config.pool_timeout
            )
        
        elif self.config.db_type == DatabaseType.POSTGRESQL:
            if not HAS_POSTGRESQL:
                raise Exception("PostgreSQL support not available. Install psycopg2-binary.")
            
            connection_string = (
                f"postgresql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/{self.config.database}"
            )
            
            engine = sqlalchemy.create_engine(
                connection_string,
                echo=self.config.echo,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout
            )
        
        elif self.config.db_type == DatabaseType.MYSQL:
            if not HAS_MYSQL:
                raise Exception("MySQL support not available. Install PyMySQL.")
            
            connection_string = (
                f"mysql+pymysql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/{self.config.database}"
            )
            
            engine = sqlalchemy.create_engine(
                connection_string,
                echo=self.config.echo,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout
            )
        
        else:
            raise Exception(f"Unsupported database type: {self.config.db_type}")
        
        logger.info(f"🔗 Created {self.config.db_type.value} database engine")
        return engine
    
    def _database_exists(self) -> bool:
        """Check if database exists"""
        
        if self.config.db_type == DatabaseType.SQLITE:
            return Path(self.config.sqlite_path).exists()
        
        elif self.config.db_type == DatabaseType.POSTGRESQL:
            try:
                # Connect to postgres database to check if target database exists
                admin_engine = sqlalchemy.create_engine(
                    f"postgresql://{self.config.username}:{self.config.password}@"
                    f"{self.config.host}:{self.config.port}/postgres"
                )
                
                with admin_engine.connect() as conn:
                    result = conn.execute(text(
                        "SELECT 1 FROM pg_database WHERE datname = :db_name"
                    ), {"db_name": self.config.database})
                    
                    return result.fetchone() is not None
                    
            except Exception:
                return False
        
        elif self.config.db_type == DatabaseType.MYSQL:
            try:
                # Connect to MySQL to check if database exists
                admin_engine = sqlalchemy.create_engine(
                    f"mysql+pymysql://{self.config.username}:{self.config.password}@"
                    f"{self.config.host}:{self.config.port}/"
                )
                
                with admin_engine.connect() as conn:
                    result = conn.execute(text(
                        "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = :db_name"
                    ), {"db_name": self.config.database})
                    
                    return result.fetchone() is not None
                    
            except Exception:
                return False
        
        return True
    
    def _create_database(self):
        """Create database if it doesn't exist"""
        
        if self.config.db_type == DatabaseType.SQLITE:
            # SQLite creates database automatically
            return
        
        elif self.config.db_type == DatabaseType.POSTGRESQL:
            logger.info(f"📦 Creating PostgreSQL database: {self.config.database}")
            
            admin_engine = sqlalchemy.create_engine(
                f"postgresql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/postgres"
            )
            
            with admin_engine.connect() as conn:
                conn.execute(text("COMMIT"))  # End any existing transaction
                conn.execute(text(f"CREATE DATABASE {self.config.database}"))
        
        elif self.config.db_type == DatabaseType.MYSQL:
            logger.info(f"📦 Creating MySQL database: {self.config.database}")
            
            admin_engine = sqlalchemy.create_engine(
                f"mysql+pymysql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/"
            )
            
            with admin_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE {self.config.database}"))
    
    def _create_tables(self):
        """Create all database tables"""
        
        logger.info("📋 Creating database tables...")
        
        # Define all tables
        tables = self._define_tables()
        
        # Create tables
        self.metadata.create_all(self.engine)
        
        logger.info(f"✅ Created {len(tables)} database tables")
    
    def _define_tables(self) -> List[Table]:
        """Define all database table schemas"""
        
        tables = []
        
        # System metadata table
        system_metadata = Table(
            'system_metadata',
            self.metadata,
            Column('key', String(255), primary_key=True),
            Column('value', Text),
            Column('description', Text),
            Column('created_at', DateTime, default=datetime.now),
            Column('updated_at', DateTime, default=datetime.now, onupdate=datetime.now)
        )
        tables.append(system_metadata)
        
        # Migration tracking table
        migrations = Table(
            'migrations',
            self.metadata,
            Column('version', String(50), primary_key=True),
            Column('name', String(255), nullable=False),
            Column('description', Text),
            Column('status', String(20), default='pending'),
            Column('applied_at', DateTime),
            Column('rolled_back_at', DateTime),
            Column('checksum', String(64))
        )
        tables.append(migrations)
        
        # Market data table
        market_data = Table(
            'market_data',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(20), nullable=False),
            Column('asset_type', String(20), nullable=False),
            Column('timestamp', DateTime, nullable=False),
            Column('open_price', Float),
            Column('high_price', Float),
            Column('low_price', Float),
            Column('close_price', Float),
            Column('volume', Float),
            Column('adj_close', Float),
            Index('idx_market_data_symbol_timestamp', 'symbol', 'timestamp'),
            Index('idx_market_data_timestamp', 'timestamp')
        )
        tables.append(market_data)
        
        # Technical indicators table
        technical_indicators = Table(
            'technical_indicators',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(20), nullable=False),
            Column('timestamp', DateTime, nullable=False),
            Column('rsi', Float),
            Column('macd', Float),
            Column('macd_signal', Float),
            Column('bollinger_upper', Float),
            Column('bollinger_lower', Float),
            Column('sma_20', Float),
            Column('ema_50', Float),
            Column('volatility', Float),
            Index('idx_technical_indicators_symbol_timestamp', 'symbol', 'timestamp')
        )
        tables.append(technical_indicators)
        
        # Trading signals table
        trading_signals = Table(
            'trading_signals',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(20), nullable=False),
            Column('signal_type', String(20), nullable=False),
            Column('confidence', Float, nullable=False),
            Column('regime', String(50)),
            Column('entry_price', Float),
            Column('stop_loss', Float),
            Column('take_profit', Float),
            Column('reasoning', Text),
            Column('timestamp', DateTime, default=datetime.now),
            Index('idx_trading_signals_symbol', 'symbol'),
            Index('idx_trading_signals_timestamp', 'timestamp')
        )
        tables.append(trading_signals)
        
        # Trades table
        trades = Table(
            'trades',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('trade_id', String(50), unique=True, nullable=False),
            Column('symbol', String(20), nullable=False),
            Column('side', String(10), nullable=False),  # BUY, SELL
            Column('quantity', Float, nullable=False),
            Column('price', Float, nullable=False),
            Column('order_type', String(20), default='MARKET'),
            Column('status', String(20), default='PENDING'),
            Column('commission', Float, default=0.0),
            Column('slippage', Float, default=0.0),
            Column('pnl', Float),
            Column('strategy', String(50)),
            Column('notes', Text),
            Column('created_at', DateTime, default=datetime.now),
            Column('executed_at', DateTime),
            Index('idx_trades_symbol', 'symbol'),
            Index('idx_trades_created_at', 'created_at')
        )
        tables.append(trades)
        
        # Positions table
        positions = Table(
            'positions',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(20), nullable=False),
            Column('asset_type', String(20), nullable=False),
            Column('quantity', Float, nullable=False),
            Column('entry_price', Float, nullable=False),
            Column('current_price', Float),
            Column('market_value', Float),
            Column('unrealized_pnl', Float),
            Column('realized_pnl', Float, default=0.0),
            Column('stop_loss', Float),
            Column('take_profit', Float),
            Column('opened_at', DateTime, default=datetime.now),
            Column('closed_at', DateTime),
            Column('strategy', String(50)),
            Index('idx_positions_symbol', 'symbol'),
            Index('idx_positions_opened_at', 'opened_at')
        )
        tables.append(positions)
        
        # Portfolio snapshots table
        portfolio_snapshots = Table(
            'portfolio_snapshots',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('timestamp', DateTime, default=datetime.now),
            Column('total_value', Float, nullable=False),
            Column('cash_balance', Float, nullable=False),
            Column('invested_amount', Float, nullable=False),
            Column('unrealized_pnl', Float, nullable=False),
            Column('realized_pnl', Float, nullable=False),
            Column('daily_pnl', Float),
            Column('total_return_pct', Float),
            Column('position_count', Integer),
            Column('max_drawdown', Float),
            Index('idx_portfolio_snapshots_timestamp', 'timestamp')
        )
        tables.append(portfolio_snapshots)
        
        # Algorithm performance table
        algorithm_performance = Table(
            'algorithm_performance',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('timestamp', DateTime, default=datetime.now),
            Column('efficiency', Float),
            Column('processing_time_ms', Float),
            Column('accuracy', Float),
            Column('memory_usage_mb', Float),
            Column('clustering_quality', Float),
            Column('regime_changes', Integer),
            Column('signals_generated', Integer),
            Index('idx_algorithm_performance_timestamp', 'timestamp')
        )
        tables.append(algorithm_performance)
        
        # Backtests table
        backtests = Table(
            'backtests',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String(255), nullable=False),
            Column('strategy', String(100), nullable=False),
            Column('start_date', DateTime, nullable=False),
            Column('end_date', DateTime, nullable=False),
            Column('initial_capital', Float, nullable=False),
            Column('final_value', Float),
            Column('total_return', Float),
            Column('sharpe_ratio', Float),
            Column('max_drawdown', Float),
            Column('win_rate', Float),
            Column('total_trades', Integer),
            Column('config', Text),  # JSON configuration
            Column('results', Text),  # JSON results
            Column('created_at', DateTime, default=datetime.now),
            Index('idx_backtests_strategy', 'strategy'),
            Index('idx_backtests_created_at', 'created_at')
        )
        tables.append(backtests)
        
        # News and sentiment table
        news_sentiment = Table(
            'news_sentiment',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(20)),
            Column('title', Text, nullable=False),
            Column('content', Text),
            Column('source', String(100)),
            Column('url', Text),
            Column('sentiment_score', Float),
            Column('relevance_score', Float),
            Column('published_at', DateTime),
            Column('processed_at', DateTime, default=datetime.now),
            Index('idx_news_sentiment_symbol', 'symbol'),
            Index('idx_news_sentiment_published_at', 'published_at')
        )
        tables.append(news_sentiment)
        
        # User settings table
        user_settings = Table(
            'user_settings',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('user_id', String(50), nullable=False),
            Column('setting_key', String(100), nullable=False),
            Column('setting_value', Text),
            Column('setting_type', String(20), default='string'),
            Column('description', Text),
            Column('updated_at', DateTime, default=datetime.now, onupdate=datetime.now),
            Index('idx_user_settings_user_id', 'user_id'),
            Index('idx_user_settings_key', 'setting_key')
        )
        tables.append(user_settings)
        
        return tables
    
    def _run_migrations(self):
        """Run database migrations"""
        
        logger.info("🔄 Running database migrations...")
        
        # Create migrations table if it doesn't exist (should already exist from _create_tables)
        
        # Get existing migrations
        applied_migrations = self._get_applied_migrations()
        
        # Get available migrations
        available_migrations = self._get_available_migrations()
        
        # Run pending migrations
        pending_migrations = [m for m in available_migrations if m.version not in applied_migrations]
        
        if not pending_migrations:
            logger.info("✅ No pending migrations")
            return
        
        logger.info(f"📦 Running {len(pending_migrations)} pending migrations...")
        
        for migration in pending_migrations:
            self._run_migration(migration)
        
        logger.info("✅ All migrations completed")
    
    def _get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version FROM migrations WHERE status = 'completed'"))
                return [row[0] for row in result]
        except:
            return []
    
    def _get_available_migrations(self) -> List[Migration]:
        """Get available migrations from migration files"""
        
        migrations = []
        
        # Add default migrations
        migrations.extend(self._get_default_migrations())
        
        # Add custom migrations from files
        migration_files = sorted(self.migrations_dir.glob("*.sql"))
        for migration_file in migration_files:
            migration = self._parse_migration_file(migration_file)
            if migration:
                migrations.append(migration)
        
        return sorted(migrations, key=lambda m: m.version)
    
    def _get_default_migrations(self) -> List[Migration]:
        """Get default system migrations"""
        
        migrations = []
        
        # Initial setup migration
        initial_migration = Migration(
            version="001_initial_setup",
            name="Initial Setup",
            description="Initialize system metadata and basic configuration",
            up_sql="""
                INSERT OR IGNORE INTO system_metadata (key, value, description) VALUES
                ('database_version', '1.0.0', 'Database schema version'),
                ('created_at', datetime('now'), 'Database creation timestamp'),
                ('neurocluster_version', '1.0.0', 'NeuroCluster Elite version'),
                ('algorithm_efficiency', '99.59', 'Target algorithm efficiency'),
                ('processing_speed_target', '0.045', 'Target processing speed in ms');
            """,
            down_sql="DELETE FROM system_metadata WHERE key IN ('database_version', 'created_at', 'neurocluster_version', 'algorithm_efficiency', 'processing_speed_target');"
        )
        migrations.append(initial_migration)
        
        # Sample data migration
        sample_data_migration = Migration(
            version="002_sample_data",
            name="Sample Data",
            description="Insert sample data for testing and demonstration",
            up_sql=self._get_sample_data_sql(),
            down_sql="DELETE FROM user_settings WHERE user_id = 'demo_user';"
        )
        migrations.append(sample_data_migration)
        
        return migrations
    
    def _get_sample_data_sql(self) -> str:
        """Get SQL for inserting sample data"""
        
        return """
            -- Sample user settings
            INSERT OR IGNORE INTO user_settings (user_id, setting_key, setting_value, setting_type, description) VALUES
            ('demo_user', 'risk_level', 'moderate', 'string', 'User risk tolerance level'),
            ('demo_user', 'initial_capital', '100000', 'float', 'Starting capital amount'),
            ('demo_user', 'max_position_size', '0.10', 'float', 'Maximum position size as percentage of portfolio'),
            ('demo_user', 'paper_trading', 'true', 'boolean', 'Enable paper trading mode'),
            ('demo_user', 'auto_trading', 'false', 'boolean', 'Enable automatic trading'),
            ('demo_user', 'preferred_assets', 'AAPL,GOOGL,MSFT,TSLA', 'string', 'Preferred trading symbols');
            
            -- Sample algorithm performance record
            INSERT OR IGNORE INTO algorithm_performance (efficiency, processing_time_ms, accuracy, memory_usage_mb, clustering_quality)
            VALUES (99.59, 0.045, 94.7, 12.4, 0.918);
        """
    
    def _parse_migration_file(self, migration_file: Path) -> Optional[Migration]:
        """Parse a migration file"""
        
        try:
            content = migration_file.read_text()
            
            # Simple parser for migration files
            # Expected format:
            # -- Migration: version_name
            # -- Description: Migration description
            # -- Up:
            # SQL statements
            # -- Down:
            # SQL statements
            
            lines = content.split('\n')
            version = None
            name = None
            description = ""
            up_sql = ""
            down_sql = ""
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('-- Migration:'):
                    version = line.replace('-- Migration:', '').strip()
                    name = version
                elif line.startswith('-- Description:'):
                    description = line.replace('-- Description:', '').strip()
                elif line == '-- Up:':
                    current_section = 'up'
                elif line == '-- Down:':
                    current_section = 'down'
                elif current_section == 'up' and not line.startswith('--'):
                    up_sql += line + '\n'
                elif current_section == 'down' and not line.startswith('--'):
                    down_sql += line + '\n'
            
            if version and up_sql:
                return Migration(
                    version=version,
                    name=name,
                    description=description,
                    up_sql=up_sql.strip(),
                    down_sql=down_sql.strip()
                )
            
        except Exception as e:
            logger.warning(f"Failed to parse migration file {migration_file}: {e}")
        
        return None
    
    def _run_migration(self, migration: Migration):
        """Run a single migration"""
        
        logger.info(f"📦 Running migration: {migration.version} - {migration.name}")
        
        try:
            with self.engine.begin() as conn:
                # Record migration start
                conn.execute(text("""
                    INSERT OR REPLACE INTO migrations (version, name, description, status, checksum)
                    VALUES (:version, :name, :description, 'running', :checksum)
                """), {
                    "version": migration.version,
                    "name": migration.name,
                    "description": migration.description,
                    "checksum": hashlib.sha256(migration.up_sql.encode()).hexdigest()
                })
                
                # Execute migration SQL
                if migration.up_sql:
                    conn.execute(text(migration.up_sql))
                
                # Record migration completion
                conn.execute(text("""
                    UPDATE migrations 
                    SET status = 'completed', applied_at = datetime('now')
                    WHERE version = :version
                """), {"version": migration.version})
                
                logger.info(f"✅ Migration {migration.version} completed")
                
        except Exception as e:
            logger.error(f"❌ Migration {migration.version} failed: {e}")
            
            # Record migration failure
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        UPDATE migrations 
                        SET status = 'failed'
                        WHERE version = :version
                    """), {"version": migration.version})
            except:
                pass
            
            raise
    
    def _seed_initial_data(self):
        """Seed database with initial data"""
        
        logger.info("🌱 Seeding initial data...")
        
        try:
            with self.engine.connect() as conn:
                # Check if data already exists
                result = conn.execute(text("SELECT COUNT(*) FROM system_metadata"))
                count = result.scalar()
                
                if count > 0:
                    logger.info("✅ Initial data already exists")
                    return
                
                # Insert system metadata
                conn.execute(text("""
                    INSERT INTO system_metadata (key, value, description) VALUES
                    ('installation_id', :installation_id, 'Unique installation identifier'),
                    ('installation_date', :installation_date, 'Installation date'),
                    ('last_backup', '', 'Last backup timestamp'),
                    ('total_trades', '0', 'Total number of trades executed'),
                    ('algorithm_runs', '0', 'Total algorithm execution count')
                """), {
                    "installation_id": str(uuid.uuid4()),
                    "installation_date": datetime.now().isoformat()
                })
                
                conn.commit()
                
                logger.info("✅ Initial data seeded successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to seed initial data: {e}")
            raise
    
    def _validate_database(self) -> bool:
        """Validate database integrity and structure"""
        
        logger.info("🔍 Validating database...")
        
        try:
            with self.engine.connect() as conn:
                # Check that all required tables exist
                tables = sqlalchemy.inspect(self.engine).get_table_names()
                
                required_tables = [
                    'system_metadata', 'migrations', 'market_data',
                    'technical_indicators', 'trading_signals', 'trades',
                    'positions', 'portfolio_snapshots', 'algorithm_performance'
                ]
                
                missing_tables = [table for table in required_tables if table not in tables]
                
                if missing_tables:
                    logger.error(f"❌ Missing tables: {missing_tables}")
                    return False
                
                # Validate system metadata
                result = conn.execute(text("SELECT COUNT(*) FROM system_metadata"))
                metadata_count = result.scalar()
                
                if metadata_count == 0:
                    logger.warning("⚠️ No system metadata found")
                
                # Test basic operations
                conn.execute(text("SELECT 1"))
                
                logger.info("✅ Database validation passed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Database validation failed: {e}")
            return False
    
    def backup_database(self, backup_name: str = None) -> bool:
        """Create a database backup"""
        
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"💾 Creating database backup: {backup_name}")
        
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                # SQLite backup
                source_path = Path(self.config.sqlite_path)
                backup_path = self.backups_dir / f"{backup_name}.db"
                
                shutil.copy2(source_path, backup_path)
                
                logger.info(f"✅ SQLite backup created: {backup_path}")
                return True
            
            else:
                # For PostgreSQL/MySQL, create SQL dump
                backup_path = self.backups_dir / f"{backup_name}.sql"
                
                # This would require pg_dump or mysqldump
                logger.warning("⚠️ SQL dump backup not implemented for this database type")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database backup failed: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics"""
        
        info = {
            "database_type": self.config.db_type.value,
            "connection_string": self._get_safe_connection_string(),
            "tables": [],
            "total_records": 0,
            "database_size": 0
        }
        
        try:
            inspector = sqlalchemy.inspect(self.engine)
            tables = inspector.get_table_names()
            
            with self.engine.connect() as conn:
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        
                        info["tables"].append({
                            "name": table,
                            "record_count": count
                        })
                        
                        info["total_records"] += count
                        
                    except Exception as e:
                        logger.warning(f"Could not get count for table {table}: {e}")
                
                # Get database size for SQLite
                if self.config.db_type == DatabaseType.SQLITE:
                    db_path = Path(self.config.sqlite_path)
                    if db_path.exists():
                        info["database_size"] = db_path.stat().st_size
        
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
        
        return info
    
    def _get_safe_connection_string(self) -> str:
        """Get connection string with password masked"""
        
        if self.config.db_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.config.sqlite_path}"
        else:
            return (
                f"{self.config.db_type.value}://{self.config.username}:***@"
                f"{self.config.host}:{self.config.port}/{self.config.database}"
            )

# ==================== MAIN FUNCTION ====================

def main():
    """Main function for database setup"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroCluster Elite Database Setup")
    parser.add_argument("--db-type", choices=["sqlite", "postgresql", "mysql"], 
                       default="sqlite", help="Database type")
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--port", type=int, help="Database port")
    parser.add_argument("--database", default="neurocluster_elite", help="Database name")
    parser.add_argument("--username", default="neurocluster", help="Database username")
    parser.add_argument("--password", help="Database password")
    parser.add_argument("--sqlite-path", default="data/databases/neurocluster_elite.db", 
                       help="SQLite database file path")
    parser.add_argument("--backup", action="store_true", help="Create backup before setup")
    parser.add_argument("--info", action="store_true", help="Show database info only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create database configuration
    config = DatabaseConfig(
        db_type=DatabaseType(args.db_type),
        host=args.host,
        port=args.port or (5432 if args.db_type == "postgresql" else 3306),
        database=args.database,
        username=args.username,
        password=args.password or "",
        sqlite_path=args.sqlite_path
    )
    
    print("🗄️ NeuroCluster Elite Database Setup")
    print("=" * 50)
    print(f"📍 Database Type: {config.db_type.value}")
    
    if config.db_type == DatabaseType.SQLITE:
        print(f"📁 Database Path: {config.sqlite_path}")
    else:
        print(f"🔗 Host: {config.host}:{config.port}")
        print(f"📊 Database: {config.database}")
    
    print("=" * 50)
    
    try:
        db_manager = DatabaseManager(config)
        
        if args.info:
            # Show database info only
            if db_manager.engine is None:
                db_manager.engine = db_manager._create_engine()
            
            info = db_manager.get_database_info()
            
            print(f"\n📊 Database Information:")
            print(f"   Type: {info['database_type']}")
            print(f"   Connection: {info['connection_string']}")
            print(f"   Total Records: {info['total_records']:,}")
            print(f"   Database Size: {info['database_size']:,} bytes")
            
            print(f"\n📋 Tables:")
            for table in info['tables']:
                print(f"   • {table['name']}: {table['record_count']:,} records")
            
            return 0
        
        # Create backup if requested
        if args.backup:
            if Path(config.sqlite_path).exists():
                db_manager.backup_database()
        
        # Initialize database
        success = db_manager.initialize()
        
        if success:
            print("\n✅ Database setup completed successfully!")
            
            # Show summary
            info = db_manager.get_database_info()
            print(f"\n📊 Database Summary:")
            print(f"   • Type: {info['database_type']}")
            print(f"   • Tables: {len(info['tables'])}")
            print(f"   • Total Records: {info['total_records']:,}")
            
            print(f"\n🚀 Next Steps:")
            print(f"   1. Start the application: python startup.py")
            print(f"   2. Access dashboard: http://localhost:8501")
            print(f"   3. Check API: http://localhost:8000/docs")
            
            return 0
        else:
            print("\n❌ Database setup failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Database setup interrupted")
        return 1
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)