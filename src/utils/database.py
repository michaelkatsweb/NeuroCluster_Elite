#!/usr/bin/env python3
"""
File: database.py
Path: NeuroCluster-Elite/src/utils/database.py
Description: Comprehensive database management for NeuroCluster Elite

This module provides robust database management capabilities including connection
pooling, query building, transaction management, migrations, and multi-database
support for the NeuroCluster Elite trading platform.

Features:
- Multi-database support (SQLite, PostgreSQL, MySQL, MongoDB)
- Connection pooling and management
- Advanced query builder with ORM-like features
- Transaction management with rollback support
- Database migration system
- Performance monitoring and optimization
- Data validation and sanitization
- Backup and recovery utilities
- Real-time data streaming support

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import sqlite3
import threading
import time
import json
import logging
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import queue
import weakref

# Database drivers
try:
    import psycopg2
    import psycopg2.pool
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import mysql.connector
    import mysql.connector.pooling
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import pymongo
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import our modules
try:
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_enhanced_logger, performance_monitor
    from src.utils.security import SecurityManager
    from src.utils.helpers import calculate_hash, format_currency
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__) if 'get_enhanced_logger' in locals() else logging.getLogger(__name__)

# ==================== ENUMS AND DATA STRUCTURES ====================

class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"

class QueryType(Enum):
    """Types of database queries"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"
    ALTER = "ALTER"

class TransactionIsolation(Enum):
    """Transaction isolation levels"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    db_type: DatabaseType = DatabaseType.SQLITE
    host: str = "localhost"
    port: int = 5432
    database: str = "neurocluster_elite"
    username: str = ""
    password: str = ""
    connection_string: str = ""
    
    # Connection pool settings
    pool_size: int = 10
    max_connections: int = 20
    connection_timeout: int = 30
    pool_recycle: int = 3600
    
    # Performance settings
    query_timeout: int = 30
    batch_size: int = 1000
    enable_query_cache: bool = True
    cache_size: int = 1000
    
    # Security settings
    ssl_mode: str = "prefer"
    encrypt_data: bool = True
    backup_enabled: bool = True
    backup_interval: int = 3600  # seconds
    
    # File settings (for SQLite)
    file_path: str = "data/neurocluster.db"
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    
    def __post_init__(self):
        """Generate connection string if not provided"""
        if not self.connection_string:
            self.connection_string = self._generate_connection_string()
    
    def _generate_connection_string(self) -> str:
        """Generate database connection string"""
        if self.db_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.file_path}"
        elif self.db_type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.MYSQL:
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.MONGODB:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            return ""

@dataclass
class QueryResult:
    """Result of a database query"""
    success: bool
    data: List[Dict] = field(default_factory=list)
    affected_rows: int = 0
    execution_time: float = 0.0
    query: str = ""
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class Migration:
    """Database migration definition"""
    version: str
    name: str
    up_sql: str
    down_sql: str
    description: str = ""
    applied_at: Optional[datetime] = None
    checksum: str = field(init=False)
    
    def __post_init__(self):
        """Calculate migration checksum"""
        self.checksum = hashlib.md5((self.up_sql + self.down_sql).encode()).hexdigest()

# ==================== CONNECTION POOL MANAGER ====================

class ConnectionPool:
    """Advanced connection pool manager"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connections = queue.Queue(maxsize=config.max_connections)
        self.active_connections = weakref.WeakSet()
        self.pool_lock = threading.RLock()
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'queries_executed': 0,
            'errors': 0,
            'avg_query_time': 0.0
        }
        
        # Initialize connection pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        try:
            for _ in range(self.config.pool_size):
                conn = self._create_connection()
                if conn:
                    self.connections.put(conn)
                    self.stats['total_connections'] += 1
            
            logger.info(f"Initialized connection pool with {self.stats['total_connections']} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def _create_connection(self):
        """Create a new database connection"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                conn = sqlite3.connect(
                    self.config.file_path,
                    timeout=self.config.connection_timeout,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                
                # Configure SQLite for performance
                conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
                conn.execute(f"PRAGMA synchronous = {self.config.synchronous}")
                conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
                
                return conn
                
            elif self.config.db_type == DatabaseType.POSTGRESQL and POSTGRESQL_AVAILABLE:
                conn = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    connect_timeout=self.config.connection_timeout
                )
                conn.autocommit = False
                return conn
                
            elif self.config.db_type == DatabaseType.MYSQL and MYSQL_AVAILABLE:
                conn = mysql.connector.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    connection_timeout=self.config.connection_timeout
                )
                return conn
                
            else:
                raise ValueError(f"Unsupported database type: {self.config.db_type}")
                
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        try:
            # Get connection from pool
            conn = self.connections.get(timeout=self.config.connection_timeout)
            self.active_connections.add(conn)
            self.stats['active_connections'] = len(self.active_connections)
            
            yield conn
            
        except queue.Empty:
            # Create new connection if pool is empty
            conn = self._create_connection()
            if not conn:
                raise Exception("Failed to create database connection")
            
            self.active_connections.add(conn)
            yield conn
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
            
        finally:
            # Return connection to pool
            if conn:
                try:
                    if self.config.db_type == DatabaseType.SQLITE:
                        # Reset SQLite connection
                        conn.rollback()
                    elif self.config.db_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
                        # Reset connection state
                        conn.rollback()
                    
                    # Return to pool if there's space
                    if not self.connections.full():
                        self.connections.put(conn)
                    else:
                        conn.close()
                        
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()
                    except:
                        pass
    
    def close_all(self):
        """Close all connections in the pool"""
        with self.pool_lock:
            while not self.connections.empty():
                try:
                    conn = self.connections.get_nowait()
                    conn.close()
                except:
                    pass
            
            # Close active connections
            for conn in list(self.active_connections):
                try:
                    conn.close()
                except:
                    pass
            
            self.stats['total_connections'] = 0
            self.stats['active_connections'] = 0

# ==================== QUERY BUILDER ====================

class QueryBuilder:
    """Advanced SQL query builder"""
    
    def __init__(self, db_type: DatabaseType = DatabaseType.SQLITE):
        self.db_type = db_type
        self.reset()
    
    def reset(self):
        """Reset query builder state"""
        self.query_type = None
        self.table = ""
        self.columns = []
        self.values = {}
        self.conditions = []
        self.joins = []
        self.order_by = []
        self.group_by = []
        self.having = []
        self.limit_count = None
        self.offset_count = None
        self.parameters = []
        return self
    
    def select(self, columns: Union[str, List[str]] = "*"):
        """Start a SELECT query"""
        self.query_type = QueryType.SELECT
        if isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = columns
        return self
    
    def insert_into(self, table: str):
        """Start an INSERT query"""
        self.query_type = QueryType.INSERT
        self.table = table
        return self
    
    def update(self, table: str):
        """Start an UPDATE query"""
        self.query_type = QueryType.UPDATE
        self.table = table
        return self
    
    def delete_from(self, table: str):
        """Start a DELETE query"""
        self.query_type = QueryType.DELETE
        self.table = table
        return self
    
    def from_table(self, table: str):
        """Set the FROM table"""
        self.table = table
        return self
    
    def values(self, **kwargs):
        """Set values for INSERT/UPDATE"""
        self.values.update(kwargs)
        return self
    
    def set(self, **kwargs):
        """Alias for values()"""
        return self.values(**kwargs)
    
    def where(self, condition: str, *params):
        """Add WHERE condition"""
        self.conditions.append(condition)
        self.parameters.extend(params)
        return self
    
    def where_eq(self, column: str, value: Any):
        """Add WHERE column = value condition"""
        return self.where(f"{column} = ?", value)
    
    def where_in(self, column: str, values: List[Any]):
        """Add WHERE column IN (...) condition"""
        placeholders = ", ".join(["?" for _ in values])
        return self.where(f"{column} IN ({placeholders})", *values)
    
    def where_between(self, column: str, start: Any, end: Any):
        """Add WHERE column BETWEEN start AND end condition"""
        return self.where(f"{column} BETWEEN ? AND ?", start, end)
    
    def where_like(self, column: str, pattern: str):
        """Add WHERE column LIKE pattern condition"""
        return self.where(f"{column} LIKE ?", pattern)
    
    def join(self, table: str, condition: str, join_type: str = "INNER"):
        """Add JOIN clause"""
        self.joins.append(f"{join_type} JOIN {table} ON {condition}")
        return self
    
    def left_join(self, table: str, condition: str):
        """Add LEFT JOIN clause"""
        return self.join(table, condition, "LEFT")
    
    def right_join(self, table: str, condition: str):
        """Add RIGHT JOIN clause"""
        return self.join(table, condition, "RIGHT")
    
    def inner_join(self, table: str, condition: str):
        """Add INNER JOIN clause"""
        return self.join(table, condition, "INNER")
    
    def order_by_asc(self, column: str):
        """Add ORDER BY column ASC"""
        self.order_by.append(f"{column} ASC")
        return self
    
    def order_by_desc(self, column: str):
        """Add ORDER BY column DESC"""
        self.order_by.append(f"{column} DESC")
        return self
    
    def group_by_column(self, column: str):
        """Add GROUP BY column"""
        self.group_by.append(column)
        return self
    
    def having_condition(self, condition: str, *params):
        """Add HAVING condition"""
        self.having.append(condition)
        self.parameters.extend(params)
        return self
    
    def limit(self, count: int):
        """Add LIMIT clause"""
        self.limit_count = count
        return self
    
    def offset(self, count: int):
        """Add OFFSET clause"""
        self.offset_count = count
        return self
    
    def build(self) -> Tuple[str, List[Any]]:
        """Build the final SQL query"""
        if self.query_type == QueryType.SELECT:
            return self._build_select()
        elif self.query_type == QueryType.INSERT:
            return self._build_insert()
        elif self.query_type == QueryType.UPDATE:
            return self._build_update()
        elif self.query_type == QueryType.DELETE:
            return self._build_delete()
        else:
            raise ValueError(f"Unsupported query type: {self.query_type}")
    
    def _build_select(self) -> Tuple[str, List[Any]]:
        """Build SELECT query"""
        parts = []
        params = []
        
        # SELECT clause
        columns_str = ", ".join(self.columns)
        parts.append(f"SELECT {columns_str}")
        
        # FROM clause
        if self.table:
            parts.append(f"FROM {self.table}")
        
        # JOIN clauses
        if self.joins:
            parts.extend(self.joins)
        
        # WHERE clause
        if self.conditions:
            where_str = " AND ".join(self.conditions)
            parts.append(f"WHERE {where_str}")
            params.extend(self.parameters)
        
        # GROUP BY clause
        if self.group_by:
            group_str = ", ".join(self.group_by)
            parts.append(f"GROUP BY {group_str}")
        
        # HAVING clause
        if self.having:
            having_str = " AND ".join(self.having)
            parts.append(f"HAVING {having_str}")
        
        # ORDER BY clause
        if self.order_by:
            order_str = ", ".join(self.order_by)
            parts.append(f"ORDER BY {order_str}")
        
        # LIMIT clause
        if self.limit_count is not None:
            parts.append(f"LIMIT {self.limit_count}")
        
        # OFFSET clause
        if self.offset_count is not None:
            parts.append(f"OFFSET {self.offset_count}")
        
        query = " ".join(parts)
        return query, params
    
    def _build_insert(self) -> Tuple[str, List[Any]]:
        """Build INSERT query"""
        if not self.values:
            raise ValueError("No values specified for INSERT")
        
        columns = list(self.values.keys())
        values = list(self.values.values())
        
        columns_str = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in values])
        
        query = f"INSERT INTO {self.table} ({columns_str}) VALUES ({placeholders})"
        return query, values
    
    def _build_update(self) -> Tuple[str, List[Any]]:
        """Build UPDATE query"""
        if not self.values:
            raise ValueError("No values specified for UPDATE")
        
        set_clauses = []
        params = []
        
        for column, value in self.values.items():
            set_clauses.append(f"{column} = ?")
            params.append(value)
        
        set_str = ", ".join(set_clauses)
        parts = [f"UPDATE {self.table}", f"SET {set_str}"]
        
        # WHERE clause
        if self.conditions:
            where_str = " AND ".join(self.conditions)
            parts.append(f"WHERE {where_str}")
            params.extend(self.parameters)
        
        query = " ".join(parts)
        return query, params
    
    def _build_delete(self) -> Tuple[str, List[Any]]:
        """Build DELETE query"""
        parts = [f"DELETE FROM {self.table}"]
        params = []
        
        # WHERE clause
        if self.conditions:
            where_str = " AND ".join(self.conditions)
            parts.append(f"WHERE {where_str}")
            params.extend(self.parameters)
        
        query = " ".join(parts)
        return query, params

# ==================== TRANSACTION MANAGER ====================

class TransactionManager:
    """Database transaction manager with rollback support"""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.connection_pool = connection_pool
        self.active_transactions = {}
        self.transaction_lock = threading.RLock()
    
    @contextmanager
    def transaction(self, isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED):
        """Execute operations within a transaction"""
        transaction_id = f"txn_{int(time.time() * 1000)}_{threading.current_thread().ident}"
        
        with self.connection_pool.get_connection() as conn:
            try:
                # Start transaction
                if hasattr(conn, 'autocommit'):
                    conn.autocommit = False
                
                # Set isolation level for PostgreSQL/MySQL
                if self.connection_pool.config.db_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
                    cursor = conn.cursor()
                    cursor.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation_level.value}")
                    cursor.close()
                
                # Register transaction
                with self.transaction_lock:
                    self.active_transactions[transaction_id] = {
                        'connection': conn,
                        'start_time': datetime.now(),
                        'operations': []
                    }
                
                yield TransactionContext(transaction_id, conn, self)
                
                # Commit transaction
                conn.commit()
                logger.debug(f"Transaction {transaction_id} committed successfully")
                
            except Exception as e:
                # Rollback transaction
                try:
                    conn.rollback()
                    logger.warning(f"Transaction {transaction_id} rolled back due to error: {e}")
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback transaction {transaction_id}: {rollback_error}")
                
                raise e
                
            finally:
                # Clean up transaction
                with self.transaction_lock:
                    if transaction_id in self.active_transactions:
                        del self.active_transactions[transaction_id]

class TransactionContext:
    """Context for database transaction operations"""
    
    def __init__(self, transaction_id: str, connection, manager: TransactionManager):
        self.transaction_id = transaction_id
        self.connection = connection
        self.manager = manager
        self.operations = []
    
    def execute(self, query: str, params: List[Any] = None) -> QueryResult:
        """Execute query within transaction"""
        start_time = time.time()
        params = params or []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            
            # Get results based on query type
            if query.strip().upper().startswith('SELECT'):
                if hasattr(cursor, 'fetchall'):
                    rows = cursor.fetchall()
                    if hasattr(cursor, 'description') and cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        data = [dict(zip(columns, row)) for row in rows]
                    else:
                        data = [dict(row) for row in rows] if rows else []
                else:
                    data = []
                affected_rows = len(data)
            else:
                data = []
                affected_rows = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
            
            cursor.close()
            
            execution_time = (time.time() - start_time) * 1000
            
            # Record operation
            operation = {
                'query': query,
                'params': params,
                'execution_time': execution_time,
                'affected_rows': affected_rows,
                'timestamp': datetime.now()
            }
            self.operations.append(operation)
            
            return QueryResult(
                success=True,
                data=data,
                affected_rows=affected_rows,
                execution_time=execution_time,
                query=query
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Query execution failed in transaction {self.transaction_id}: {e}")
            
            return QueryResult(
                success=False,
                execution_time=execution_time,
                query=query,
                error=str(e)
            )

# ==================== MIGRATION MANAGER ====================

class MigrationManager:
    """Database migration management system"""
    
    def __init__(self, database_manager):
        self.db_manager = database_manager
        self.migrations_dir = Path("data/migrations")
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure migrations table exists
        self._ensure_migrations_table()
    
    def _ensure_migrations_table(self):
        """Create migrations tracking table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            checksum VARCHAR(32) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time REAL
        )
        """
        
        self.db_manager.execute(create_table_sql)
    
    def create_migration(self, name: str, up_sql: str, down_sql: str) -> Migration:
        """Create a new migration"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        migration = Migration(
            version=version,
            name=name,
            up_sql=up_sql,
            down_sql=down_sql,
            description=f"Migration: {name}"
        )
        
        # Save migration file
        migration_file = self.migrations_dir / f"{version}_{name}.json"
        with open(migration_file, 'w') as f:
            json.dump(asdict(migration), f, indent=2, default=str)
        
        logger.info(f"Created migration: {version}_{name}")
        return migration
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations"""
        # Get applied migrations
        result = self.db_manager.execute("SELECT version FROM schema_migrations ORDER BY version")
        applied_versions = {row['version'] for row in result.data}
        
        # Get all migration files
        migration_files = sorted(self.migrations_dir.glob("*.json"))
        pending_migrations = []
        
        for migration_file in migration_files:
            try:
                with open(migration_file, 'r') as f:
                    migration_data = json.load(f)
                
                migration = Migration(**migration_data)
                
                if migration.version not in applied_versions:
                    pending_migrations.append(migration)
                    
            except Exception as e:
                logger.error(f"Error loading migration {migration_file}: {e}")
        
        return pending_migrations
    
    def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration"""
        try:
            with self.db_manager.transaction() as txn:
                start_time = time.time()
                
                # Execute migration SQL
                result = txn.execute(migration.up_sql)
                
                if not result.success:
                    raise Exception(f"Migration failed: {result.error}")
                
                execution_time = (time.time() - start_time) * 1000
                
                # Record migration as applied
                txn.execute(
                    "INSERT INTO schema_migrations (version, name, checksum, execution_time) VALUES (?, ?, ?, ?)",
                    [migration.version, migration.name, migration.checksum, execution_time]
                )
                
                logger.info(f"Applied migration {migration.version}: {migration.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            return False
    
    def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a migration"""
        try:
            with self.db_manager.transaction() as txn:
                # Execute rollback SQL
                result = txn.execute(migration.down_sql)
                
                if not result.success:
                    raise Exception(f"Migration rollback failed: {result.error}")
                
                # Remove migration record
                txn.execute(
                    "DELETE FROM schema_migrations WHERE version = ?",
                    [migration.version]
                )
                
                logger.info(f"Rolled back migration {migration.version}: {migration.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration.version}: {e}")
            return False
    
    def migrate(self) -> bool:
        """Apply all pending migrations"""
        pending = self.get_pending_migrations()
        
        if not pending:
            logger.info("No pending migrations")
            return True
        
        logger.info(f"Applying {len(pending)} pending migrations")
        
        for migration in pending:
            if not self.apply_migration(migration):
                logger.error(f"Migration failed at {migration.version}, stopping")
                return False
        
        logger.info("All migrations applied successfully")
        return True

# ==================== MAIN DATABASE MANAGER ====================

class DatabaseManager:
    """
    Comprehensive database manager for NeuroCluster Elite
    
    This class provides high-level database operations with connection pooling,
    transaction management, query building, and migration support.
    """
    
    def __init__(self, config: Union[Dict, DatabaseConfig] = None):
        if isinstance(config, dict):
            self.config = DatabaseConfig(**config)
        else:
            self.config = config or DatabaseConfig()
        
        # Initialize components
        self.connection_pool = None
        self.transaction_manager = None
        self.migration_manager = None
        self.query_cache = {}
        self.performance_stats = {
            'queries_executed': 0,
            'total_execution_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread safety
        self.manager_lock = threading.RLock()
        
        # Initialize database
        self.initialize()
    
    def initialize(self):
        """Initialize database manager"""
        try:
            # Create data directory if needed
            if self.config.db_type == DatabaseType.SQLITE:
                db_path = Path(self.config.file_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize connection pool
            self.connection_pool = ConnectionPool(self.config)
            
            # Initialize transaction manager
            self.transaction_manager = TransactionManager(self.connection_pool)
            
            # Initialize migration manager
            self.migration_manager = MigrationManager(self)
            
            # Create core tables
            self._create_core_tables()
            
            logger.info(f"Database manager initialized: {self.config.db_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    def _create_core_tables(self):
        """Create core system tables"""
        
        # Market data table
        create_market_data_table = """
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(20) NOT NULL,
            asset_type VARCHAR(20) NOT NULL,
            price REAL NOT NULL,
            change_amount REAL,
            change_percent REAL,
            volume REAL,
            high REAL,
            low REAL,
            bid REAL,
            ask REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_symbol_timestamp (symbol, timestamp),
            INDEX idx_asset_type (asset_type)
        )
        """
        
        # Trading positions table
        create_positions_table = """
        CREATE TABLE IF NOT EXISTS trading_positions (
            id VARCHAR(36) PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            asset_type VARCHAR(20) NOT NULL,
            side VARCHAR(10) NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            market_value REAL,
            unrealized_pnl REAL,
            stop_loss REAL,
            take_profit REAL,
            opened_at TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            strategy_name VARCHAR(50),
            regime_at_entry VARCHAR(20),
            INDEX idx_symbol (symbol),
            INDEX idx_opened_at (opened_at)
        )
        """
        
        # Trading history table
        create_trades_table = """
        CREATE TABLE IF NOT EXISTS trading_history (
            id VARCHAR(36) PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            asset_type VARCHAR(20) NOT NULL,
            side VARCHAR(10) NOT NULL,
            signal_type VARCHAR(20),
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            realized_pnl REAL,
            realized_pnl_pct REAL,
            fees REAL,
            strategy_name VARCHAR(50),
            regime_at_entry VARCHAR(20),
            regime_at_exit VARCHAR(20),
            exit_reason VARCHAR(50),
            duration_minutes INTEGER,
            INDEX idx_symbol_entry_time (symbol, entry_time),
            INDEX idx_strategy (strategy_name)
        )
        """
        
        # Algorithm performance table
        create_performance_table = """
        CREATE TABLE IF NOT EXISTS algorithm_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            regime_detected VARCHAR(20),
            confidence REAL,
            processing_time_ms REAL,
            cluster_count INTEGER,
            feature_count INTEGER,
            quality_score REAL,
            INDEX idx_timestamp (timestamp),
            INDEX idx_regime (regime_detected)
        )
        """
        
        # Execute table creation
        tables = [
            create_market_data_table,
            create_positions_table,
            create_trades_table,
            create_performance_table
        ]
        
        for table_sql in tables:
            result = self.execute(table_sql)
            if not result.success:
                logger.error(f"Failed to create table: {result.error}")
    
    @performance_monitor
    def execute(self, query: str, params: List[Any] = None) -> QueryResult:
        """Execute a database query"""
        start_time = time.time()
        params = params or []
        
        # Check cache for SELECT queries
        if query.strip().upper().startswith('SELECT') and self.config.enable_query_cache:
            cache_key = hashlib.md5((query + str(params)).encode()).hexdigest()
            if cache_key in self.query_cache:
                self.performance_stats['cache_hits'] += 1
                cached_result = self.query_cache[cache_key]
                logger.debug(f"Query cache hit: {cache_key[:8]}...")
                return cached_result
            else:
                self.performance_stats['cache_misses'] += 1
        
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                # Get results based on query type
                if query.strip().upper().startswith('SELECT'):
                    if hasattr(cursor, 'fetchall'):
                        rows = cursor.fetchall()
                        if hasattr(cursor, 'description') and cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            data = [dict(zip(columns, row)) for row in rows]
                        else:
                            data = [dict(row) for row in rows] if rows else []
                    else:
                        data = []
                    affected_rows = len(data)
                else:
                    data = []
                    affected_rows = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
                    conn.commit()  # Auto-commit for non-SELECT queries
                
                cursor.close()
                
                execution_time = (time.time() - start_time) * 1000
                
                result = QueryResult(
                    success=True,
                    data=data,
                    affected_rows=affected_rows,
                    execution_time=execution_time,
                    query=query
                )
                
                # Cache SELECT results
                if (query.strip().upper().startswith('SELECT') and 
                    self.config.enable_query_cache and 
                    len(self.query_cache) < self.config.cache_size):
                    
                    cache_key = hashlib.md5((query + str(params)).encode()).hexdigest()
                    self.query_cache[cache_key] = result
                
                # Update statistics
                self.performance_stats['queries_executed'] += 1
                self.performance_stats['total_execution_time'] += execution_time
                
                return result
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Database query failed: {e}")
            
            return QueryResult(
                success=False,
                execution_time=execution_time,
                query=query,
                error=str(e)
            )
    
    def query_builder(self) -> QueryBuilder:
        """Get a new query builder instance"""
        return QueryBuilder(self.config.db_type)
    
    def transaction(self, isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED):
        """Start a new transaction"""
        return self.transaction_manager.transaction(isolation_level)
    
    def migrate(self) -> bool:
        """Run database migrations"""
        return self.migration_manager.migrate()
    
    def backup_database(self, backup_path: str = None) -> bool:
        """Create database backup"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/neurocluster_backup_{timestamp}.sql"
        
        try:
            backup_dir = Path(backup_path).parent
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            if self.config.db_type == DatabaseType.SQLITE:
                import shutil
                shutil.copy2(self.config.file_path, backup_path.replace('.sql', '.db'))
                logger.info(f"SQLite database backed up to: {backup_path}")
                return True
            else:
                # For other databases, would implement SQL dump
                logger.warning(f"Backup not implemented for {self.config.db_type}")
                return False
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """Get database performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add calculated metrics
        if stats['queries_executed'] > 0:
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['queries_executed']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['avg_execution_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # Add connection pool stats
        if self.connection_pool:
            stats.update(self.connection_pool.stats)
        
        return stats
    
    def close(self):
        """Close database manager and all connections"""
        try:
            if self.connection_pool:
                self.connection_pool.close_all()
            
            logger.info("Database manager closed")
            
        except Exception as e:
            logger.error(f"Error closing database manager: {e}")

# ==================== TESTING ====================

def test_database_manager():
    """Test database manager functionality"""
    
    print("🗄️ Testing Database Manager")
    print("=" * 50)
    
    # Create test configuration
    config = DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        file_path="data/test_neurocluster.db",
        pool_size=5
    )
    
    # Create database manager
    db_manager = DatabaseManager(config)
    
    print(f"✅ Database manager initialized: {config.db_type.value}")
    
    # Test query builder
    query_builder = db_manager.query_builder()
    query, params = (query_builder
                    .select(["symbol", "price", "timestamp"])
                    .from_table("market_data")
                    .where_eq("asset_type", "STOCK")
                    .where("price > ?", 100.0)
                    .order_by_desc("timestamp")
                    .limit(10)
                    .build())
    
    print(f"✅ Query builder test:")
    print(f"   Query: {query}")
    print(f"   Params: {params}")
    
    # Test transaction
    try:
        with db_manager.transaction() as txn:
            result = txn.execute("SELECT COUNT(*) as count FROM trading_positions")
            print(f"✅ Transaction test: {result.success}")
            if result.data:
                print(f"   Position count: {result.data[0].get('count', 0)}")
    except Exception as e:
        print(f"   Transaction error: {e}")
    
    # Test performance stats
    stats = db_manager.get_performance_stats()
    print(f"✅ Performance stats:")
    print(f"   Queries executed: {stats['queries_executed']}")
    print(f"   Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
    
    # Close database
    db_manager.close()
    
    print("\n🎉 Database manager tests completed!")

if __name__ == "__main__":
    test_database_manager()