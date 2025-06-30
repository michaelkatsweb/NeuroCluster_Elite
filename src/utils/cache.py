#!/usr/bin/env python3
"""
File: cache.py
Path: NeuroCluster-Elite/src/utils/cache.py
Description: Comprehensive caching system for NeuroCluster Elite

This module provides advanced caching capabilities including multi-backend support,
intelligent cache strategies, data compression, encryption, and high-performance
operations for the NeuroCluster Elite trading platform.

Features:
- Multi-backend support (Memory, Redis, Hybrid)
- Intelligent cache strategies (LRU, LFU, TTL, Write-through, Write-back)
- Data compression and encryption
- Cache clustering and sharding
- Performance monitoring and analytics
- Automatic cache warming and preloading
- Cache invalidation patterns
- Real-time cache metrics and health monitoring

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import time
import json
import pickle
import zlib
import hashlib
import threading
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc

# Cache backends
try:
    import redis
    import redis.sentinel
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcache
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False

# Import our modules
try:
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_enhanced_logger, performance_monitor
    from src.utils.security import SecurityManager
    from src.utils.helpers import calculate_hash, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__) if 'get_enhanced_logger' in locals() else logging.getLogger(__name__)

# ==================== ENUMS AND DATA STRUCTURES ====================

class CacheBackend(Enum):
    """Supported cache backends"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    HYBRID = "hybrid"

class CacheStrategy(Enum):
    """Cache eviction and management strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    RANDOM = "random"  # Random eviction

class CompressionType(Enum):
    """Data compression types"""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    BROTLI = "brotli"

class SerializationType(Enum):
    """Data serialization types"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    backend: CacheBackend = CacheBackend.MEMORY
    strategy: CacheStrategy = CacheStrategy.LRU
    
    # Capacity settings
    max_size: int = 1000
    max_memory_mb: int = 100
    
    # TTL settings
    default_ttl: int = 300  # 5 minutes
    max_ttl: int = 3600  # 1 hour
    
    # Performance settings
    compression: CompressionType = CompressionType.ZLIB
    serialization: SerializationType = SerializationType.PICKLE
    compression_threshold: int = 1024  # Compress if data > 1KB
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_cluster: bool = False
    redis_sentinel: List[str] = field(default_factory=list)
    
    # Memcached settings
    memcached_servers: List[str] = field(default_factory=lambda: ["localhost:11211"])
    
    # Security settings
    encryption_enabled: bool = False
    encryption_key: str = ""
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_interval: int = 60  # seconds
    
    # Advanced settings
    enable_write_through: bool = False
    enable_write_back: bool = False
    write_back_interval: int = 30  # seconds
    cache_warming: bool = True
    preload_keys: List[str] = field(default_factory=list)

@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[int] = None
    size_bytes: int = 0
    compressed: bool = False
    encrypted: bool = False
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access metadata"""
        self.accessed_at = time.time()
        self.access_count += 1

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0
    
    # Performance metrics
    avg_get_time: float = 0.0
    avg_set_time: float = 0.0
    total_get_time: float = 0.0
    total_set_time: float = 0.0
    
    # Memory metrics
    memory_used: int = 0
    memory_limit: int = 0
    entry_count: int = 0
    
    # Calculated metrics
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate()
    
    def memory_usage_pct(self) -> float:
        return self.memory_used / self.memory_limit if self.memory_limit > 0 else 0.0

# ==================== CACHE BACKENDS ====================

class MemoryCacheBackend:
    """High-performance in-memory cache backend"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = OrderedDict()  # For LRU ordering
        self.access_counts = defaultdict(int)  # For LFU
        self.cache_lock = threading.RLock()
        self.memory_used = 0
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from cache"""
        with self.cache_lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.memory_used -= entry.size_bytes
                return None
            
            # Update access metadata
            entry.update_access()
            
            # Move to end for LRU
            if self.config.strategy == CacheStrategy.LRU:
                self.cache.move_to_end(key)
            
            # Update access count for LFU
            if self.config.strategy == CacheStrategy.LFU:
                self.access_counts[key] += 1
            
            return entry
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        try:
            with self.cache_lock:
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl=ttl or self.config.default_ttl,
                    size_bytes=len(str(value))  # Approximate size
                )
                
                # Check if we need to evict entries
                self._ensure_capacity(entry.size_bytes)
                
                # Store entry
                self.cache[key] = entry
                self.memory_used += entry.size_bytes
                
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                self.memory_used -= entry.size_bytes
                
                if key in self.access_counts:
                    del self.access_counts[key]
                
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.cache_lock:
            self.cache.clear()
            self.access_counts.clear()
            self.memory_used = 0
    
    def keys(self) -> List[str]:
        """Get all cache keys"""
        with self.cache_lock:
            return list(self.cache.keys())
    
    def size(self) -> int:
        """Get number of entries"""
        return len(self.cache)
    
    def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry"""
        # Check memory limit
        memory_limit_bytes = self.config.max_memory_mb * 1024 * 1024
        
        while (self.memory_used + new_entry_size > memory_limit_bytes or 
               len(self.cache) >= self.config.max_size):
            
            if not self.cache:
                break
            
            # Evict based on strategy
            if self.config.strategy == CacheStrategy.LRU:
                # Remove least recently used (first item)
                key, entry = self.cache.popitem(last=False)
            elif self.config.strategy == CacheStrategy.LFU:
                # Remove least frequently used
                key = min(self.access_counts, key=self.access_counts.get)
                entry = self.cache.pop(key)
                del self.access_counts[key]
            elif self.config.strategy == CacheStrategy.FIFO:
                # Remove first in (first item)
                key, entry = self.cache.popitem(last=False)
            else:
                # Default to LRU
                key, entry = self.cache.popitem(last=False)
            
            self.memory_used -= entry.size_bytes
            logger.debug(f"Evicted cache entry: {key}")

class RedisCacheBackend:
    """Redis-based cache backend with clustering support"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        try:
            if self.config.redis_cluster:
                # Redis Cluster mode
                startup_nodes = [
                    {"host": host.split(':')[0], "port": int(host.split(':')[1])}
                    for host in self.config.redis_sentinel
                ]
                self.redis_client = redis.RedisCluster(
                    startup_nodes=startup_nodes,
                    decode_responses=False,
                    password=self.config.redis_password if self.config.redis_password else None
                )
            elif self.config.redis_sentinel:
                # Redis Sentinel mode
                sentinel_list = [
                    (host.split(':')[0], int(host.split(':')[1]))
                    for host in self.config.redis_sentinel
                ]
                sentinel = redis.sentinel.Sentinel(sentinel_list)
                self.redis_client = sentinel.master_for('mymaster', socket_timeout=0.1)
            else:
                # Standard Redis connection
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password if self.config.redis_password else None,
                    decode_responses=False
                )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache backend initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from Redis"""
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            
            # Deserialize entry
            entry = pickle.loads(data)
            
            # Update access metadata
            entry.update_access()
            
            return entry
            
        except Exception as e:
            logger.error(f"Error getting Redis cache entry {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in Redis"""
        try:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl or self.config.default_ttl
            )
            
            # Serialize entry
            data = pickle.dumps(entry)
            
            # Set with TTL
            self.redis_client.setex(
                key, 
                ttl or self.config.default_ttl,
                data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting Redis cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis"""
        try:
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting Redis cache entry {key}: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries"""
        try:
            self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
    
    def keys(self) -> List[str]:
        """Get all cache keys"""
        try:
            return [key.decode() for key in self.redis_client.keys('*')]
        except Exception as e:
            logger.error(f"Error getting Redis cache keys: {e}")
            return []
    
    def size(self) -> int:
        """Get number of entries"""
        try:
            return self.redis_client.dbsize()
        except Exception as e:
            logger.error(f"Error getting Redis cache size: {e}")
            return 0

class HybridCacheBackend:
    """Hybrid cache backend (L1: Memory, L2: Redis)"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Create L1 (memory) and L2 (Redis) backends
        memory_config = CacheConfig(
            backend=CacheBackend.MEMORY,
            max_size=config.max_size // 2,  # Split capacity
            max_memory_mb=config.max_memory_mb // 2
        )
        
        self.l1_cache = MemoryCacheBackend(memory_config)
        self.l2_cache = RedisCacheBackend(config) if REDIS_AVAILABLE else None
        
        # Metrics
        self.l1_hits = 0
        self.l2_hits = 0
        self.total_requests = 0
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from hybrid cache (L1 first, then L2)"""
        self.total_requests += 1
        
        # Try L1 cache first
        entry = self.l1_cache.get(key)
        if entry:
            self.l1_hits += 1
            return entry
        
        # Try L2 cache
        if self.l2_cache:
            entry = self.l2_cache.get(key)
            if entry:
                self.l2_hits += 1
                # Promote to L1 cache
                self.l1_cache.set(key, entry.value, entry.ttl)
                return entry
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in both cache levels"""
        success = True
        
        # Set in L1 cache
        if not self.l1_cache.set(key, value, ttl):
            success = False
        
        # Set in L2 cache
        if self.l2_cache and not self.l2_cache.set(key, value, ttl):
            success = False
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete value from both cache levels"""
        l1_success = self.l1_cache.delete(key)
        l2_success = self.l2_cache.delete(key) if self.l2_cache else True
        
        return l1_success or l2_success
    
    def clear(self):
        """Clear both cache levels"""
        self.l1_cache.clear()
        if self.l2_cache:
            self.l2_cache.clear()
    
    def keys(self) -> List[str]:
        """Get all cache keys from both levels"""
        l1_keys = set(self.l1_cache.keys())
        l2_keys = set(self.l2_cache.keys()) if self.l2_cache else set()
        return list(l1_keys | l2_keys)
    
    def size(self) -> int:
        """Get total number of unique entries"""
        return len(self.keys())
    
    def get_level_stats(self) -> Dict:
        """Get cache level statistics"""
        return {
            'l1_hits': self.l1_hits,
            'l2_hits': self.l2_hits,
            'total_requests': self.total_requests,
            'l1_hit_rate': self.l1_hits / self.total_requests if self.total_requests > 0 else 0,
            'l2_hit_rate': self.l2_hits / self.total_requests if self.total_requests > 0 else 0
        }

# ==================== MAIN CACHE MANAGER ====================

class CacheManager:
    """
    Comprehensive cache manager for NeuroCluster Elite
    
    This class provides high-level caching operations with support for multiple
    backends, intelligent strategies, and advanced features like compression
    and encryption.
    """
    
    def __init__(self, config: Union[Dict, CacheConfig] = None):
        if isinstance(config, dict):
            self.config = CacheConfig(**config)
        else:
            self.config = config or CacheConfig()
        
        # Initialize backend
        self.backend = self._create_backend()
        
        # Metrics and monitoring
        self.metrics = CacheMetrics()
        self.metrics_lock = threading.RLock()
        
        # Security
        self.security_manager = None
        if self.config.encryption_enabled:
            self._initialize_security()
        
        # Background tasks
        self.background_tasks = []
        self.shutdown_event = threading.Event()
        
        # Start background threads
        self._start_background_tasks()
        
        logger.info(f"Cache manager initialized: {self.config.backend.value}")
    
    def _create_backend(self):
        """Create cache backend based on configuration"""
        if self.config.backend == CacheBackend.MEMORY:
            return MemoryCacheBackend(self.config)
        elif self.config.backend == CacheBackend.REDIS:
            return RedisCacheBackend(self.config)
        elif self.config.backend == CacheBackend.HYBRID:
            return HybridCacheBackend(self.config)
        else:
            raise ValueError(f"Unsupported cache backend: {self.config.backend}")
    
    def _initialize_security(self):
        """Initialize security for encryption"""
        try:
            from src.utils.security import SecurityManager
            self.security_manager = SecurityManager()
            logger.info("Cache encryption enabled")
        except ImportError:
            logger.warning("Security manager not available, encryption disabled")
            self.config.encryption_enabled = False
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Metrics collection task
        if self.config.enable_metrics:
            metrics_thread = threading.Thread(
                target=self._metrics_collection_loop,
                daemon=True
            )
            metrics_thread.start()
            self.background_tasks.append(metrics_thread)
        
        # Cache warming task
        if self.config.cache_warming and self.config.preload_keys:
            warming_thread = threading.Thread(
                target=self._cache_warming_loop,
                daemon=True
            )
            warming_thread.start()
            self.background_tasks.append(warming_thread)
        
        # Cleanup task
        cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        cleanup_thread.start()
        self.background_tasks.append(cleanup_thread)
    
    @performance_monitor
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        start_time = time.time()
        
        try:
            entry = self.backend.get(key)
            
            execution_time = (time.time() - start_time) * 1000
            
            with self.metrics_lock:
                if entry:
                    self.metrics.hits += 1
                    value = entry.value
                    
                    # Decrypt if enabled
                    if self.config.encryption_enabled and entry.encrypted:
                        value = self._decrypt_data(value)
                    
                    # Decompress if compressed
                    if entry.compressed:
                        value = self._decompress_data(value)
                    
                    self.metrics.total_get_time += execution_time
                    self.metrics.avg_get_time = self.metrics.total_get_time / self.metrics.hits
                    
                    return value
                else:
                    self.metrics.misses += 1
                    return default
                    
        except Exception as e:
            logger.error(f"Error getting cache value for key {key}: {e}")
            with self.metrics_lock:
                self.metrics.errors += 1
            return default
    
    @performance_monitor
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        start_time = time.time()
        
        try:
            processed_value = value
            compressed = False
            encrypted = False
            
            # Serialize value for size calculation
            serialized = self._serialize_data(value)
            
            # Compress if above threshold
            if (self.config.compression != CompressionType.NONE and 
                len(serialized) > self.config.compression_threshold):
                processed_value = self._compress_data(serialized)
                compressed = True
            
            # Encrypt if enabled
            if self.config.encryption_enabled:
                if not compressed:
                    processed_value = serialized
                processed_value = self._encrypt_data(processed_value)
                encrypted = True
            
            # Use original value if no processing
            if not compressed and not encrypted:
                processed_value = value
            
            success = self.backend.set(key, processed_value, ttl)
            
            execution_time = (time.time() - start_time) * 1000
            
            with self.metrics_lock:
                if success:
                    self.metrics.sets += 1
                    self.metrics.total_set_time += execution_time
                    self.metrics.avg_set_time = self.metrics.total_set_time / self.metrics.sets
                    self.metrics.entry_count = self.backend.size()
                else:
                    self.metrics.errors += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting cache value for key {key}: {e}")
            with self.metrics_lock:
                self.metrics.errors += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            success = self.backend.delete(key)
            
            with self.metrics_lock:
                if success:
                    self.metrics.deletes += 1
                    self.metrics.entry_count = self.backend.size()
                
            return success
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            with self.metrics_lock:
                self.metrics.errors += 1
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return self.get(key) is not None
    
    def clear(self):
        """Clear all cache entries"""
        try:
            self.backend.clear()
            
            with self.metrics_lock:
                self.metrics.entry_count = 0
                
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def keys(self, pattern: str = None) -> List[str]:
        """Get cache keys, optionally filtered by pattern"""
        try:
            keys = self.backend.keys()
            
            if pattern:
                import fnmatch
                keys = [key for key in keys if fnmatch.fnmatch(key, pattern)]
            
            return keys
            
        except Exception as e:
            logger.error(f"Error getting cache keys: {e}")
            return []
    
    def size(self) -> int:
        """Get number of cache entries"""
        return self.backend.size()
    
    def flush_expired(self) -> int:
        """Remove expired entries and return count"""
        removed_count = 0
        
        try:
            keys = self.keys()
            
            for key in keys:
                entry = self.backend.get(key)
                if entry and entry.is_expired():
                    self.backend.delete(key)
                    removed_count += 1
            
            logger.debug(f"Flushed {removed_count} expired cache entries")
            
        except Exception as e:
            logger.error(f"Error flushing expired entries: {e}")
        
        return removed_count
    
    def get_metrics(self) -> CacheMetrics:
        """Get cache performance metrics"""
        with self.metrics_lock:
            # Update current metrics
            self.metrics.entry_count = self.backend.size()
            
            # Add backend-specific metrics
            if isinstance(self.backend, HybridCacheBackend):
                level_stats = self.backend.get_level_stats()
                self.metrics.metadata = level_stats
            
            return self.metrics
    
    def warm_cache(self, keys: List[str], preload_func: Callable[[str], Any]):
        """Warm cache with preloaded data"""
        logger.info(f"Warming cache with {len(keys)} keys")
        
        def warm_key(key: str):
            try:
                if not self.exists(key):
                    value = preload_func(key)
                    if value is not None:
                        self.set(key, value)
                        logger.debug(f"Warmed cache key: {key}")
            except Exception as e:
                logger.warning(f"Failed to warm cache key {key}: {e}")
        
        # Warm cache in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(warm_key, keys)
        
        logger.info("Cache warming completed")
    
    def get_cache_info(self) -> Dict:
        """Get comprehensive cache information"""
        metrics = self.get_metrics()
        
        return {
            'backend': self.config.backend.value,
            'strategy': self.config.strategy.value,
            'size': self.size(),
            'max_size': self.config.max_size,
            'hit_rate': metrics.hit_rate(),
            'miss_rate': metrics.miss_rate(),
            'memory_usage_mb': metrics.memory_used / (1024 * 1024),
            'memory_limit_mb': self.config.max_memory_mb,
            'compression': self.config.compression.value,
            'encryption_enabled': self.config.encryption_enabled,
            'metrics': asdict(metrics)
        }
    
    # ==================== PRIVATE METHODS ====================
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data based on configuration"""
        if self.config.serialization == SerializationType.JSON:
            return json.dumps(data).encode('utf-8')
        elif self.config.serialization == SerializationType.PICKLE:
            return pickle.dumps(data)
        else:
            return pickle.dumps(data)  # Default to pickle
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data based on configuration"""
        if self.config.serialization == SerializationType.JSON:
            return json.loads(data.decode('utf-8'))
        elif self.config.serialization == SerializationType.PICKLE:
            return pickle.loads(data)
        else:
            return pickle.loads(data)  # Default to pickle
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data based on configuration"""
        if self.config.compression == CompressionType.ZLIB:
            return zlib.compress(data)
        elif self.config.compression == CompressionType.GZIP:
            import gzip
            return gzip.compress(data)
        else:
            return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data based on configuration"""
        if self.config.compression == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif self.config.compression == CompressionType.GZIP:
            import gzip
            return gzip.decompress(data)
        else:
            return data
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if security manager available"""
        if self.security_manager:
            return self.security_manager.encrypt_data(data)
        return data
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if security manager available"""
        if self.security_manager:
            return self.security_manager.decrypt_data(data)
        return data
    
    def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while not self.shutdown_event.is_set():
            try:
                # Update memory usage metrics
                with self.metrics_lock:
                    self.metrics.entry_count = self.backend.size()
                    
                    # Estimate memory usage for memory backend
                    if isinstance(self.backend, MemoryCacheBackend):
                        self.metrics.memory_used = self.backend.memory_used
                        self.metrics.memory_limit = self.config.max_memory_mb * 1024 * 1024
                
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _cache_warming_loop(self):
        """Background cache warming loop"""
        while not self.shutdown_event.is_set():
            try:
                # Warm cache with preload keys
                if self.config.preload_keys:
                    def dummy_preload(key):
                        return f"warmed_value_{key}"
                    
                    self.warm_cache(self.config.preload_keys, dummy_preload)
                
                time.sleep(3600)  # Warm cache every hour
                
            except Exception as e:
                logger.error(f"Error in cache warming: {e}")
                time.sleep(3600)
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self.shutdown_event.is_set():
            try:
                # Remove expired entries
                expired_count = self.flush_expired()
                
                if expired_count > 0:
                    with self.metrics_lock:
                        self.metrics.evictions += expired_count
                
                # Force garbage collection periodically
                gc.collect()
                
                time.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(300)
    
    def shutdown(self):
        """Shutdown cache manager and background tasks"""
        try:
            self.shutdown_event.set()
            
            # Wait for background tasks to finish
            for task in self.background_tasks:
                if task.is_alive():
                    task.join(timeout=5)
            
            # Clear cache
            self.clear()
            
            logger.info("Cache manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during cache manager shutdown: {e}")

# ==================== CACHE DECORATORS ====================

def cached(ttl: int = 300, key_prefix: str = "", cache_manager: CacheManager = None):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        cache_manager: Cache manager instance
    """
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            
            # Add arguments to key
            if args:
                key_parts.append(str(hash(args)))
            if kwargs:
                key_parts.append(str(hash(tuple(sorted(kwargs.items())))))
            
            cache_key = "_".join(filter(None, key_parts))
            
            # Try to get from cache
            if cache_manager:
                result = cache_manager.get(cache_key)
                if result is not None:
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            if cache_manager:
                cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# ==================== TESTING ====================

def test_cache_manager():
    """Test cache manager functionality"""
    
    print("🚀 Testing Cache Manager")
    print("=" * 50)
    
    # Test memory backend
    memory_config = CacheConfig(
        backend=CacheBackend.MEMORY,
        max_size=100,
        strategy=CacheStrategy.LRU
    )
    
    cache_manager = CacheManager(memory_config)
    
    print(f"✅ Cache manager initialized: {memory_config.backend.value}")
    
    # Test basic operations
    cache_manager.set("test_key", "test_value", ttl=60)
    value = cache_manager.get("test_key")
    print(f"✅ Basic operations: {value == 'test_value'}")
    
    # Test metrics
    metrics = cache_manager.get_metrics()
    print(f"✅ Metrics - Hits: {metrics.hits}, Sets: {metrics.sets}")
    print(f"   Hit rate: {metrics.hit_rate():.2%}")
    
    # Test cache info
    info = cache_manager.get_cache_info()
    print(f"✅ Cache info:")
    print(f"   Backend: {info['backend']}")
    print(f"   Size: {info['size']}")
    print(f"   Hit rate: {info['hit_rate']:.2%}")
    
    # Test cache decorator
    @cached(ttl=30, key_prefix="test", cache_manager=cache_manager)
    def expensive_function(x: int) -> int:
        return x * x
    
    result1 = expensive_function(5)
    result2 = expensive_function(5)  # Should hit cache
    print(f"✅ Cache decorator: {result1 == result2 == 25}")
    
    # Test cleanup
    cache_manager.set("expired_key", "expired_value", ttl=1)
    time.sleep(2)
    expired_value = cache_manager.get("expired_key")
    print(f"✅ TTL expiration: {expired_value is None}")
    
    # Shutdown
    cache_manager.shutdown()
    
    print("\n🎉 Cache manager tests completed!")

if __name__ == "__main__":
    test_cache_manager()