#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/integrations/__init__.py
Description: Integrations package initialization for NeuroCluster Elite

This module initializes the integrations package and provides a unified interface
for all external integrations including brokers, exchanges, and notification systems.

Features:
- Broker integration management
- Exchange connection handling
- Notification system coordination
- Integration health monitoring
- Connection pooling and retry logic
- Authentication and security management

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== INTEGRATION TYPES ====================

class IntegrationType(Enum):
    """Types of integrations available"""
    BROKER = "broker"
    EXCHANGE = "exchange"
    NOTIFICATION = "notification"
    DATA_PROVIDER = "data_provider"
    ANALYTICS = "analytics"

class IntegrationStatus(Enum):
    """Integration connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"

class AuthMethod(Enum):
    """Authentication methods"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    BASIC_AUTH = "basic_auth"
    CUSTOM = "custom"

# ==================== DATA STRUCTURES ====================

@dataclass
class IntegrationConfig:
    """Configuration for an integration"""
    name: str
    integration_type: IntegrationType
    enabled: bool = False
    
    # Connection settings
    base_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Authentication
    auth_method: AuthMethod = AuthMethod.API_KEY
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    additional_auth: Dict[str, Any] = field(default_factory=dict)
    
    # Rate limiting
    rate_limit: Optional[int] = None  # requests per minute
    burst_limit: Optional[int] = None
    
    # Environment settings
    sandbox_mode: bool = True
    paper_trading: bool = True
    
    # Features
    supports_live_trading: bool = False
    supports_websocket: bool = False
    supports_options: bool = False
    supports_crypto: bool = False
    
    # Metadata
    description: str = ""
    version: str = "1.0.0"
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class IntegrationStatus:
    """Status information for an integration"""
    name: str
    status: IntegrationStatus
    last_connection: Optional[datetime] = None
    last_error: Optional[str] = None
    connection_count: int = 0
    error_count: int = 0
    uptime_percentage: float = 0.0
    latency_ms: float = 0.0
    
    # Performance metrics
    requests_today: int = 0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    
    # Capabilities
    available_features: List[str] = field(default_factory=list)
    supported_assets: List[str] = field(default_factory=list)

# ==================== BASE INTEGRATION CLASS ====================

class BaseIntegration(ABC):
    """Base class for all integrations"""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize integration"""
        self.config = config
        self.status = IntegrationStatus(
            name=config.name,
            status=IntegrationStatus.DISCONNECTED
        )
        self._session = None
        self._connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the integration"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the integration"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if connection is working"""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[str]:
        """Get list of supported capabilities"""
        pass
    
    def is_connected(self) -> bool:
        """Check if integration is connected"""
        return self._connected and self.status.status == IntegrationStatus.CONNECTED
    
    def is_enabled(self) -> bool:
        """Check if integration is enabled"""
        return self.config.enabled
    
    def get_status(self) -> IntegrationStatus:
        """Get current integration status"""
        return self.status
    
    def update_status(self, status: IntegrationStatus, error: Optional[str] = None):
        """Update integration status"""
        self.status.status = status
        if error:
            self.status.last_error = error
            self.status.error_count += 1
        if status == IntegrationStatus.CONNECTED:
            self.status.last_connection = datetime.now()
            self.status.connection_count += 1

# ==================== INTEGRATION MANAGER ====================

class IntegrationManager:
    """Manager for all integrations"""
    
    def __init__(self):
        """Initialize integration manager"""
        self.integrations: Dict[str, BaseIntegration] = {}
        self.configs: Dict[str, IntegrationConfig] = {}
        self._health_check_interval = 60  # seconds
        self._health_check_task = None
    
    def register_integration(self, integration: BaseIntegration):
        """Register an integration"""
        name = integration.config.name
        self.integrations[name] = integration
        self.configs[name] = integration.config
        logger.info(f"Registered integration: {name}")
    
    def unregister_integration(self, name: str):
        """Unregister an integration"""
        if name in self.integrations:
            del self.integrations[name]
            del self.configs[name]
            logger.info(f"Unregistered integration: {name}")
    
    async def connect_all(self):
        """Connect all enabled integrations"""
        connection_tasks = []
        
        for name, integration in self.integrations.items():
            if integration.is_enabled():
                logger.info(f"Connecting to {name}...")
                task = asyncio.create_task(self._connect_integration(integration))
                connection_tasks.append(task)
        
        if connection_tasks:
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)
            
            success_count = sum(1 for result in results if result is True)
            error_count = len(results) - success_count
            
            logger.info(f"Connected {success_count} integrations, {error_count} failed")
    
    async def disconnect_all(self):
        """Disconnect all integrations"""
        disconnection_tasks = []
        
        for name, integration in self.integrations.items():
            if integration.is_connected():
                logger.info(f"Disconnecting from {name}...")
                task = asyncio.create_task(integration.disconnect())
                disconnection_tasks.append(task)
        
        if disconnection_tasks:
            await asyncio.gather(*disconnection_tasks, return_exceptions=True)
            logger.info("All integrations disconnected")
    
    async def _connect_integration(self, integration: BaseIntegration) -> bool:
        """Connect a single integration with error handling"""
        try:
            success = await integration.connect()
            if success:
                logger.info(f"✅ Connected to {integration.config.name}")
                return True
            else:
                logger.warning(f"❌ Failed to connect to {integration.config.name}")
                return False
        except Exception as e:
            logger.error(f"❌ Error connecting to {integration.config.name}: {e}")
            integration.update_status(IntegrationStatus.ERROR, str(e))
            return False
    
    def get_integration(self, name: str) -> Optional[BaseIntegration]:
        """Get integration by name"""
        return self.integrations.get(name)
    
    def get_integrations_by_type(self, integration_type: IntegrationType) -> List[BaseIntegration]:
        """Get all integrations of a specific type"""
        return [
            integration for integration in self.integrations.values()
            if integration.config.integration_type == integration_type
        ]
    
    def get_connected_integrations(self) -> List[BaseIntegration]:
        """Get all connected integrations"""
        return [
            integration for integration in self.integrations.values()
            if integration.is_connected()
        ]
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all integration statuses"""
        total = len(self.integrations)
        connected = len(self.get_connected_integrations())
        enabled = len([i for i in self.integrations.values() if i.is_enabled()])
        
        by_type = {}
        for integration_type in IntegrationType:
            type_integrations = self.get_integrations_by_type(integration_type)
            by_type[integration_type.value] = {
                'total': len(type_integrations),
                'connected': len([i for i in type_integrations if i.is_connected()]),
                'enabled': len([i for i in type_integrations if i.is_enabled()])
            }
        
        return {
            'total_integrations': total,
            'connected_integrations': connected,
            'enabled_integrations': enabled,
            'connection_rate': connected / total if total > 0 else 0,
            'by_type': by_type,
            'last_updated': datetime.now()
        }
    
    async def start_health_monitoring(self):
        """Start health check monitoring"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Started integration health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop health check monitoring"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Stopped integration health monitoring")
    
    async def _health_check_loop(self):
        """Health check loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self._health_check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all connected integrations"""
        health_check_tasks = []
        
        for name, integration in self.integrations.items():
            if integration.is_connected():
                task = asyncio.create_task(self._check_integration_health(integration))
                health_check_tasks.append(task)
        
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_integration_health(self, integration: BaseIntegration):
        """Check health of a single integration"""
        try:
            start_time = datetime.now()
            is_healthy = await integration.test_connection()
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            integration.status.latency_ms = latency
            
            if not is_healthy:
                logger.warning(f"Health check failed for {integration.config.name}")
                integration.update_status(IntegrationStatus.ERROR, "Health check failed")
            
        except Exception as e:
            logger.error(f"Health check error for {integration.config.name}: {e}")
            integration.update_status(IntegrationStatus.ERROR, str(e))

# ==================== INTEGRATION FACTORY ====================

class IntegrationFactory:
    """Factory for creating integrations"""
    
    # Registry of available integration classes
    _integration_classes: Dict[str, type] = {}
    
    @classmethod
    def register_integration_class(cls, name: str, integration_class: type):
        """Register an integration class"""
        cls._integration_classes[name] = integration_class
        logger.info(f"Registered integration class: {name}")
    
    @classmethod
    def create_integration(cls, config: IntegrationConfig) -> Optional[BaseIntegration]:
        """Create an integration instance"""
        integration_class = cls._integration_classes.get(config.name)
        
        if integration_class:
            try:
                return integration_class(config)
            except Exception as e:
                logger.error(f"Failed to create integration {config.name}: {e}")
                return None
        else:
            logger.warning(f"Integration class not found: {config.name}")
            return None
    
    @classmethod
    def get_available_integrations(cls) -> List[str]:
        """Get list of available integration types"""
        return list(cls._integration_classes.keys())

# ==================== DEFAULT CONFIGURATIONS ====================

def get_default_broker_configs() -> Dict[str, IntegrationConfig]:
    """Get default broker configurations"""
    
    configs = {
        "alpaca": IntegrationConfig(
            name="alpaca",
            integration_type=IntegrationType.BROKER,
            enabled=False,  # Disabled by default
            base_url="https://paper-api.alpaca.markets",
            auth_method=AuthMethod.API_KEY,
            sandbox_mode=True,
            paper_trading=True,
            supports_live_trading=True,
            supports_options=False,
            supports_crypto=True,
            description="Alpaca Trading - Commission-free stock and crypto trading"
        ),
        
        "interactive_brokers": IntegrationConfig(
            name="interactive_brokers",
            integration_type=IntegrationType.BROKER,
            enabled=False,  # Disabled by default
            base_url="http://localhost:5000",
            auth_method=AuthMethod.CUSTOM,
            sandbox_mode=True,
            paper_trading=True,
            supports_live_trading=True,
            supports_options=True,
            supports_crypto=False,
            description="Interactive Brokers - Professional trading platform"
        ),
        
        "paper_trading": IntegrationConfig(
            name="paper_trading",
            integration_type=IntegrationType.BROKER,
            enabled=True,  # Enabled by default for safety
            auth_method=AuthMethod.CUSTOM,
            sandbox_mode=True,
            paper_trading=True,
            supports_live_trading=False,
            supports_options=True,
            supports_crypto=True,
            description="Built-in paper trading simulator"
        )
    }
    
    return configs

def get_default_exchange_configs() -> Dict[str, IntegrationConfig]:
    """Get default exchange configurations"""
    
    configs = {
        "binance": IntegrationConfig(
            name="binance",
            integration_type=IntegrationType.EXCHANGE,
            enabled=False,  # Disabled by default
            base_url="https://testnet.binance.vision",
            auth_method=AuthMethod.API_KEY,
            sandbox_mode=True,
            paper_trading=True,
            supports_live_trading=True,
            supports_websocket=True,
            supports_crypto=True,
            rate_limit=1200,  # requests per minute
            description="Binance - Leading cryptocurrency exchange"
        ),
        
        "coinbase": IntegrationConfig(
            name="coinbase",
            integration_type=IntegrationType.EXCHANGE,
            enabled=False,  # Disabled by default
            base_url="https://api-public.sandbox.pro.coinbase.com",
            auth_method=AuthMethod.API_KEY,
            sandbox_mode=True,
            paper_trading=True,
            supports_live_trading=True,
            supports_websocket=True,
            supports_crypto=True,
            rate_limit=10,  # requests per second
            description="Coinbase Pro - Professional cryptocurrency trading"
        )
    }
    
    return configs

def get_default_notification_configs() -> Dict[str, IntegrationConfig]:
    """Get default notification configurations"""
    
    configs = {
        "email": IntegrationConfig(
            name="email",
            integration_type=IntegrationType.NOTIFICATION,
            enabled=False,  # Disabled until SMTP configured
            auth_method=AuthMethod.BASIC_AUTH,
            description="Email notifications via SMTP"
        ),
        
        "discord": IntegrationConfig(
            name="discord",
            integration_type=IntegrationType.NOTIFICATION,
            enabled=False,  # Disabled until webhook configured
            auth_method=AuthMethod.API_KEY,
            supports_websocket=True,
            description="Discord notifications via webhook"
        ),
        
        "telegram": IntegrationConfig(
            name="telegram",
            integration_type=IntegrationType.NOTIFICATION,
            enabled=False,  # Disabled until bot configured
            auth_method=AuthMethod.API_KEY,
            description="Telegram notifications via bot"
        )
    }
    
    return configs

# ==================== GLOBAL INTEGRATION MANAGER ====================

# Global integration manager instance
integration_manager = IntegrationManager()

# ==================== PACKAGE EXPORTS ====================

__all__ = [
    # Enums
    'IntegrationType',
    'IntegrationStatus',
    'AuthMethod',
    
    # Data structures
    'IntegrationConfig',
    'IntegrationStatus',
    
    # Base classes
    'BaseIntegration',
    
    # Manager and factory
    'IntegrationManager',
    'IntegrationFactory',
    
    # Default configs
    'get_default_broker_configs',
    'get_default_exchange_configs',
    'get_default_notification_configs',
    
    # Global instance
    'integration_manager'
]

# ==================== INITIALIZATION ====================

def initialize_integrations():
    """Initialize the integrations package"""
    
    logger.info("🔗 Initializing NeuroCluster Elite integrations package")
    
    # Load default configurations
    broker_configs = get_default_broker_configs()
    exchange_configs = get_default_exchange_configs()
    notification_configs = get_default_notification_configs()
    
    all_configs = {**broker_configs, **exchange_configs, **notification_configs}
    
    logger.info(f"📋 Loaded {len(all_configs)} default integration configurations")
    
    # Register configurations with the global manager
    for name, config in all_configs.items():
        integration_manager.configs[name] = config
    
    logger.info("✅ Integrations package initialized successfully")

# Auto-initialize when package is imported
initialize_integrations()

# ==================== UTILITY FUNCTIONS ====================

def list_available_integrations() -> Dict[str, List[str]]:
    """List all available integrations by type"""
    
    result = {}
    
    for integration_type in IntegrationType:
        integrations = [
            name for name, config in integration_manager.configs.items()
            if config.integration_type == integration_type
        ]
        result[integration_type.value] = integrations
    
    return result

def get_integration_summary() -> str:
    """Get a summary of integration status"""
    
    summary = integration_manager.get_status_summary()
    
    summary_text = f"""
NeuroCluster Elite - Integration Summary
======================================

Total Integrations: {summary['total_integrations']}
Connected: {summary['connected_integrations']}
Enabled: {summary['enabled_integrations']}
Connection Rate: {summary['connection_rate']:.1%}

By Type:
"""
    
    for type_name, type_data in summary['by_type'].items():
        summary_text += f"  {type_name.title()}: {type_data['connected']}/{type_data['total']} connected\n"
    
    summary_text += f"\nLast Updated: {summary['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
    
    return summary_text

# ==================== TESTING FUNCTION ====================

def test_integrations():
    """Test integrations package functionality"""
    
    print("🔗 Testing Integrations Package")
    print("=" * 50)
    
    # Test configuration loading
    broker_configs = get_default_broker_configs()
    exchange_configs = get_default_exchange_configs()
    notification_configs = get_default_notification_configs()
    
    print(f"✅ Loaded {len(broker_configs)} broker configurations")
    print(f"✅ Loaded {len(exchange_configs)} exchange configurations")
    print(f"✅ Loaded {len(notification_configs)} notification configurations")
    
    # Test integration manager
    manager = IntegrationManager()
    status_summary = manager.get_status_summary()
    print(f"✅ Integration manager created: {status_summary['total_integrations']} total integrations")
    
    # Test available integrations listing
    available = list_available_integrations()
    for integration_type, integrations in available.items():
        print(f"✅ {integration_type}: {', '.join(integrations)}")
    
    # Test integration factory
    factory_integrations = IntegrationFactory.get_available_integrations()
    print(f"✅ Factory has {len(factory_integrations)} registered integration classes")
    
    print("\n🎉 Integrations package testing completed!")

if __name__ == "__main__":
    test_integrations()