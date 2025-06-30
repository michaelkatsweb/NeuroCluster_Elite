#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/integrations/notifications/__init__.py
Description: Notification integrations package for NeuroCluster Elite

This module initializes the notification integrations package and provides a unified
interface for all supported notification channels including email, Discord, Telegram,
mobile push notifications, and other alert systems.

Features:
- Multi-channel notification delivery
- Message formatting and templating
- Priority-based routing
- Rate limiting and throttling
- Notification history and tracking
- Template customization
- Rich media support (images, charts, etc.)

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
from abc import ABC, abstractmethod

# Import base integration classes
try:
    from src.integrations import BaseIntegration, IntegrationConfig, IntegrationStatus, IntegrationType
    from src.utils.helpers import format_currency, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== NOTIFICATION ENUMS ====================

class NotificationType(Enum):
    """Types of notifications"""
    EMAIL = "email"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SLACK = "slack"
    SMS = "sms"
    MOBILE_PUSH = "mobile_push"
    WEBHOOK = "webhook"
    VOICE = "voice"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class NotificationCategory(Enum):
    """Categories of notifications"""
    TRADE_EXECUTION = "trade_execution"
    ORDER_STATUS = "order_status"
    MARKET_ALERT = "market_alert"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_WARNING = "risk_warning"
    SYSTEM_STATUS = "system_status"
    STRATEGY_SIGNAL = "strategy_signal"
    NEWS_ALERT = "news_alert"
    TECHNICAL_ANALYSIS = "technical_analysis"
    PRICE_ALERT = "price_alert"

class NotificationStatus(Enum):
    """Status of notification delivery"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    REJECTED = "rejected"
    THROTTLED = "throttled"

# ==================== DATA STRUCTURES ====================

@dataclass
class NotificationTemplate:
    """Template for formatting notifications"""
    name: str
    title_template: str
    message_template: str
    category: NotificationCategory
    
    # Formatting options
    supports_html: bool = False
    supports_markdown: bool = True
    supports_images: bool = False
    supports_attachments: bool = False
    
    # Template variables
    required_variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)
    
    # Media settings
    include_charts: bool = False
    include_logo: bool = True
    color_scheme: str = "default"

@dataclass
class NotificationMessage:
    """Notification message data structure"""
    id: str
    title: str
    message: str
    category: NotificationCategory
    priority: NotificationPriority
    
    # Targeting
    channels: List[NotificationType]
    recipients: List[str] = field(default_factory=list)
    
    # Content
    data: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)
    image_url: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Tracking
    status: NotificationStatus = NotificationStatus.PENDING
    delivery_attempts: int = 0
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None
    
    # Settings
    retry_count: int = 3
    throttle_key: Optional[str] = None

@dataclass
class NotificationRule:
    """Rules for automatic notification triggering"""
    id: str
    name: str
    category: NotificationCategory
    priority: NotificationPriority
    
    # Conditions
    triggers: List[str]  # Event types that trigger this rule
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Actions
    template: str
    channels: List[NotificationType]
    recipients: List[str] = field(default_factory=list)
    
    # Timing
    enabled: bool = True
    cooldown_minutes: int = 0
    last_triggered: Optional[datetime] = None
    
    # Filtering
    filters: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[int] = None  # Max notifications per hour

# ==================== BASE NOTIFICATION CLASS ====================

class BaseNotificationChannel(BaseIntegration):
    """
    Base class for all notification channel integrations
    
    This abstract base class defines the standard interface that all notification
    channels must implement, ensuring consistency across different platforms.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize base notification channel"""
        super().__init__(config)
        self.channel_type = NotificationType(config.name)
        
        # Channel capabilities
        self.supports_html = getattr(config, 'supports_html', False)
        self.supports_markdown = getattr(config, 'supports_markdown', True)
        self.supports_images = getattr(config, 'supports_images', False)
        self.supports_attachments = getattr(config, 'supports_attachments', False)
        self.supports_rich_formatting = getattr(config, 'supports_rich_formatting', False)
        
        # Rate limiting
        self.rate_limit = getattr(config, 'rate_limit', 60)  # messages per hour
        self.message_timestamps: List[datetime] = []
        
        # Templates
        self.templates: Dict[str, NotificationTemplate] = {}
        self.load_default_templates()
        
        # Message queue
        self.message_queue: List[NotificationMessage] = []
        self.processing_queue = False
    
    # ==================== ABSTRACT METHODS ====================
    
    @abstractmethod
    async def send_message(self, message: NotificationMessage) -> bool:
        """Send a notification message"""
        pass
    
    @abstractmethod
    async def format_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Format message for the specific channel"""
        pass
    
    # ==================== COMMON METHODS ====================
    
    def load_default_templates(self):
        """Load default notification templates"""
        
        # Trade execution template
        self.templates['trade_execution'] = NotificationTemplate(
            name="Trade Execution",
            title_template="Trade Executed: {side} {quantity} {symbol}",
            message_template="**Trade Details:**\n"
                           "• Symbol: {symbol}\n"
                           "• Side: {side}\n"
                           "• Quantity: {quantity}\n"
                           "• Price: ${price}\n"
                           "• Total: ${total_value}\n"
                           "• Time: {timestamp}",
            category=NotificationCategory.TRADE_EXECUTION,
            supports_markdown=True,
            required_variables=['symbol', 'side', 'quantity', 'price', 'total_value', 'timestamp']
        )
        
        # Price alert template
        self.templates['price_alert'] = NotificationTemplate(
            name="Price Alert",
            title_template="Price Alert: {symbol} {direction} ${price}",
            message_template="**Price Alert Triggered:**\n"
                           "• Symbol: {symbol}\n"
                           "• Current Price: ${current_price}\n"
                           "• Trigger Price: ${trigger_price}\n"
                           "• Change: {change_percent}%\n"
                           "• Direction: {direction}",
            category=NotificationCategory.PRICE_ALERT,
            supports_markdown=True,
            required_variables=['symbol', 'current_price', 'trigger_price', 'change_percent', 'direction']
        )
        
        # Strategy signal template
        self.templates['strategy_signal'] = NotificationTemplate(
            name="Strategy Signal",
            title_template="Strategy Signal: {strategy_name} - {signal_type}",
            message_template="**Strategy Signal Generated:**\n"
                           "• Strategy: {strategy_name}\n"
                           "• Signal: {signal_type}\n"
                           "• Symbol: {symbol}\n"
                           "• Confidence: {confidence}%\n"
                           "• Reasoning: {reasoning}",
            category=NotificationCategory.STRATEGY_SIGNAL,
            supports_markdown=True,
            required_variables=['strategy_name', 'signal_type', 'symbol', 'confidence', 'reasoning']
        )
        
        # Portfolio update template
        self.templates['portfolio_update'] = NotificationTemplate(
            name="Portfolio Update",
            title_template="Portfolio Update - Total Value: ${total_value}",
            message_template="**Portfolio Summary:**\n"
                           "• Total Value: ${total_value}\n"
                           "• Daily P&L: ${daily_pnl} ({daily_pnl_percent}%)\n"
                           "• Open Positions: {open_positions}\n"
                           "• Cash Balance: ${cash_balance}\n"
                           "• Updated: {timestamp}",
            category=NotificationCategory.PORTFOLIO_UPDATE,
            supports_markdown=True,
            required_variables=['total_value', 'daily_pnl', 'daily_pnl_percent', 'open_positions', 'cash_balance', 'timestamp']
        )
        
        # Risk warning template
        self.templates['risk_warning'] = NotificationTemplate(
            name="Risk Warning",
            title_template="⚠️ Risk Warning: {warning_type}",
            message_template="**Risk Warning:**\n"
                           "• Type: {warning_type}\n"
                           "• Severity: {severity}\n"
                           "• Description: {description}\n"
                           "• Recommended Action: {recommendation}\n"
                           "• Time: {timestamp}",
            category=NotificationCategory.RISK_WARNING,
            supports_markdown=True,
            required_variables=['warning_type', 'severity', 'description', 'recommendation', 'timestamp']
        )
        
        logger.info(f"📋 Loaded {len(self.templates)} notification templates")
    
    def create_message_from_template(self, template_name: str, data: Dict[str, Any],
                                   priority: NotificationPriority = NotificationPriority.NORMAL,
                                   channels: Optional[List[NotificationType]] = None) -> NotificationMessage:
        """Create a notification message from a template"""
        
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        # Check required variables
        missing_vars = [var for var in template.required_variables if var not in data]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Format title and message
        try:
            title = template.title_template.format(**data)
            message = template.message_template.format(**data)
        except KeyError as e:
            raise ValueError(f"Template formatting error: missing variable {e}")
        
        # Create message
        return NotificationMessage(
            id=str(uuid.uuid4()),
            title=title,
            message=message,
            category=template.category,
            priority=priority,
            channels=channels or [self.channel_type],
            data=data.copy()
        )
    
    def check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        
        # Remove old timestamps (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        self.message_timestamps = [ts for ts in self.message_timestamps if ts > cutoff]
        
        # Check if we're under the limit
        return len(self.message_timestamps) < self.rate_limit
    
    def record_message(self):
        """Record a new message timestamp"""
        self.message_timestamps.append(datetime.now())
    
    async def queue_message(self, message: NotificationMessage):
        """Add message to queue for processing"""
        self.message_queue.append(message)
        
        # Start processing if not already running
        if not self.processing_queue:
            asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process the message queue"""
        self.processing_queue = True
        
        try:
            while self.message_queue:
                message = self.message_queue.pop(0)
                
                # Check if message has expired
                if message.expires_at and datetime.now() > message.expires_at:
                    message.status = NotificationStatus.REJECTED
                    logger.warning(f"⏰ Message {message.id} expired, skipping")
                    continue
                
                # Check rate limiting
                if not self.check_rate_limit():
                    # Re-queue the message for later
                    self.message_queue.insert(0, message)
                    message.status = NotificationStatus.THROTTLED
                    logger.warning(f"⏸️ Rate limit reached, throttling message {message.id}")
                    await asyncio.sleep(60)  # Wait 1 minute
                    continue
                
                # Attempt to send
                try:
                    success = await self.send_message(message)
                    
                    if success:
                        message.status = NotificationStatus.SENT
                        message.delivered_at = datetime.now()
                        self.record_message()
                        logger.info(f"✅ Message {message.id} sent successfully")
                    else:
                        message.delivery_attempts += 1
                        
                        if message.delivery_attempts >= message.retry_count:
                            message.status = NotificationStatus.FAILED
                            logger.error(f"❌ Message {message.id} failed after {message.retry_count} attempts")
                        else:
                            # Re-queue for retry
                            self.message_queue.append(message)
                            logger.warning(f"🔄 Retrying message {message.id} (attempt {message.delivery_attempts + 1})")
                            await asyncio.sleep(5)  # Wait before retry
                            
                except Exception as e:
                    message.delivery_attempts += 1
                    message.error_message = str(e)
                    
                    if message.delivery_attempts >= message.retry_count:
                        message.status = NotificationStatus.FAILED
                        logger.error(f"❌ Message {message.id} failed with error: {e}")
                    else:
                        # Re-queue for retry
                        self.message_queue.append(message)
                        logger.warning(f"🔄 Retrying message {message.id} after error: {e}")
                        await asyncio.sleep(5)
                
                # Small delay between messages
                await asyncio.sleep(0.1)
                
        finally:
            self.processing_queue = False
    
    # ==================== HELPER METHODS ====================
    
    def format_currency(self, amount: float, currency: str = "USD") -> str:
        """Format currency amount"""
        return f"${amount:,.2f}" if currency == "USD" else f"{amount:,.2f} {currency}"
    
    def format_percentage(self, value: float) -> str:
        """Format percentage value"""
        return f"{value:+.2f}%"
    
    def format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for display"""
        return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    def truncate_message(self, message: str, max_length: int) -> str:
        """Truncate message to maximum length"""
        if len(message) <= max_length:
            return message
        return message[:max_length - 3] + "..."

# ==================== NOTIFICATION FACTORY ====================

class NotificationFactory:
    """Factory for creating notification channel instances"""
    
    # Registry of available notification classes
    _channel_classes: Dict[NotificationType, Type[BaseNotificationChannel]] = {}
    
    @classmethod
    def register_channel(cls, channel_type: NotificationType, channel_class: Type[BaseNotificationChannel]):
        """Register a notification channel class"""
        cls._channel_classes[channel_type] = channel_class
        logger.info(f"Registered notification channel: {channel_type.value}")
    
    @classmethod
    def create_channel(cls, config: IntegrationConfig) -> Optional[BaseNotificationChannel]:
        """Create a notification channel instance"""
        try:
            channel_type = NotificationType(config.name)
            channel_class = cls._channel_classes.get(channel_type)
            
            if channel_class:
                return channel_class(config)
            else:
                logger.error(f"Notification channel class not found: {channel_type.value}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create notification channel {config.name}: {e}")
            return None
    
    @classmethod
    def get_available_channels(cls) -> List[NotificationType]:
        """Get list of available notification channel types"""
        return list(cls._channel_classes.keys())

# ==================== NOTIFICATION MANAGER ====================

class NotificationManager:
    """Manages multiple notification channels and routing"""
    
    def __init__(self):
        self.channels: Dict[str, BaseNotificationChannel] = {}
        self.rules: Dict[str, NotificationRule] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        
        # Message history
        self.message_history: List[NotificationMessage] = []
        self.max_history = 1000
        
        # Global settings
        self.enabled = True
        self.global_rate_limit = 100  # messages per hour across all channels
        self.global_message_timestamps: List[datetime] = []
    
    async def add_channel(self, name: str, config: IntegrationConfig) -> bool:
        """Add a notification channel"""
        try:
            channel = NotificationFactory.create_channel(config)
            if channel:
                # Test connection
                if await channel.connect():
                    self.channels[name] = channel
                    
                    # Merge templates
                    self.templates.update(channel.templates)
                    
                    logger.info(f"✅ Added notification channel: {name}")
                    return True
                else:
                    logger.error(f"❌ Failed to connect notification channel: {name}")
                    return False
            else:
                logger.error(f"❌ Failed to create notification channel: {name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error adding notification channel {name}: {e}")
            return False
    
    async def remove_channel(self, name: str) -> bool:
        """Remove a notification channel"""
        try:
            if name in self.channels:
                await self.channels[name].disconnect()
                del self.channels[name]
                logger.info(f"✅ Removed notification channel: {name}")
                return True
            else:
                logger.warning(f"⚠️ Notification channel not found: {name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error removing notification channel {name}: {e}")
            return False
    
    async def send_notification(self, message: NotificationMessage) -> Dict[str, bool]:
        """Send notification to specified channels"""
        results = {}
        
        if not self.enabled:
            logger.warning("⚠️ Notifications disabled globally")
            return results
        
        # Check global rate limiting
        if not self._check_global_rate_limit():
            logger.warning("⚠️ Global rate limit exceeded")
            message.status = NotificationStatus.THROTTLED
            return results
        
        # Send to each specified channel
        for channel_type in message.channels:
            # Find channel instance
            channel_instance = None
            for name, channel in self.channels.items():
                if channel.channel_type == channel_type:
                    channel_instance = channel
                    break
            
            if channel_instance:
                try:
                    await channel_instance.queue_message(message)
                    results[channel_type.value] = True
                except Exception as e:
                    logger.error(f"❌ Error sending to {channel_type.value}: {e}")
                    results[channel_type.value] = False
            else:
                logger.warning(f"⚠️ No {channel_type.value} channel configured")
                results[channel_type.value] = False
        
        # Add to history
        self._add_to_history(message)
        
        # Record global timestamp
        self.global_message_timestamps.append(datetime.now())
        
        return results
    
    async def send_template_notification(self, template_name: str, data: Dict[str, Any],
                                       priority: NotificationPriority = NotificationPriority.NORMAL,
                                       channels: Optional[List[NotificationType]] = None) -> Dict[str, bool]:
        """Send notification using a template"""
        
        # Use first available channel to create message if none specified
        if not channels:
            channels = [list(self.channels.values())[0].channel_type] if self.channels else []
        
        if not channels:
            logger.error("❌ No notification channels available")
            return {}
        
        # Get template from any channel that has it
        template_channel = None
        for channel in self.channels.values():
            if template_name in channel.templates:
                template_channel = channel
                break
        
        if not template_channel:
            logger.error(f"❌ Template '{template_name}' not found")
            return {}
        
        # Create message
        message = template_channel.create_message_from_template(
            template_name, data, priority, channels
        )
        
        # Send notification
        return await self.send_notification(message)
    
    def add_rule(self, rule: NotificationRule):
        """Add a notification rule"""
        self.rules[rule.id] = rule
        logger.info(f"📋 Added notification rule: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove a notification rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"📋 Removed notification rule: {rule_id}")
    
    async def process_event(self, event_type: str, event_data: Dict[str, Any]):
        """Process an event and trigger applicable rules"""
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check if event type matches
            if event_type not in rule.triggers:
                continue
            
            # Check cooldown
            if rule.last_triggered and rule.cooldown_minutes > 0:
                time_diff = (datetime.now() - rule.last_triggered).total_seconds() / 60
                if time_diff < rule.cooldown_minutes:
                    continue
            
            # Check conditions
            if not self._evaluate_conditions(rule.conditions, event_data):
                continue
            
            # Check rate limit
            if rule.rate_limit and self._check_rule_rate_limit(rule):
                continue
            
            # Create and send notification
            try:
                await self.send_template_notification(
                    rule.template,
                    event_data,
                    rule.priority,
                    rule.channels
                )
                
                rule.last_triggered = datetime.now()
                logger.info(f"📋 Triggered notification rule: {rule.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing rule {rule.name}: {e}")
    
    def _check_global_rate_limit(self) -> bool:
        """Check global rate limiting"""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        self.global_message_timestamps = [ts for ts in self.global_message_timestamps if ts > cutoff]
        return len(self.global_message_timestamps) < self.global_rate_limit
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate rule conditions against event data"""
        # Simple condition evaluation - can be extended
        for key, expected_value in conditions.items():
            if key not in data:
                return False
            
            actual_value = data[key]
            
            if isinstance(expected_value, dict):
                # Handle operators like {"gt": 100}, {"eq": "value"}
                for operator, value in expected_value.items():
                    if operator == "gt" and actual_value <= value:
                        return False
                    elif operator == "lt" and actual_value >= value:
                        return False
                    elif operator == "eq" and actual_value != value:
                        return False
                    elif operator == "ne" and actual_value == value:
                        return False
            else:
                # Direct equality check
                if actual_value != expected_value:
                    return False
        
        return True
    
    def _check_rule_rate_limit(self, rule: NotificationRule) -> bool:
        """Check if rule has exceeded its rate limit"""
        # Implementation would track rule-specific message counts
        return False  # Simplified for now
    
    def _add_to_history(self, message: NotificationMessage):
        """Add message to history"""
        self.message_history.append(message)
        
        # Trim history if too long
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
    
    def get_channel_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all notification channels"""
        status = {}
        for name, channel in self.channels.items():
            status[name] = {
                'type': channel.channel_type.value,
                'status': channel.status.value,
                'connected': channel.status == IntegrationStatus.CONNECTED,
                'rate_limit': channel.rate_limit,
                'messages_sent': len(channel.message_timestamps),
                'queue_size': len(channel.message_queue)
            }
        return status

# ==================== DEFAULT CONFIGURATIONS ====================

def get_default_notification_configs() -> Dict[str, IntegrationConfig]:
    """Get default notification channel configurations"""
    
    configs = {
        "email": IntegrationConfig(
            name="email",
            integration_type=IntegrationType.NOTIFICATION,
            enabled=False,  # Disabled until SMTP configured
            description="Email notifications via SMTP",
            rate_limit=30  # 30 emails per hour
        ),
        
        "discord": IntegrationConfig(
            name="discord",
            integration_type=IntegrationType.NOTIFICATION,
            enabled=False,  # Disabled until webhook configured
            description="Discord notifications via webhooks",
            rate_limit=50  # 50 messages per hour
        ),
        
        "telegram": IntegrationConfig(
            name="telegram",
            integration_type=IntegrationType.NOTIFICATION,
            enabled=False,  # Disabled until bot configured
            description="Telegram notifications via bot API",
            rate_limit=30  # 30 messages per hour
        ),
        
        "mobile_push": IntegrationConfig(
            name="mobile_push",
            integration_type=IntegrationType.NOTIFICATION,
            enabled=False,  # Disabled until push service configured
            description="Mobile push notifications",
            rate_limit=20  # 20 pushes per hour
        )
    }
    
    return configs

# ==================== INITIALIZATION ====================

def register_notification_channels():
    """Register all available notification channel implementations"""
    try:
        # Import notification implementations
        try:
            from .email_alerts import EmailNotificationChannel
            NotificationFactory.register_channel(NotificationType.EMAIL, EmailNotificationChannel)
        except ImportError:
            logger.debug("Email notification channel not available")
        
        try:
            from .discord_bot import DiscordNotificationChannel
            NotificationFactory.register_channel(NotificationType.DISCORD, DiscordNotificationChannel)
        except ImportError:
            logger.debug("Discord notification channel not available")
        
        try:
            from .telegram_bot import TelegramNotificationChannel
            NotificationFactory.register_channel(NotificationType.TELEGRAM, TelegramNotificationChannel)
        except ImportError:
            logger.debug("Telegram notification channel not available")
        
        try:
            from .mobile_push import MobilePushNotificationChannel
            NotificationFactory.register_channel(NotificationType.MOBILE_PUSH, MobilePushNotificationChannel)
        except ImportError:
            logger.debug("Mobile push notification channel not available")
            
        logger.info(f"✅ Registered {len(NotificationFactory.get_available_channels())} notification channel(s)")
        
    except Exception as e:
        logger.error(f"❌ Error registering notification channels: {e}")

# Register channels on import
register_notification_channels()

# ==================== EXPORTS ====================

__all__ = [
    # Enums
    'NotificationType',
    'NotificationPriority',
    'NotificationCategory',
    'NotificationStatus',
    
    # Data structures
    'NotificationTemplate',
    'NotificationMessage',
    'NotificationRule',
    
    # Classes
    'BaseNotificationChannel',
    'NotificationFactory',
    'NotificationManager',
    
    # Functions
    'get_default_notification_configs',
    'register_notification_channels'
]