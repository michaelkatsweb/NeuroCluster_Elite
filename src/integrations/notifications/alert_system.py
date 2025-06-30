#!/usr/bin/env python3
"""
File: alert_system.py
Path: NeuroCluster-Elite/src/integrations/notifications/alert_system.py
Description: Advanced alert and notification system for NeuroCluster Elite

This module provides comprehensive alert capabilities through multiple channels
including Discord, Telegram, email, SMS, Slack, and mobile push notifications.

Features:
- Multi-channel alert delivery
- Priority-based alert routing
- Template-based message formatting
- Rate limiting and throttling
- Alert history and analytics
- Custom alert rules and conditions

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
import smtplib
import json
import time
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.image import MimeImage
import logging
import os
import re
from pathlib import Path

# Import from our modules
from src.utils.logger import get_enhanced_logger, LogCategory
from src.utils.helpers import format_currency, format_percentage, format_timestamp
from src.core.neurocluster_elite import RegimeType, AssetType

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.SYSTEM)

# ==================== ENUMS AND DATA STRUCTURES ====================

class AlertType(Enum):
    """Types of alerts"""
    PRICE_ALERT = "price_alert"
    REGIME_CHANGE = "regime_change"
    TRADING_SIGNAL = "trading_signal"
    PORTFOLIO_UPDATE = "portfolio_update"
    SYSTEM_STATUS = "system_status"
    ERROR_ALERT = "error_alert"
    PERFORMANCE_ALERT = "performance_alert"
    SECURITY_ALERT = "security_alert"

class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SLACK = "slack"
    SMS = "sms"
    MOBILE_PUSH = "mobile_push"
    WEBHOOK = "webhook"
    CONSOLE = "console"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime
    
    # Optional data
    symbol: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    channels: List[NotificationChannel] = field(default_factory=list)
    
    # Delivery tracking
    sent_channels: List[NotificationChannel] = field(default_factory=list)
    failed_channels: List[NotificationChannel] = field(default_factory=list)
    delivery_attempts: int = 0
    delivered: bool = False

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    alert_type: AlertType
    conditions: Dict[str, Any]
    channels: List[NotificationChannel]
    priority: AlertPriority = AlertPriority.NORMAL
    enabled: bool = True
    
    # Rate limiting
    rate_limit_minutes: int = 0  # 0 = no limit
    last_triggered: Optional[datetime] = None

@dataclass
class ChannelConfig:
    """Notification channel configuration"""
    channel: NotificationChannel
    enabled: bool
    config: Dict[str, Any]
    rate_limit_per_hour: int = 60  # Max alerts per hour
    current_hour_count: int = 0
    hour_reset_time: datetime = field(default_factory=datetime.now)

# ==================== NOTIFICATION CHANNELS ====================

class BaseNotificationChannel:
    """Base class for notification channels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.rate_limit = config.get('rate_limit_per_hour', 60)
        self.sent_count = 0
        self.last_reset = datetime.now()
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        if not self.enabled:
            return False
        
        if not self._check_rate_limit():
            logger.warning(f"Rate limit exceeded for {self.__class__.__name__}")
            return False
        
        try:
            success = await self._send_message(alert)
            if success:
                self.sent_count += 1
            return success
        except Exception as e:
            logger.error(f"Error sending alert via {self.__class__.__name__}: {e}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """Check if within rate limits"""
        now = datetime.now()
        
        # Reset counter every hour
        if (now - self.last_reset).total_seconds() > 3600:
            self.sent_count = 0
            self.last_reset = now
        
        return self.sent_count < self.rate_limit
    
    async def _send_message(self, alert: Alert) -> bool:
        """Send message - to be implemented by subclasses"""
        raise NotImplementedError

class DiscordChannel(BaseNotificationChannel):
    """Discord webhook notification channel"""
    
    async def _send_message(self, alert: Alert) -> bool:
        """Send alert to Discord webhook"""
        
        webhook_url = self.config.get('webhook_url')
        if not webhook_url:
            logger.error("Discord webhook URL not configured")
            return False
        
        # Create Discord embed
        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": self._get_color_for_priority(alert.priority),
            "timestamp": alert.timestamp.isoformat(),
            "fields": [
                {
                    "name": "Priority",
                    "value": alert.priority.value.upper(),
                    "inline": True
                },
                {
                    "name": "Type",
                    "value": alert.alert_type.value.replace('_', ' ').title(),
                    "inline": True
                }
            ]
        }
        
        # Add symbol field if available
        if alert.symbol:
            embed["fields"].append({
                "name": "Symbol",
                "value": alert.symbol,
                "inline": True
            })
        
        # Add additional data fields
        if alert.data:
            for key, value in alert.data.items():
                if len(embed["fields"]) < 25:  # Discord limit
                    embed["fields"].append({
                        "name": key.replace('_', ' ').title(),
                        "value": str(value),
                        "inline": True
                    })
        
        payload = {
            "username": "NeuroCluster Elite",
            "embeds": [embed]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 204
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")
            return False
    
    def _get_color_for_priority(self, priority: AlertPriority) -> int:
        """Get Discord embed color for priority"""
        colors = {
            AlertPriority.LOW: 0x808080,      # Gray
            AlertPriority.NORMAL: 0x0099ff,   # Blue
            AlertPriority.HIGH: 0xff9900,     # Orange
            AlertPriority.URGENT: 0xff3300,   # Red
            AlertPriority.CRITICAL: 0x990000  # Dark Red
        }
        return colors.get(priority, 0x0099ff)

class TelegramChannel(BaseNotificationChannel):
    """Telegram bot notification channel"""
    
    async def _send_message(self, alert: Alert) -> bool:
        """Send alert to Telegram"""
        
        bot_token = self.config.get('bot_token')
        chat_id = self.config.get('chat_id')
        
        if not bot_token or not chat_id:
            logger.error("Telegram bot token or chat ID not configured")
            return False
        
        # Format message for Telegram
        message = f"🚨 *{alert.title}*\n\n"
        message += f"{alert.message}\n\n"
        message += f"*Priority:* {alert.priority.value.upper()}\n"
        message += f"*Type:* {alert.alert_type.value.replace('_', ' ').title()}\n"
        
        if alert.symbol:
            message += f"*Symbol:* {alert.symbol}\n"
        
        message += f"*Time:* {format_timestamp(alert.timestamp)}"
        
        # Add data fields
        if alert.data:
            message += "\n\n*Details:*\n"
            for key, value in alert.data.items():
                if len(message) < 4000:  # Telegram message limit
                    message += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Telegram API error: {e}")
            return False

class EmailChannel(BaseNotificationChannel):
    """Email notification channel"""
    
    async def _send_message(self, alert: Alert) -> bool:
        """Send alert via email"""
        
        smtp_host = self.config.get('smtp_host')
        smtp_port = self.config.get('smtp_port', 587)
        username = self.config.get('username')
        password = self.config.get('password')
        from_email = self.config.get('from_email', username)
        to_email = self.config.get('to_email')
        
        if not all([smtp_host, username, password, to_email]):
            logger.error("Email configuration incomplete")
            return False
        
        try:
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = f"[NeuroCluster Elite] {alert.title}"
            msg['From'] = from_email
            msg['To'] = to_email
            
            # Create HTML email content
            html_content = self._create_html_email(alert)
            
            # Create text version
            text_content = self._create_text_email(alert)
            
            # Attach both versions
            msg.attach(MimeText(text_content, 'plain'))
            msg.attach(MimeText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Email sending error: {e}")
            return False
    
    def _create_html_email(self, alert: Alert) -> str:
        """Create HTML email content"""
        
        priority_colors = {
            AlertPriority.LOW: '#808080',
            AlertPriority.NORMAL: '#0099ff',
            AlertPriority.HIGH: '#ff9900',
            AlertPriority.URGENT: '#ff3300',
            AlertPriority.CRITICAL: '#990000'
        }
        
        color = priority_colors.get(alert.priority, '#0099ff')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ margin: 20px 0; }}
                .details {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .field {{ margin: 5px 0; }}
                .label {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 NeuroCluster Elite Alert</h1>
                <h2>{alert.title}</h2>
            </div>
            
            <div class="content">
                <p>{alert.message}</p>
            </div>
            
            <div class="details">
                <div class="field">
                    <span class="label">Priority:</span> {alert.priority.value.upper()}
                </div>
                <div class="field">
                    <span class="label">Type:</span> {alert.alert_type.value.replace('_', ' ').title()}
                </div>
                <div class="field">
                    <span class="label">Time:</span> {format_timestamp(alert.timestamp)}
                </div>
        """
        
        if alert.symbol:
            html += f"""
                <div class="field">
                    <span class="label">Symbol:</span> {alert.symbol}
                </div>
            """
        
        if alert.data:
            html += "<hr><h3>Additional Details:</h3>"
            for key, value in alert.data.items():
                html += f"""
                <div class="field">
                    <span class="label">{key.replace('_', ' ').title()}:</span> {value}
                </div>
                """
        
        html += """
            </div>
            
            <p><em>This alert was sent by NeuroCluster Elite Trading Platform</em></p>
        </body>
        </html>
        """
        
        return html
    
    def _create_text_email(self, alert: Alert) -> str:
        """Create plain text email content"""
        
        text = f"NeuroCluster Elite Alert\n"
        text += "=" * 30 + "\n\n"
        text += f"Title: {alert.title}\n\n"
        text += f"Message: {alert.message}\n\n"
        text += f"Priority: {alert.priority.value.upper()}\n"
        text += f"Type: {alert.alert_type.value.replace('_', ' ').title()}\n"
        text += f"Time: {format_timestamp(alert.timestamp)}\n"
        
        if alert.symbol:
            text += f"Symbol: {alert.symbol}\n"
        
        if alert.data:
            text += "\nAdditional Details:\n"
            text += "-" * 20 + "\n"
            for key, value in alert.data.items():
                text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        text += "\n---\nThis alert was sent by NeuroCluster Elite Trading Platform"
        
        return text

class SlackChannel(BaseNotificationChannel):
    """Slack webhook notification channel"""
    
    async def _send_message(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        
        webhook_url = self.config.get('webhook_url')
        if not webhook_url:
            logger.error("Slack webhook URL not configured")
            return False
        
        # Create Slack message
        color = {
            AlertPriority.LOW: "#808080",
            AlertPriority.NORMAL: "#0099ff",
            AlertPriority.HIGH: "#ff9900",
            AlertPriority.URGENT: "#ff3300",
            AlertPriority.CRITICAL: "#990000"
        }.get(alert.priority, "#0099ff")
        
        fields = [
            {
                "title": "Priority",
                "value": alert.priority.value.upper(),
                "short": True
            },
            {
                "title": "Type",
                "value": alert.alert_type.value.replace('_', ' ').title(),
                "short": True
            }
        ]
        
        if alert.symbol:
            fields.append({
                "title": "Symbol",
                "value": alert.symbol,
                "short": True
            })
        
        attachment = {
            "color": color,
            "title": alert.title,
            "text": alert.message,
            "fields": fields,
            "ts": int(alert.timestamp.timestamp())
        }
        
        payload = {
            "username": "NeuroCluster Elite",
            "icon_emoji": ":chart_with_upwards_trend:",
            "attachments": [attachment]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Slack webhook error: {e}")
            return False

class ConsoleChannel(BaseNotificationChannel):
    """Console/log notification channel"""
    
    async def _send_message(self, alert: Alert) -> bool:
        """Print alert to console"""
        
        # Create formatted console message
        priority_emojis = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.NORMAL: "📢",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.URGENT: "🚨",
            AlertPriority.CRITICAL: "🔥"
        }
        
        emoji = priority_emojis.get(alert.priority, "📢")
        
        print(f"\n{emoji} ALERT: {alert.title}")
        print("=" * 50)
        print(f"Message: {alert.message}")
        print(f"Priority: {alert.priority.value.upper()}")
        print(f"Type: {alert.alert_type.value.replace('_', ' ').title()}")
        print(f"Time: {format_timestamp(alert.timestamp)}")
        
        if alert.symbol:
            print(f"Symbol: {alert.symbol}")
        
        if alert.data:
            print("\nDetails:")
            for key, value in alert.data.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("=" * 50)
        
        return True

# ==================== MAIN ALERT SYSTEM ====================

class AdvancedAlertSystem:
    """
    Advanced alert and notification system
    
    Features:
    - Multi-channel alert delivery
    - Priority-based routing
    - Rate limiting and throttling
    - Template-based messaging
    - Alert rules and conditions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize alert system"""
        
        self.config = config or {}
        
        # Initialize notification channels
        self.channels = self._initialize_channels()
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Alert] = []
        self.failed_alerts: List[Alert] = []
        
        # Performance tracking
        self.stats = {
            'total_alerts': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'channels_used': {},
            'alert_types': {}
        }
        
        logger.info("🔔 Advanced Alert System initialized")
    
    def _initialize_channels(self) -> Dict[NotificationChannel, BaseNotificationChannel]:
        """Initialize notification channels"""
        
        channels = {}
        
        # Discord
        discord_config = self.config.get('discord', {})
        if discord_config.get('enabled', False):
            channels[NotificationChannel.DISCORD] = DiscordChannel(discord_config)
        
        # Telegram
        telegram_config = self.config.get('telegram', {})
        if telegram_config.get('enabled', False):
            channels[NotificationChannel.TELEGRAM] = TelegramChannel(telegram_config)
        
        # Email
        email_config = self.config.get('email', {})
        if email_config.get('enabled', False):
            channels[NotificationChannel.EMAIL] = EmailChannel(email_config)
        
        # Slack
        slack_config = self.config.get('slack', {})
        if slack_config.get('enabled', False):
            channels[NotificationChannel.SLACK] = SlackChannel(slack_config)
        
        # Console (always enabled)
        console_config = self.config.get('console', {'enabled': True})
        channels[NotificationChannel.CONSOLE] = ConsoleChannel(console_config)
        
        logger.info(f"✅ Initialized {len(channels)} notification channels")
        return channels
    
    async def send_alert(self, alert_type: AlertType, title: str, message: str,
                        priority: AlertPriority = AlertPriority.NORMAL,
                        symbol: str = None, data: Dict[str, Any] = None,
                        channels: List[NotificationChannel] = None) -> str:
        """
        Send an alert through specified channels
        
        Args:
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            priority: Alert priority
            symbol: Related symbol (optional)
            data: Additional data (optional)
            channels: Specific channels to use (optional)
            
        Returns:
            Alert ID
        """
        
        # Create alert
        alert_id = f"alert_{int(time.time())}_{len(self.alert_history)}"
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            timestamp=datetime.now(),
            symbol=symbol,
            data=data or {},
            channels=channels or self._get_default_channels(priority)
        )
        
        # Send to channels
        await self._deliver_alert(alert)
        
        # Store in history
        self.alert_history.append(alert)
        self._update_stats(alert)
        
        # Keep limited history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        logger.info(f"🔔 Alert sent: {title} ({alert_id})")
        return alert_id
    
    async def _deliver_alert(self, alert: Alert):
        """Deliver alert to all specified channels"""
        
        delivery_tasks = []
        
        for channel_type in alert.channels:
            if channel_type in self.channels:
                channel = self.channels[channel_type]
                task = asyncio.create_task(
                    self._send_to_channel(alert, channel_type, channel)
                )
                delivery_tasks.append(task)
        
        # Wait for all deliveries to complete
        if delivery_tasks:
            results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
            
            # Check results
            for i, result in enumerate(results):
                channel_type = alert.channels[i]
                if isinstance(result, Exception):
                    logger.error(f"Error delivering to {channel_type.value}: {result}")
                    alert.failed_channels.append(channel_type)
                elif result:
                    alert.sent_channels.append(channel_type)
                else:
                    alert.failed_channels.append(channel_type)
            
            alert.delivered = len(alert.sent_channels) > 0
        
        alert.delivery_attempts += 1
    
    async def _send_to_channel(self, alert: Alert, channel_type: NotificationChannel,
                              channel: BaseNotificationChannel) -> bool:
        """Send alert to specific channel"""
        
        try:
            success = await channel.send_alert(alert)
            
            if success:
                logger.debug(f"✅ Alert delivered to {channel_type.value}")
            else:
                logger.warning(f"❌ Failed to deliver alert to {channel_type.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Exception delivering to {channel_type.value}: {e}")
            return False
    
    def _get_default_channels(self, priority: AlertPriority) -> List[NotificationChannel]:
        """Get default channels based on priority"""
        
        # Priority-based channel selection
        if priority == AlertPriority.CRITICAL:
            return [NotificationChannel.EMAIL, NotificationChannel.DISCORD, 
                   NotificationChannel.TELEGRAM, NotificationChannel.CONSOLE]
        elif priority == AlertPriority.URGENT:
            return [NotificationChannel.DISCORD, NotificationChannel.TELEGRAM, 
                   NotificationChannel.CONSOLE]
        elif priority == AlertPriority.HIGH:
            return [NotificationChannel.DISCORD, NotificationChannel.CONSOLE]
        else:
            return [NotificationChannel.CONSOLE]
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"✅ Alert rule added: {rule.name}")
    
    async def check_alert_rules(self, market_data: Dict[str, Any]):
        """Check all alert rules against current market data"""
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check rate limiting
            if rule.rate_limit_minutes > 0 and rule.last_triggered:
                time_since = (datetime.now() - rule.last_triggered).total_seconds() / 60
                if time_since < rule.rate_limit_minutes:
                    continue
            
            # Check rule conditions
            if self._evaluate_rule_conditions(rule, market_data):
                await self._trigger_rule(rule, market_data)
    
    def _evaluate_rule_conditions(self, rule: AlertRule, market_data: Dict[str, Any]) -> bool:
        """Evaluate if rule conditions are met"""
        
        # Simple condition evaluation
        # In production, this would be more sophisticated
        
        conditions = rule.conditions
        
        # Price-based conditions
        if 'symbol' in conditions and 'price_above' in conditions:
            symbol = conditions['symbol']
            threshold = conditions['price_above']
            
            if symbol in market_data:
                current_price = market_data[symbol].get('price', 0)
                return current_price > threshold
        
        if 'symbol' in conditions and 'price_below' in conditions:
            symbol = conditions['symbol']
            threshold = conditions['price_below']
            
            if symbol in market_data:
                current_price = market_data[symbol].get('price', 0)
                return current_price < threshold
        
        # Change-based conditions
        if 'symbol' in conditions and 'change_percent_above' in conditions:
            symbol = conditions['symbol']
            threshold = conditions['change_percent_above']
            
            if symbol in market_data:
                change_pct = market_data[symbol].get('change_percent', 0)
                return change_pct > threshold
        
        return False
    
    async def _trigger_rule(self, rule: AlertRule, market_data: Dict[str, Any]):
        """Trigger alert rule"""
        
        # Create alert from rule
        symbol = rule.conditions.get('symbol', '')
        
        # Generate dynamic message based on conditions
        message = self._generate_rule_message(rule, market_data)
        
        await self.send_alert(
            alert_type=rule.alert_type,
            title=f"Alert Rule Triggered: {rule.name}",
            message=message,
            priority=rule.priority,
            symbol=symbol,
            channels=rule.channels
        )
        
        # Update rule state
        rule.last_triggered = datetime.now()
        
        logger.info(f"🔔 Alert rule triggered: {rule.name}")
    
    def _generate_rule_message(self, rule: AlertRule, market_data: Dict[str, Any]) -> str:
        """Generate message for triggered rule"""
        
        symbol = rule.conditions.get('symbol', '')
        
        if symbol in market_data:
            data = market_data[symbol]
            price = data.get('price', 0)
            change_pct = data.get('change_percent', 0)
            
            return f"{symbol} is now at {format_currency(price)} ({format_percentage(change_pct)})"
        
        return f"Rule conditions met for {rule.name}"
    
    def _update_stats(self, alert: Alert):
        """Update alert statistics"""
        
        self.stats['total_alerts'] += 1
        
        if alert.delivered:
            self.stats['successful_deliveries'] += 1
        else:
            self.stats['failed_deliveries'] += 1
        
        # Track channel usage
        for channel in alert.sent_channels:
            channel_name = channel.value
            self.stats['channels_used'][channel_name] = self.stats['channels_used'].get(channel_name, 0) + 1
        
        # Track alert types
        alert_type = alert.alert_type.value
        self.stats['alert_types'][alert_type] = self.stats['alert_types'].get(alert_type, 0) + 1
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history"""
        
        recent_alerts = self.alert_history[-limit:] if self.alert_history else []
        
        return [
            {
                'alert_id': alert.alert_id,
                'type': alert.alert_type.value,
                'priority': alert.priority.value,
                'title': alert.title,
                'message': alert.message,
                'symbol': alert.symbol,
                'timestamp': alert.timestamp.isoformat(),
                'delivered': alert.delivered,
                'channels': [c.value for c in alert.sent_channels]
            }
            for alert in recent_alerts
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        
        return {
            'total_alerts': self.stats['total_alerts'],
            'successful_deliveries': self.stats['successful_deliveries'],
            'failed_deliveries': self.stats['failed_deliveries'],
            'success_rate': (self.stats['successful_deliveries'] / max(1, self.stats['total_alerts'])) * 100,
            'channels_used': self.stats['channels_used'],
            'alert_types': self.stats['alert_types'],
            'active_channels': len(self.channels),
            'active_rules': len([r for r in self.alert_rules.values() if r.enabled])
        }
    
    def get_channel_status(self) -> Dict[str, Any]:
        """Get status of all notification channels"""
        
        status = {}
        
        for channel_type, channel in self.channels.items():
            status[channel_type.value] = {
                'enabled': channel.enabled,
                'rate_limit': channel.rate_limit,
                'sent_count': channel.sent_count,
                'last_reset': channel.last_reset.isoformat()
            }
        
        return status

# ==================== CONVENIENCE FUNCTIONS ====================

async def send_price_alert(alert_system: AdvancedAlertSystem, symbol: str, 
                          current_price: float, threshold: float, 
                          above: bool = True) -> str:
    """Send price threshold alert"""
    
    direction = "above" if above else "below"
    
    return await alert_system.send_alert(
        alert_type=AlertType.PRICE_ALERT,
        title=f"Price Alert: {symbol}",
        message=f"{symbol} is now {direction} {format_currency(threshold)} at {format_currency(current_price)}",
        priority=AlertPriority.HIGH,
        symbol=symbol,
        data={
            'current_price': current_price,
            'threshold': threshold,
            'direction': direction
        }
    )

async def send_regime_change_alert(alert_system: AdvancedAlertSystem, 
                                  old_regime: RegimeType, new_regime: RegimeType,
                                  confidence: float) -> str:
    """Send market regime change alert"""
    
    return await alert_system.send_alert(
        alert_type=AlertType.REGIME_CHANGE,
        title="Market Regime Change Detected",
        message=f"Market regime changed from {old_regime.value} to {new_regime.value} with {confidence:.1f}% confidence",
        priority=AlertPriority.HIGH,
        data={
            'old_regime': old_regime.value,
            'new_regime': new_regime.value,
            'confidence': confidence
        }
    )

async def send_trading_signal_alert(alert_system: AdvancedAlertSystem, symbol: str,
                                   signal_type: str, confidence: float,
                                   price: float) -> str:
    """Send trading signal alert"""
    
    return await alert_system.send_alert(
        alert_type=AlertType.TRADING_SIGNAL,
        title=f"Trading Signal: {symbol}",
        message=f"Generated {signal_type} signal for {symbol} at {format_currency(price)} with {confidence:.1f}% confidence",
        priority=AlertPriority.NORMAL,
        symbol=symbol,
        data={
            'signal_type': signal_type,
            'confidence': confidence,
            'price': price
        }
    )

# ==================== TESTING ====================

def test_alert_system():
    """Test alert system functionality"""
    
    print("🧪 Testing Alert System")
    print("=" * 40)
    
    # Create alert system with console only
    config = {
        'console': {'enabled': True},
        'discord': {'enabled': False},
        'email': {'enabled': False}
    }
    
    alert_system = AdvancedAlertSystem(config)
    
    async def run_tests():
        # Test basic alert
        alert_id = await alert_system.send_alert(
            alert_type=AlertType.PRICE_ALERT,
            title="Test Price Alert",
            message="AAPL has reached $150.00",
            priority=AlertPriority.HIGH,
            symbol="AAPL",
            data={'price': 150.00, 'threshold': 150.00}
        )
        
        print(f"✅ Alert sent with ID: {alert_id}")
        
        # Test different priority
        await alert_system.send_alert(
            alert_type=AlertType.SYSTEM_STATUS,
            title="System Status",
            message="All systems operational",
            priority=AlertPriority.LOW
        )
        
        # Get statistics
        stats = alert_system.get_statistics()
        print(f"✅ Statistics: {stats['total_alerts']} alerts sent")
        
        # Get history
        history = alert_system.get_alert_history(5)
        print(f"✅ History: {len(history)} recent alerts")
    
    # Run async tests
    asyncio.run(run_tests())
    
    print("\n🎉 Alert system tests completed!")

if __name__ == "__main__":
    test_alert_system()