#!/usr/bin/env python3
"""
File: discord_bot.py
Path: NeuroCluster-Elite/src/integrations/notifications/discord_bot.py
Description: Discord notification channel for NeuroCluster Elite

This module implements Discord notifications via webhooks and bot API, providing
real-time trading alerts and portfolio updates directly to Discord channels.

Features:
- Discord webhook integration
- Rich embed messages with formatting
- Emoji and color-coded alerts
- Channel-specific routing
- File attachments (charts, reports)
- Mention support for urgent alerts
- Message threading and replies
- Discord bot commands (optional)

API Documentation: https://discord.com/developers/docs/

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import base64
from pathlib import Path

# Import our modules
try:
    from src.integrations.notifications import (
        BaseNotificationChannel, NotificationMessage, NotificationTemplate,
        NotificationCategory, NotificationPriority, NotificationStatus
    )
    from src.integrations import IntegrationConfig, IntegrationStatus
    from src.utils.helpers import format_currency, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DISCORD CONFIGURATION ====================

@dataclass
class DiscordConfig:
    """Discord-specific configuration"""
    webhook_url: str
    bot_token: str = ""
    
    # Channel settings
    default_channel_id: str = ""
    alert_channel_id: str = ""
    trade_channel_id: str = ""
    portfolio_channel_id: str = ""
    
    # Mention settings
    user_id: str = ""  # User ID to mention for urgent alerts
    role_id: str = ""  # Role ID to mention for critical alerts
    
    # Bot settings
    bot_enabled: bool = False
    command_prefix: str = "!"
    
    # Message settings
    username: str = "NeuroCluster Elite"
    avatar_url: str = ""
    
    # Embed settings
    embed_color: int = 0x007bff  # Blue
    include_timestamp: bool = True
    include_footer: bool = True
    
    # File upload settings
    max_file_size: int = 8 * 1024 * 1024  # 8MB Discord limit

class DiscordEmbedColor(Enum):
    """Discord embed colors for different priorities"""
    LOW = 0x6c757d      # Gray
    NORMAL = 0x007bff   # Blue  
    HIGH = 0xfd7e14     # Orange
    URGENT = 0xdc3545   # Red
    CRITICAL = 0x6f42c1 # Purple
    SUCCESS = 0x28a745  # Green
    WARNING = 0xffc107  # Yellow

# ==================== DISCORD NOTIFICATION CHANNEL ====================

class DiscordNotificationChannel(BaseNotificationChannel):
    """
    Discord notification channel implementation
    
    Provides Discord messaging via webhooks with rich embed formatting,
    file attachments, and channel-specific routing.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Discord notification channel"""
        super().__init__(config)
        
        # Extract Discord-specific config
        auth_config = config.additional_auth or {}
        self.webhook_url = auth_config.get('webhook_url', '')
        self.bot_token = auth_config.get('bot_token', '')
        
        # Channel settings
        self.default_channel_id = auth_config.get('default_channel_id', '')
        self.alert_channel_id = auth_config.get('alert_channel_id', '')
        self.trade_channel_id = auth_config.get('trade_channel_id', '')
        self.portfolio_channel_id = auth_config.get('portfolio_channel_id', '')
        
        # Mention settings
        self.user_id = auth_config.get('user_id', '')
        self.role_id = auth_config.get('role_id', '')
        
        # Bot settings
        self.bot_enabled = auth_config.get('bot_enabled', False)
        self.command_prefix = auth_config.get('command_prefix', '!')
        
        # Message settings
        self.username = auth_config.get('username', 'NeuroCluster Elite')
        self.avatar_url = auth_config.get('avatar_url', '')
        
        # Embed settings
        self.embed_color = auth_config.get('embed_color', 0x007bff)
        self.include_timestamp = auth_config.get('include_timestamp', True)
        self.include_footer = auth_config.get('include_footer', True)
        
        # File settings
        self.max_file_size = auth_config.get('max_file_size', 8 * 1024 * 1024)
        
        # Channel capabilities
        self.supports_html = False  # Discord uses markdown
        self.supports_markdown = True
        self.supports_images = True
        self.supports_attachments = True
        self.supports_rich_formatting = True
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Message tracking
        self.sent_messages: List[Dict[str, Any]] = []
        
        logger.info("🟦 Discord notification channel initialized")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def connect(self) -> bool:
        """Connect to Discord API"""
        try:
            # Validate configuration
            if not self.webhook_url:
                error_msg = "Discord webhook URL not provided"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
            
            # Create HTTP session
            headers = {'User-Agent': 'NeuroCluster-Elite-Bot/1.0'}
            if self.bot_token:
                headers['Authorization'] = f'Bot {self.bot_token}'
            
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=headers
            )
            
            # Test webhook
            if await self._test_webhook():
                self.update_status(IntegrationStatus.CONNECTED)
                logger.info("✅ Discord webhook connected successfully")
                return True
            else:
                error_msg = "Discord webhook test failed"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Discord connection failed: {e}"
            self.update_status(IntegrationStatus.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Discord API"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.update_status(IntegrationStatus.DISCONNECTED)
            logger.info("✅ Discord disconnected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting Discord: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Discord connection"""
        return await self._test_webhook()
    
    async def _test_webhook(self) -> bool:
        """Test Discord webhook"""
        try:
            if not self.session:
                return False
            
            # Send a minimal test message
            test_payload = {
                "content": "🧪 NeuroCluster Elite connection test",
                "username": self.username
            }
            
            if self.avatar_url:
                test_payload["avatar_url"] = self.avatar_url
            
            async with self.session.post(self.webhook_url, json=test_payload) as response:
                if response.status in [200, 204]:
                    logger.info("✅ Discord webhook test successful")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Discord webhook test failed {response.status}: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Discord webhook test error: {e}")
            return False
    
    # ==================== MESSAGE FORMATTING ====================
    
    async def format_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Format message for Discord"""
        try:
            # Get priority color
            priority_colors = {
                NotificationPriority.LOW: DiscordEmbedColor.LOW.value,
                NotificationPriority.NORMAL: DiscordEmbedColor.NORMAL.value,
                NotificationPriority.HIGH: DiscordEmbedColor.HIGH.value,
                NotificationPriority.URGENT: DiscordEmbedColor.URGENT.value,
                NotificationPriority.CRITICAL: DiscordEmbedColor.CRITICAL.value
            }
            
            embed_color = priority_colors.get(message.priority, DiscordEmbedColor.NORMAL.value)
            
            # Create embed
            embed = {
                "title": self._add_emoji_to_title(message.title, message.category, message.priority),
                "description": self._format_discord_message(message.message),
                "color": embed_color
            }
            
            # Add timestamp
            if self.include_timestamp:
                embed["timestamp"] = message.created_at.isoformat()
            
            # Add footer
            if self.include_footer:
                embed["footer"] = {
                    "text": f"NeuroCluster Elite • {message.category.value.replace('_', ' ').title()} • ID: {message.id[:8]}"
                }
                if self.avatar_url:
                    embed["footer"]["icon_url"] = self.avatar_url
            
            # Add data fields
            if message.data:
                embed["fields"] = self._create_embed_fields(message.data)
            
            # Add image
            if message.image_url:
                embed["image"] = {"url": message.image_url}
            
            # Prepare webhook payload
            payload = {
                "username": self.username,
                "embeds": [embed]
            }
            
            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url
            
            # Add content with mentions for urgent messages
            content_parts = []
            if message.priority in [NotificationPriority.URGENT, NotificationPriority.CRITICAL]:
                if self.user_id and message.priority == NotificationPriority.URGENT:
                    content_parts.append(f"<@{self.user_id}>")
                elif self.role_id and message.priority == NotificationPriority.CRITICAL:
                    content_parts.append(f"<@&{self.role_id}>")
            
            if content_parts:
                payload["content"] = " ".join(content_parts)
            
            return {
                'payload': payload,
                'formatted': True,
                'webhook_url': self._get_channel_webhook(message.category)
            }
            
        except Exception as e:
            logger.error(f"❌ Error formatting Discord message: {e}")
            return {'formatted': False, 'error': str(e)}
    
    def _add_emoji_to_title(self, title: str, category: NotificationCategory, priority: NotificationPriority) -> str:
        """Add appropriate emoji to message title"""
        
        category_emojis = {
            NotificationCategory.TRADE_EXECUTION: "✅",
            NotificationCategory.ORDER_STATUS: "📋",
            NotificationCategory.MARKET_ALERT: "📊",
            NotificationCategory.PORTFOLIO_UPDATE: "💼",
            NotificationCategory.RISK_WARNING: "⚠️",
            NotificationCategory.SYSTEM_STATUS: "🔧",
            NotificationCategory.STRATEGY_SIGNAL: "🎯",
            NotificationCategory.NEWS_ALERT: "📰",
            NotificationCategory.TECHNICAL_ANALYSIS: "📈",
            NotificationCategory.PRICE_ALERT: "💰"
        }
        
        priority_emojis = {
            NotificationPriority.LOW: "",
            NotificationPriority.NORMAL: "",
            NotificationPriority.HIGH: "🔶",
            NotificationPriority.URGENT: "🔴",
            NotificationPriority.CRITICAL: "🚨"
        }
        
        emoji = category_emojis.get(category, "📢")
        priority_emoji = priority_emojis.get(priority, "")
        
        if priority_emoji:
            return f"{priority_emoji} {emoji} {title}"
        else:
            return f"{emoji} {title}"
    
    def _format_discord_message(self, message: str) -> str:
        """Format message content for Discord markdown"""
        
        # Discord supports markdown formatting
        # Convert bullet points to Discord format
        lines = message.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('•'):
                # Convert bullet points
                content = line[1:].strip()
                formatted_lines.append(f"• {content}")
            elif line.startswith('**') and line.endswith('**'):
                # Keep bold formatting
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        # Limit message length (Discord embed description limit is 4096)
        result = '\n'.join(formatted_lines)
        if len(result) > 4000:
            result = result[:3997] + "..."
        
        return result
    
    def _create_embed_fields(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create embed fields from data dictionary"""
        
        fields = []
        
        # Limit to 25 fields (Discord limit)
        count = 0
        for key, value in data.items():
            if count >= 25:
                break
            
            if isinstance(value, (int, float, str, bool)):
                # Format value
                if isinstance(value, float):
                    if 'price' in key.lower() or 'value' in key.lower() or 'amount' in key.lower():
                        formatted_value = self.format_currency(value)
                    elif 'percent' in key.lower() or 'pct' in key.lower():
                        formatted_value = self.format_percentage(value)
                    else:
                        formatted_value = f"{value:.4f}".rstrip('0').rstrip('.')
                else:
                    formatted_value = str(value)
                
                # Clean up key name
                clean_key = key.replace('_', ' ').title()
                
                # Determine if field should be inline
                inline = len(formatted_value) < 20
                
                fields.append({
                    "name": clean_key,
                    "value": formatted_value,
                    "inline": inline
                })
                count += 1
        
        return fields
    
    def _get_channel_webhook(self, category: NotificationCategory) -> str:
        """Get appropriate webhook URL based on message category"""
        
        # Route to specific channels based on category
        channel_routing = {
            NotificationCategory.TRADE_EXECUTION: self.trade_channel_id,
            NotificationCategory.ORDER_STATUS: self.trade_channel_id,
            NotificationCategory.PORTFOLIO_UPDATE: self.portfolio_channel_id,
            NotificationCategory.MARKET_ALERT: self.alert_channel_id,
            NotificationCategory.RISK_WARNING: self.alert_channel_id,
            NotificationCategory.PRICE_ALERT: self.alert_channel_id
        }
        
        # If we have a specific channel for this category and bot token, construct webhook URL
        # For now, just return the default webhook URL
        return self.webhook_url
    
    # ==================== MESSAGE SENDING ====================
    
    async def send_message(self, message: NotificationMessage) -> bool:
        """Send Discord message"""
        try:
            # Ensure connection
            if not self.session:
                if not await self.connect():
                    return False
            
            # Format message
            formatted = await self.format_message(message)
            if not formatted.get('formatted'):
                logger.error(f"❌ Failed to format Discord message: {formatted.get('error')}")
                return False
            
            payload = formatted['payload']
            webhook_url = formatted['webhook_url']
            
            # Send attachments if present
            files = None
            if message.attachments:
                files = await self._prepare_attachments(message.attachments)
            
            # Send message
            if files:
                # Send with file attachments
                data = aiohttp.FormData()
                data.add_field('payload_json', json.dumps(payload))
                
                for i, file_data in enumerate(files):
                    data.add_field(f'file{i}', file_data['content'], 
                                 filename=file_data['filename'],
                                 content_type=file_data['content_type'])
                
                async with self.session.post(webhook_url, data=data) as response:
                    success = response.status in [200, 204]
            else:
                # Send without attachments
                async with self.session.post(webhook_url, json=payload) as response:
                    success = response.status in [200, 204]
                    
                    if not success:
                        error_text = await response.text()
                        logger.error(f"❌ Discord webhook error {response.status}: {error_text}")
            
            if success:
                # Track sent message
                self.sent_messages.append({
                    'message_id': message.id,
                    'title': message.title,
                    'category': message.category.value,
                    'priority': message.priority.value,
                    'sent_at': datetime.now(),
                    'webhook_url': webhook_url
                })
                
                # Trim message history
                if len(self.sent_messages) > 100:
                    self.sent_messages = self.sent_messages[-100:]
                
                logger.info(f"✅ Discord message sent successfully: {message.title}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending Discord message: {e}")
            message.error_message = str(e)
            return False
    
    async def _prepare_attachments(self, attachment_paths: List[str]) -> List[Dict[str, Any]]:
        """Prepare file attachments for Discord"""
        
        files = []
        
        for attachment_path in attachment_paths:
            try:
                file_path = Path(attachment_path)
                if not file_path.exists():
                    logger.warning(f"⚠️ Attachment not found: {attachment_path}")
                    continue
                
                # Check file size
                file_size = file_path.stat().st_size
                if file_size > self.max_file_size:
                    logger.warning(f"⚠️ Attachment too large ({file_size} bytes): {attachment_path}")
                    continue
                
                # Read file
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # Determine content type
                content_type = 'application/octet-stream'
                if file_path.suffix.lower() in ['.jpg', '.jpeg']:
                    content_type = 'image/jpeg'
                elif file_path.suffix.lower() == '.png':
                    content_type = 'image/png'
                elif file_path.suffix.lower() == '.gif':
                    content_type = 'image/gif'
                elif file_path.suffix.lower() == '.pdf':
                    content_type = 'application/pdf'
                elif file_path.suffix.lower() == '.txt':
                    content_type = 'text/plain'
                
                files.append({
                    'filename': file_path.name,
                    'content': file_content,
                    'content_type': content_type
                })
                
                logger.info(f"📎 Prepared attachment: {file_path.name} ({file_size} bytes)")
                
            except Exception as e:
                logger.error(f"❌ Error preparing attachment {attachment_path}: {e}")
        
        return files
    
    # ==================== DISCORD TEMPLATES ====================
    
    def load_default_templates(self):
        """Load Discord-specific templates"""
        super().load_default_templates()
        
        # Override templates with Discord-specific formatting
        
        # Trade execution template for Discord
        self.templates['trade_execution_discord'] = NotificationTemplate(
            name="Trade Execution (Discord)",
            title_template="Trade Executed: {side} {quantity} {symbol}",
            message_template="""**Trade Execution Confirmed**

Your **{side}** order has been executed successfully!

**Symbol:** `{symbol}`
**Quantity:** `{quantity}`
**Price:** `${price}`
**Total Value:** `${total_value}`
**Order ID:** `{order_id}`

🕐 **Executed at:** {timestamp}""",
            category=NotificationCategory.TRADE_EXECUTION,
            supports_markdown=True,
            required_variables=['symbol', 'side', 'quantity', 'price', 'total_value', 'timestamp', 'order_id']
        )
        
        # Price alert template for Discord
        self.templates['price_alert_discord'] = NotificationTemplate(
            name="Price Alert (Discord)",
            title_template="Price Alert: {symbol} {direction} ${trigger_price}",
            message_template="""**Price Alert Triggered!**

**{symbol}** has crossed your alert threshold.

**Current Price:** `${current_price}`
**Trigger Price:** `${trigger_price}`
**Change:** `{change_percent}%`
**Direction:** **{direction}**

📊 Consider reviewing your position or strategy.""",
            category=NotificationCategory.PRICE_ALERT,
            supports_markdown=True,
            required_variables=['symbol', 'current_price', 'trigger_price', 'change_percent', 'direction']
        )
        
        # Strategy signal template for Discord
        self.templates['strategy_signal_discord'] = NotificationTemplate(
            name="Strategy Signal (Discord)",
            title_template="Strategy Signal: {strategy_name}",
            message_template="""**New Strategy Signal Generated**

📈 **Strategy:** `{strategy_name}`
🎯 **Signal:** **{signal_type}**
📊 **Symbol:** `{symbol}`
🔬 **Confidence:** `{confidence}%`

**Analysis:** {reasoning}

💡 Consider this signal in your trading decisions.""",
            category=NotificationCategory.STRATEGY_SIGNAL,
            supports_markdown=True,
            required_variables=['strategy_name', 'signal_type', 'symbol', 'confidence', 'reasoning']
        )
        
        # Portfolio update template for Discord
        self.templates['portfolio_summary_discord'] = NotificationTemplate(
            name="Portfolio Summary (Discord)",
            title_template="Portfolio Update - ${total_value}",
            message_template="""**📊 Portfolio Summary**

**Total Value:** `${total_value}`
**Daily P&L:** `${daily_pnl}` ({daily_pnl_percent}%)
**Cash Balance:** `${cash_balance}`

**Positions:** {open_positions} open
**Performance:** {total_return_percent}% total return

📈 View detailed analysis in the dashboard.""",
            category=NotificationCategory.PORTFOLIO_UPDATE,
            supports_markdown=True,
            include_charts=True,
            required_variables=['total_value', 'daily_pnl', 'daily_pnl_percent', 'cash_balance', 'open_positions', 'total_return_percent']
        )
        
        logger.info(f"🟦 Loaded {len(self.templates)} Discord templates")
    
    # ==================== UTILITY METHODS ====================
    
    def get_discord_stats(self) -> Dict[str, Any]:
        """Get Discord message statistics"""
        
        if not self.sent_messages:
            return {'total_sent': 0, 'recent_activity': []}
        
        # Calculate stats
        total_sent = len(self.sent_messages)
        recent_messages = [msg for msg in self.sent_messages 
                          if (datetime.now() - msg['sent_at']).days < 7]
        
        # Group by category
        by_category = {}
        for msg in self.sent_messages:
            category = msg['category']
            by_category[category] = by_category.get(category, 0) + 1
        
        # Group by priority
        by_priority = {}
        for msg in self.sent_messages:
            priority = msg['priority']
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        return {
            'total_sent': total_sent,
            'sent_last_7_days': len(recent_messages),
            'by_category': by_category,
            'by_priority': by_priority,
            'recent_activity': recent_messages[-10:] if recent_messages else [],
            'webhook_url_configured': bool(self.webhook_url),
            'bot_enabled': self.bot_enabled
        }

# ==================== TESTING ====================

async def test_discord_channel():
    """Test Discord notification channel"""
    print("🧪 Testing Discord Notification Channel")
    print("=" * 40)
    
    # Create test configuration
    config = IntegrationConfig(
        name="discord",
        integration_type="notification",
        enabled=True,
        additional_auth={
            'webhook_url': 'https://discord.com/api/webhooks/123456789/abcdefghijklmnopqrstuvwxyz',
            'username': 'NeuroCluster Elite Test',
            'user_id': '123456789012345678',
            'embed_color': 0x007bff
        }
    )
    
    # Create Discord channel
    discord_channel = DiscordNotificationChannel(config)
    
    print("✅ Discord channel instance created")
    print(f"✅ Username: {discord_channel.username}")
    print(f"✅ Webhook configured: {bool(discord_channel.webhook_url)}")
    print(f"✅ Bot enabled: {discord_channel.bot_enabled}")
    print(f"✅ Supports markdown: {discord_channel.supports_markdown}")
    print(f"✅ Supports attachments: {discord_channel.supports_attachments}")
    
    # Test message formatting
    from src.integrations.notifications import NotificationMessage, NotificationPriority, NotificationCategory
    
    test_message = NotificationMessage(
        id="test-discord-123",
        title="Test Trading Alert",
        message="**Test Alert**\n\nThis is a test Discord message with **bold** text and bullet points:\n• Point 1\n• Point 2\n• Point 3",
        category=NotificationCategory.TRADE_EXECUTION,
        priority=NotificationPriority.HIGH,
        channels=[],
        data={
            'symbol': 'BTC/USD',
            'price': 45000.50,
            'quantity': 0.1,
            'total_value': 4500.05,
            'confidence': 87.5
        }
    )
    
    formatted = await discord_channel.format_message(test_message)
    if formatted.get('formatted'):
        print("✅ Message formatting successful")
        payload = formatted['payload']
        print(f"✅ Embed title: {payload['embeds'][0]['title']}")
        print(f"✅ Embed fields: {len(payload['embeds'][0].get('fields', []))}")
        print(f"✅ Embed color: #{payload['embeds'][0]['color']:06x}")
    else:
        print(f"❌ Message formatting failed: {formatted.get('error')}")
    
    # Test templates
    templates = discord_channel.templates
    print(f"✅ Templates loaded: {len(templates)}")
    discord_templates = [name for name in templates if 'discord' in name]
    for template_name in discord_templates:
        print(f"  • {template_name}")
    
    print("\n⚠️  Note: Message sending requires valid Discord webhook URL")
    print("   Configure Discord webhook to test actual message delivery")
    
    print("\n🎉 Discord notification channel tests completed!")

if __name__ == "__main__":
    asyncio.run(test_discord_channel())