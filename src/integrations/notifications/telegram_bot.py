#!/usr/bin/env python3
"""
File: telegram_bot.py
Path: NeuroCluster-Elite/src/integrations/notifications/telegram_bot.py
Description: Telegram notification channel for NeuroCluster Elite

This module implements Telegram notifications via Bot API, providing instant
trading alerts and portfolio updates directly to Telegram chats and channels.

Features:
- Telegram Bot API integration
- Rich message formatting with Markdown/HTML
- Inline keyboards for interactive responses
- File attachments (charts, reports)
- Group and channel broadcasting
- Private message support
- Message threading and replies
- Command handling for bot interactions

API Documentation: https://core.telegram.org/bots/api

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
import html
import re
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

# ==================== TELEGRAM CONFIGURATION ====================

@dataclass
class TelegramConfig:
    """Telegram-specific configuration"""
    bot_token: str
    
    # Chat settings
    default_chat_id: str = ""
    alert_chat_id: str = ""
    trade_chat_id: str = ""
    portfolio_chat_id: str = ""
    admin_chat_id: str = ""
    
    # Message settings
    parse_mode: str = "MarkdownV2"  # MarkdownV2, Markdown, HTML
    disable_notification: bool = False
    disable_web_page_preview: bool = True
    
    # Bot settings
    bot_username: str = ""
    commands_enabled: bool = True
    
    # File settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB Telegram limit
    photo_quality: int = 95  # JPEG quality for photos
    
    # Threading settings
    reply_to_message_id: Optional[int] = None
    message_thread_id: Optional[int] = None

class TelegramParseMode(Enum):
    """Telegram message parsing modes"""
    MARKDOWNV2 = "MarkdownV2"
    MARKDOWN = "Markdown"
    HTML = "HTML"

# ==================== TELEGRAM NOTIFICATION CHANNEL ====================

class TelegramNotificationChannel(BaseNotificationChannel):
    """
    Telegram notification channel implementation
    
    Provides Telegram messaging via Bot API with rich formatting,
    file attachments, and interactive keyboards.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Telegram notification channel"""
        super().__init__(config)
        
        # Extract Telegram-specific config
        auth_config = config.additional_auth or {}
        self.bot_token = config.api_key or auth_config.get('bot_token', '')
        
        # Chat settings
        self.default_chat_id = auth_config.get('default_chat_id', '')
        self.alert_chat_id = auth_config.get('alert_chat_id', '')
        self.trade_chat_id = auth_config.get('trade_chat_id', '')
        self.portfolio_chat_id = auth_config.get('portfolio_chat_id', '')
        self.admin_chat_id = auth_config.get('admin_chat_id', '')
        
        # Message settings
        self.parse_mode = auth_config.get('parse_mode', 'MarkdownV2')
        self.disable_notification = auth_config.get('disable_notification', False)
        self.disable_web_page_preview = auth_config.get('disable_web_page_preview', True)
        
        # Bot settings
        self.bot_username = auth_config.get('bot_username', '')
        self.commands_enabled = auth_config.get('commands_enabled', True)
        
        # File settings
        self.max_file_size = auth_config.get('max_file_size', 50 * 1024 * 1024)
        self.photo_quality = auth_config.get('photo_quality', 95)
        
        # API base URL
        self.api_base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # Channel capabilities
        self.supports_html = True
        self.supports_markdown = True
        self.supports_images = True
        self.supports_attachments = True
        self.supports_rich_formatting = True
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Bot info
        self.bot_info: Optional[Dict[str, Any]] = None
        
        # Message tracking
        self.sent_messages: List[Dict[str, Any]] = []
        
        logger.info("🤖 Telegram notification channel initialized")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def connect(self) -> bool:
        """Connect to Telegram Bot API"""
        try:
            # Validate configuration
            if not self.bot_token:
                error_msg = "Telegram bot token not provided"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test bot connection and get info
            bot_info = await self._get_bot_info()
            if bot_info:
                self.bot_info = bot_info
                self.bot_username = bot_info.get('username', '')
                
                self.update_status(IntegrationStatus.CONNECTED)
                logger.info(f"✅ Telegram bot connected: @{self.bot_username}")
                return True
            else:
                error_msg = "Failed to get bot information"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Telegram connection failed: {e}"
            self.update_status(IntegrationStatus.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Telegram Bot API"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.update_status(IntegrationStatus.DISCONNECTED)
            logger.info("✅ Telegram bot disconnected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting Telegram bot: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            bot_info = await self._get_bot_info()
            return bot_info is not None
        except Exception as e:
            logger.error(f"❌ Telegram connection test failed: {e}")
            return False
    
    async def _get_bot_info(self) -> Optional[Dict[str, Any]]:
        """Get bot information from Telegram API"""
        try:
            if not self.session:
                return None
            
            async with self.session.get(f"{self.api_base_url}/getMe") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        return data.get('result')
                    else:
                        logger.error(f"❌ Telegram API error: {data.get('description')}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Telegram API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error getting bot info: {e}")
            return None
    
    # ==================== MESSAGE FORMATTING ====================
    
    async def format_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Format message for Telegram"""
        try:
            # Determine chat ID based on category
            chat_id = self._get_chat_id_for_category(message.category)
            
            if not chat_id:
                chat_id = self.default_chat_id
            
            if not chat_id:
                return {'formatted': False, 'error': 'No chat ID configured'}
            
            # Format message based on parse mode
            if self.parse_mode == "HTML":
                formatted_text = self._format_html_message(message)
            elif self.parse_mode == "Markdown":
                formatted_text = self._format_markdown_message(message)
            else:  # MarkdownV2
                formatted_text = self._format_markdownv2_message(message)
            
            # Prepare message payload
            payload = {
                'chat_id': chat_id,
                'text': formatted_text,
                'parse_mode': self.parse_mode,
                'disable_notification': self.disable_notification,
                'disable_web_page_preview': self.disable_web_page_preview
            }
            
            # Add inline keyboard for interactive responses (if applicable)
            if message.priority in [NotificationPriority.HIGH, NotificationPriority.URGENT]:
                payload['reply_markup'] = self._create_inline_keyboard(message)
            
            return {
                'formatted': True,
                'payload': payload,
                'chat_id': chat_id,
                'attachments': message.attachments
            }
            
        except Exception as e:
            logger.error(f"❌ Error formatting Telegram message: {e}")
            return {'formatted': False, 'error': str(e)}
    
    def _format_html_message(self, message: NotificationMessage) -> str:
        """Format message using HTML"""
        
        # Priority emoji
        priority_emojis = {
            NotificationPriority.LOW: "",
            NotificationPriority.NORMAL: "",
            NotificationPriority.HIGH: "🔶 ",
            NotificationPriority.URGENT: "🔴 ",
            NotificationPriority.CRITICAL: "🚨 "
        }
        
        # Category emoji
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
        
        priority_emoji = priority_emojis.get(message.priority, "")
        category_emoji = category_emojis.get(message.category, "📢")
        
        # Build message
        text_parts = []
        
        # Title with emojis
        title = f"{priority_emoji}{category_emoji} <b>{html.escape(message.title)}</b>"
        text_parts.append(title)
        text_parts.append("")  # Empty line
        
        # Message content
        content = html.escape(message.message)
        # Convert markdown-style formatting to HTML
        content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
        content = re.sub(r'\*(.*?)\*', r'<i>\1</i>', content)
        content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)
        content = content.replace('\n', '\n')
        
        text_parts.append(content)
        
        # Add data table
        if message.data:
            text_parts.append("")
            text_parts.append("<b>📊 Details:</b>")
            for key, value in message.data.items():
                if isinstance(value, (int, float, str, bool)):
                    formatted_value = self._format_value(key, value)
                    clean_key = key.replace('_', ' ').title()
                    text_parts.append(f"• <b>{html.escape(clean_key)}:</b> <code>{html.escape(str(formatted_value))}</code>")
        
        # Footer
        text_parts.append("")
        text_parts.append(f"🕐 <i>{message.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>")
        text_parts.append(f"🆔 <code>{message.id[:8]}</code>")
        
        return "\n".join(text_parts)
    
    def _format_markdown_message(self, message: NotificationMessage) -> str:
        """Format message using Markdown (legacy)"""
        
        # Priority emoji
        priority_emojis = {
            NotificationPriority.LOW: "",
            NotificationPriority.NORMAL: "",
            NotificationPriority.HIGH: "🔶 ",
            NotificationPriority.URGENT: "🔴 ",
            NotificationPriority.CRITICAL: "🚨 "
        }
        
        # Category emoji
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
        
        priority_emoji = priority_emojis.get(message.priority, "")
        category_emoji = category_emojis.get(message.category, "📢")
        
        # Build message
        text_parts = []
        
        # Title with emojis
        title = f"{priority_emoji}{category_emoji} *{message.title}*"
        text_parts.append(title)
        text_parts.append("")  # Empty line
        
        # Message content (already in markdown format)
        text_parts.append(message.message)
        
        # Add data table
        if message.data:
            text_parts.append("")
            text_parts.append("*📊 Details:*")
            for key, value in message.data.items():
                if isinstance(value, (int, float, str, bool)):
                    formatted_value = self._format_value(key, value)
                    clean_key = key.replace('_', ' ').title()
                    text_parts.append(f"• *{clean_key}:* `{formatted_value}`")
        
        # Footer
        text_parts.append("")
        text_parts.append(f"🕐 _{message.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}_")
        text_parts.append(f"🆔 `{message.id[:8]}`")
        
        return "\n".join(text_parts)
    
    def _format_markdownv2_message(self, message: NotificationMessage) -> str:
        """Format message using MarkdownV2"""
        
        # MarkdownV2 requires escaping special characters
        def escape_markdownv2(text: str) -> str:
            """Escape special characters for MarkdownV2"""
            special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
            for char in special_chars:
                text = text.replace(char, f'\\{char}')
            return text
        
        # Priority emoji
        priority_emojis = {
            NotificationPriority.LOW: "",
            NotificationPriority.NORMAL: "",
            NotificationPriority.HIGH: "🔶 ",
            NotificationPriority.URGENT: "🔴 ",
            NotificationPriority.CRITICAL: "🚨 "
        }
        
        # Category emoji
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
        
        priority_emoji = priority_emojis.get(message.priority, "")
        category_emoji = category_emojis.get(message.category, "📢")
        
        # Build message
        text_parts = []
        
        # Title with emojis
        escaped_title = escape_markdownv2(message.title)
        title = f"{priority_emoji}{category_emoji} *{escaped_title}*"
        text_parts.append(title)
        text_parts.append("")  # Empty line
        
        # Message content - convert from simple markdown to MarkdownV2
        content = message.message
        # First escape everything
        content = escape_markdownv2(content)
        # Then restore intended formatting
        content = re.sub(r'\\\*\\\*(.*?)\\\*\\\*', r'*\1*', content)  # Bold
        content = re.sub(r'\\\*(.*?)\\\*', r'_\1_', content)  # Italic
        content = re.sub(r'\\`(.*?)\\`', r'`\1`', content)  # Code
        
        text_parts.append(content)
        
        # Add data table
        if message.data:
            text_parts.append("")
            text_parts.append("*📊 Details:*")
            for key, value in message.data.items():
                if isinstance(value, (int, float, str, bool)):
                    formatted_value = self._format_value(key, value)
                    clean_key = escape_markdownv2(key.replace('_', ' ').title())
                    escaped_value = escape_markdownv2(str(formatted_value))
                    text_parts.append(f"• *{clean_key}:* `{escaped_value}`")
        
        # Footer
        text_parts.append("")
        timestamp = escape_markdownv2(message.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'))
        message_id = escape_markdownv2(message.id[:8])
        text_parts.append(f"🕐 _{timestamp}_")
        text_parts.append(f"🆔 `{message_id}`")
        
        return "\n".join(text_parts)
    
    def _format_value(self, key: str, value: Any) -> str:
        """Format value based on key name"""
        
        if isinstance(value, float):
            if 'price' in key.lower() or 'value' in key.lower() or 'amount' in key.lower():
                return self.format_currency(value)
            elif 'percent' in key.lower() or 'pct' in key.lower():
                return self.format_percentage(value)
            else:
                return f"{value:.4f}".rstrip('0').rstrip('.')
        else:
            return str(value)
    
    def _get_chat_id_for_category(self, category: NotificationCategory) -> str:
        """Get appropriate chat ID based on message category"""
        
        channel_routing = {
            NotificationCategory.TRADE_EXECUTION: self.trade_chat_id,
            NotificationCategory.ORDER_STATUS: self.trade_chat_id,
            NotificationCategory.PORTFOLIO_UPDATE: self.portfolio_chat_id,
            NotificationCategory.MARKET_ALERT: self.alert_chat_id,
            NotificationCategory.RISK_WARNING: self.alert_chat_id,
            NotificationCategory.PRICE_ALERT: self.alert_chat_id,
            NotificationCategory.SYSTEM_STATUS: self.admin_chat_id
        }
        
        return channel_routing.get(category, self.default_chat_id)
    
    def _create_inline_keyboard(self, message: NotificationMessage) -> Dict[str, Any]:
        """Create inline keyboard for interactive responses"""
        
        buttons = []
        
        if message.category == NotificationCategory.STRATEGY_SIGNAL:
            # Add buttons for strategy signals
            buttons.append([
                {"text": "📈 View Chart", "callback_data": f"chart_{message.id}"},
                {"text": "📊 Analysis", "callback_data": f"analysis_{message.id}"}
            ])
            buttons.append([
                {"text": "✅ Executed", "callback_data": f"executed_{message.id}"},
                {"text": "❌ Ignored", "callback_data": f"ignored_{message.id}"}
            ])
        
        elif message.category == NotificationCategory.PORTFOLIO_UPDATE:
            # Add buttons for portfolio updates
            buttons.append([
                {"text": "📊 Full Report", "callback_data": f"report_{message.id}"},
                {"text": "📈 Performance", "callback_data": f"performance_{message.id}"}
            ])
        
        elif message.category == NotificationCategory.RISK_WARNING:
            # Add buttons for risk warnings
            buttons.append([
                {"text": "🔍 Details", "callback_data": f"risk_details_{message.id}"},
                {"text": "✅ Acknowledged", "callback_data": f"risk_ack_{message.id}"}
            ])
        
        if buttons:
            return {"inline_keyboard": buttons}
        
        return {}
    
    # ==================== MESSAGE SENDING ====================
    
    async def send_message(self, message: NotificationMessage) -> bool:
        """Send Telegram message"""
        try:
            # Ensure connection
            if not self.session:
                if not await self.connect():
                    return False
            
            # Format message
            formatted = await self.format_message(message)
            if not formatted.get('formatted'):
                logger.error(f"❌ Failed to format Telegram message: {formatted.get('error')}")
                return False
            
            payload = formatted['payload']
            chat_id = formatted['chat_id']
            attachments = formatted.get('attachments', [])
            
            # Send attachments first if present
            if attachments:
                await self._send_attachments(chat_id, attachments)
            
            # Send main message
            async with self.session.post(f"{self.api_base_url}/sendMessage", json=payload) as response:
                if response.status == 200:
                    response_data = await response.json()
                    
                    if response_data.get('ok'):
                        message_result = response_data.get('result', {})
                        message_id = message_result.get('message_id')
                        
                        # Track sent message
                        self.sent_messages.append({
                            'message_id': message.id,
                            'telegram_message_id': message_id,
                            'chat_id': chat_id,
                            'title': message.title,
                            'category': message.category.value,
                            'priority': message.priority.value,
                            'sent_at': datetime.now()
                        })
                        
                        # Trim message history
                        if len(self.sent_messages) > 100:
                            self.sent_messages = self.sent_messages[-100:]
                        
                        logger.info(f"✅ Telegram message sent successfully: {message.title}")
                        return True
                    else:
                        error_description = response_data.get('description', 'Unknown error')
                        logger.error(f"❌ Telegram API error: {error_description}")
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Telegram API error {response.status}: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
            message.error_message = str(e)
            return False
    
    async def _send_attachments(self, chat_id: str, attachment_paths: List[str]):
        """Send file attachments to Telegram"""
        
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
                
                # Determine file type and send accordingly
                file_extension = file_path.suffix.lower()
                
                if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                    # Send as photo
                    await self._send_photo(chat_id, file_path)
                elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Send as video (if under size limit)
                    await self._send_video(chat_id, file_path)
                else:
                    # Send as document
                    await self._send_document(chat_id, file_path)
                    
                logger.info(f"📎 Sent attachment: {file_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Error sending attachment {attachment_path}: {e}")
    
    async def _send_photo(self, chat_id: str, file_path: Path):
        """Send photo to Telegram"""
        
        data = aiohttp.FormData()
        data.add_field('chat_id', chat_id)
        
        with open(file_path, 'rb') as f:
            data.add_field('photo', f, filename=file_path.name)
            
            async with self.session.post(f"{self.api_base_url}/sendPhoto", data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Error sending photo: {error_text}")
    
    async def _send_video(self, chat_id: str, file_path: Path):
        """Send video to Telegram"""
        
        data = aiohttp.FormData()
        data.add_field('chat_id', chat_id)
        
        with open(file_path, 'rb') as f:
            data.add_field('video', f, filename=file_path.name)
            
            async with self.session.post(f"{self.api_base_url}/sendVideo", data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Error sending video: {error_text}")
    
    async def _send_document(self, chat_id: str, file_path: Path):
        """Send document to Telegram"""
        
        data = aiohttp.FormData()
        data.add_field('chat_id', chat_id)
        
        with open(file_path, 'rb') as f:
            data.add_field('document', f, filename=file_path.name)
            
            async with self.session.post(f"{self.api_base_url}/sendDocument", data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Error sending document: {error_text}")
    
    # ==================== TELEGRAM TEMPLATES ====================
    
    def load_default_templates(self):
        """Load Telegram-specific templates"""
        super().load_default_templates()
        
        # Override templates with Telegram-specific formatting
        
        # Trade execution template for Telegram
        self.templates['trade_execution_telegram'] = NotificationTemplate(
            name="Trade Execution (Telegram)",
            title_template="Trade Executed: {side} {quantity} {symbol}",
            message_template="""**🎯 Trade Execution Confirmed**

Your **{side}** order has been executed successfully!

**Symbol:** `{symbol}`
**Quantity:** `{quantity}`
**Price:** `${price}`
**Total Value:** `${total_value}`
**Order ID:** `{order_id}`

🕐 **Executed at:** {timestamp}

💡 _The trade has been reflected in your portfolio._""",
            category=NotificationCategory.TRADE_EXECUTION,
            supports_markdown=True,
            required_variables=['symbol', 'side', 'quantity', 'price', 'total_value', 'timestamp', 'order_id']
        )
        
        # Strategy signal template for Telegram
        self.templates['strategy_signal_telegram'] = NotificationTemplate(
            name="Strategy Signal (Telegram)",
            title_template="Strategy Signal: {strategy_name}",
            message_template="""**🎯 New Strategy Signal**

📈 **Strategy:** `{strategy_name}`
🎯 **Signal:** **{signal_type}**
📊 **Symbol:** `{symbol}`
🔬 **Confidence:** `{confidence}%`

**💡 Analysis:** {reasoning}

📋 _Use the buttons below to track your action on this signal._""",
            category=NotificationCategory.STRATEGY_SIGNAL,
            supports_markdown=True,
            required_variables=['strategy_name', 'signal_type', 'symbol', 'confidence', 'reasoning']
        )
        
        logger.info(f"🤖 Loaded {len(self.templates)} Telegram templates")
    
    # ==================== UTILITY METHODS ====================
    
    def get_telegram_stats(self) -> Dict[str, Any]:
        """Get Telegram message statistics"""
        
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
        
        # Group by chat
        by_chat = {}
        for msg in self.sent_messages:
            chat_id = msg['chat_id']
            by_chat[chat_id] = by_chat.get(chat_id, 0) + 1
        
        return {
            'total_sent': total_sent,
            'sent_last_7_days': len(recent_messages),
            'by_category': by_category,
            'by_priority': by_priority,
            'by_chat': by_chat,
            'bot_info': self.bot_info,
            'parse_mode': self.parse_mode,
            'commands_enabled': self.commands_enabled,
            'recent_activity': recent_messages[-10:] if recent_messages else []
        }

# ==================== TESTING ====================

async def test_telegram_channel():
    """Test Telegram notification channel"""
    print("🧪 Testing Telegram Notification Channel")
    print("=" * 40)
    
    # Create test configuration
    config = IntegrationConfig(
        name="telegram",
        integration_type="notification",
        enabled=True,
        api_key="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
        additional_auth={
            'default_chat_id': '123456789',
            'parse_mode': 'MarkdownV2',
            'commands_enabled': True
        }
    )
    
    # Create Telegram channel
    telegram_channel = TelegramNotificationChannel(config)
    
    print("✅ Telegram channel instance created")
    print(f"✅ Bot token configured: {bool(telegram_channel.bot_token)}")
    print(f"✅ Parse mode: {telegram_channel.parse_mode}")
    print(f"✅ Commands enabled: {telegram_channel.commands_enabled}")
    print(f"✅ Supports markdown: {telegram_channel.supports_markdown}")
    print(f"✅ Supports attachments: {telegram_channel.supports_attachments}")
    
    # Test message formatting
    from src.integrations.notifications import NotificationMessage, NotificationPriority, NotificationCategory
    
    test_message = NotificationMessage(
        id="test-telegram-123",
        title="Test Trading Alert",
        message="**Test Alert**\n\nThis is a test Telegram message with **bold** text and bullet points:\n• Point 1\n• Point 2\n• Point 3",
        category=NotificationCategory.STRATEGY_SIGNAL,
        priority=NotificationPriority.HIGH,
        channels=[],
        data={
            'symbol': 'ETH/USD',
            'price': 2500.75,
            'quantity': 1.5,
            'total_value': 3751.13,
            'confidence': 92.3
        }
    )
    
    formatted = await telegram_channel.format_message(test_message)
    if formatted.get('formatted'):
        print("✅ Message formatting successful")
        payload = formatted['payload']
        print(f"✅ Chat ID: {payload['chat_id']}")
        print(f"✅ Parse mode: {payload['parse_mode']}")
        print(f"✅ Has inline keyboard: {'reply_markup' in payload}")
        print(f"✅ Message length: {len(payload['text'])} characters")
    else:
        print(f"❌ Message formatting failed: {formatted.get('error')}")
    
    # Test templates
    templates = telegram_channel.templates
    print(f"✅ Templates loaded: {len(templates)}")
    telegram_templates = [name for name in templates if 'telegram' in name]
    for template_name in telegram_templates:
        print(f"  • {template_name}")
    
    print("\n⚠️  Note: Message sending requires valid Telegram bot token and chat ID")
    print("   Create a Telegram bot via @BotFather to get credentials")
    
    print("\n🎉 Telegram notification channel tests completed!")

if __name__ == "__main__":
    asyncio.run(test_telegram_channel())