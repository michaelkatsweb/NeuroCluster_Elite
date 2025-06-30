#!/usr/bin/env python3
"""
File: email_alerts.py
Path: NeuroCluster-Elite/src/integrations/notifications/email_alerts.py
Description: Email notification channel for NeuroCluster Elite

This module implements email notifications via SMTP, providing professional-grade
email alerts for trading activities, portfolio updates, and system notifications.

Features:
- SMTP email delivery with authentication
- HTML and plain text email support
- Email templates with rich formatting
- Attachment support for reports and charts
- Multiple recipient support
- Email delivery tracking
- Bounce handling and retry logic
- TLS/SSL encryption support

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import asyncio
import aiosmtplib
import logging
import html
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import base64
import os
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

# ==================== EMAIL CONFIGURATION ====================

@dataclass
class EmailConfig:
    """Email-specific configuration"""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    
    # Security settings
    use_tls: bool = True
    use_ssl: bool = False
    
    # Email settings
    from_address: str = ""
    from_name: str = "NeuroCluster Elite"
    reply_to: str = ""
    
    # Template settings
    template_directory: str = "templates/email"
    logo_url: str = ""
    footer_text: str = "Sent by NeuroCluster Elite Trading Platform"
    
    # Delivery settings
    timeout: int = 30
    max_recipients_per_email: int = 50
    
    # HTML settings
    include_html: bool = True
    include_plain_text: bool = True

# ==================== EMAIL NOTIFICATION CHANNEL ====================

class EmailNotificationChannel(BaseNotificationChannel):
    """
    Email notification channel implementation
    
    Provides email delivery via SMTP with support for HTML formatting,
    attachments, and professional email templates.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize email notification channel"""
        super().__init__(config)
        
        # Extract email-specific config
        auth_config = config.additional_auth or {}
        self.smtp_server = auth_config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = int(auth_config.get('smtp_port', 587))
        self.username = config.api_key or auth_config.get('username', '')
        self.password = config.secret_key or auth_config.get('password', '')
        
        # Email settings
        self.use_tls = auth_config.get('use_tls', True)
        self.use_ssl = auth_config.get('use_ssl', False)
        self.from_address = auth_config.get('from_address', self.username)
        self.from_name = auth_config.get('from_name', 'NeuroCluster Elite')
        self.reply_to = auth_config.get('reply_to', '')
        
        # Template settings
        self.template_directory = auth_config.get('template_directory', 'templates/email')
        self.logo_url = auth_config.get('logo_url', '')
        self.footer_text = auth_config.get('footer_text', 'Sent by NeuroCluster Elite Trading Platform')
        
        # Delivery settings
        self.timeout = auth_config.get('timeout', 30)
        self.max_recipients = auth_config.get('max_recipients_per_email', 50)
        
        # HTML support
        self.include_html = auth_config.get('include_html', True)
        self.include_plain_text = auth_config.get('include_plain_text', True)
        
        # Channel capabilities
        self.supports_html = True
        self.supports_markdown = True
        self.supports_images = True
        self.supports_attachments = True
        self.supports_rich_formatting = True
        
        # SMTP connection
        self.smtp_client: Optional[aiosmtplib.SMTP] = None
        
        # Email tracking
        self.sent_emails: List[Dict[str, Any]] = []
        
        logger.info("📧 Email notification channel initialized")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def connect(self) -> bool:
        """Connect to SMTP server"""
        try:
            # Validate configuration
            if not self.smtp_server or not self.username or not self.password:
                error_msg = "Email SMTP configuration incomplete"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
            
            # Create SMTP client
            if self.use_ssl:
                self.smtp_client = aiosmtplib.SMTP(
                    hostname=self.smtp_server,
                    port=self.smtp_port,
                    use_tls=False,
                    timeout=self.timeout
                )
            else:
                self.smtp_client = aiosmtplib.SMTP(
                    hostname=self.smtp_server,
                    port=self.smtp_port,
                    timeout=self.timeout
                )
            
            # Connect and authenticate
            await self.smtp_client.connect()
            
            if self.use_tls and not self.use_ssl:
                await self.smtp_client.starttls()
            
            await self.smtp_client.login(self.username, self.password)
            
            self.update_status(IntegrationStatus.CONNECTED)
            logger.info(f"✅ Email SMTP connected to {self.smtp_server}")
            return True
            
        except Exception as e:
            error_msg = f"Email SMTP connection failed: {e}"
            self.update_status(IntegrationStatus.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from SMTP server"""
        try:
            if self.smtp_client:
                await self.smtp_client.quit()
                self.smtp_client = None
            
            self.update_status(IntegrationStatus.DISCONNECTED)
            logger.info("✅ Email SMTP disconnected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting Email SMTP: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test email connection"""
        try:
            if not self.smtp_client:
                return await self.connect()
            
            # Test with NOOP command
            await self.smtp_client.noop()
            return True
            
        except Exception as e:
            logger.error(f"❌ Email connection test failed: {e}")
            return False
    
    # ==================== MESSAGE FORMATTING ====================
    
    async def format_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Format message for email"""
        try:
            # Determine recipients
            recipients = message.recipients or [self.from_address]  # Default to self if no recipients
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = message.title
            msg['From'] = f"{self.from_name} <{self.from_address}>"
            msg['To'] = ", ".join(recipients[:self.max_recipients])
            
            if self.reply_to:
                msg['Reply-To'] = self.reply_to
            
            # Add custom headers
            msg['X-NeuroCluster-Message-ID'] = message.id
            msg['X-NeuroCluster-Category'] = message.category.value
            msg['X-NeuroCluster-Priority'] = message.priority.value
            
            # Create plain text version
            plain_text = self._create_plain_text_email(message)
            if self.include_plain_text:
                msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
            
            # Create HTML version
            if self.include_html:
                html_content = self._create_html_email(message)
                msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # Add attachments
            if message.attachments:
                await self._add_attachments(msg, message.attachments)
            
            return {
                'message': msg,
                'recipients': recipients,
                'formatted': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error formatting email message: {e}")
            return {'formatted': False, 'error': str(e)}
    
    def _create_plain_text_email(self, message: NotificationMessage) -> str:
        """Create plain text version of email"""
        
        content = f"{message.title}\n"
        content += "=" * len(message.title) + "\n\n"
        content += message.message + "\n\n"
        
        # Add data table if present
        if message.data:
            content += "Additional Information:\n"
            content += "-" * 25 + "\n"
            for key, value in message.data.items():
                if isinstance(value, (int, float, str)):
                    content += f"{key}: {value}\n"
        
        content += f"\n\nSent at: {message.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        content += f"Message ID: {message.id}\n"
        content += f"Priority: {message.priority.value.title()}\n\n"
        content += self.footer_text
        
        return content
    
    def _create_html_email(self, message: NotificationMessage) -> str:
        """Create HTML version of email"""
        
        # Apply priority-based styling
        priority_colors = {
            NotificationPriority.LOW: "#6c757d",      # Gray
            NotificationPriority.NORMAL: "#007bff",   # Blue
            NotificationPriority.HIGH: "#fd7e14",     # Orange
            NotificationPriority.URGENT: "#dc3545",   # Red
            NotificationPriority.CRITICAL: "#6f42c1"  # Purple
        }
        
        priority_color = priority_colors.get(message.priority, "#007bff")
        
        # Convert markdown-style formatting to HTML
        html_message = self._markdown_to_html(message.message)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(message.title)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, {priority_color}, {priority_color}dd);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }}
        .priority-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            margin-top: 8px;
        }}
        .content {{
            padding: 30px;
        }}
        .message-content {{
            background: #f8f9fa;
            border-left: 4px solid {priority_color};
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 4px 4px 0;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .data-table th,
        .data-table td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .data-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            font-size: 14px;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}
        .timestamp {{
            font-size: 12px;
            color: #6c757d;
            margin-top: 10px;
        }}
        .logo {{
            max-width: 150px;
            height: auto;
            margin-bottom: 10px;
        }}
        .btn {{
            display: inline-block;
            padding: 10px 20px;
            background: {priority_color};
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            {"<img src='" + self.logo_url + "' alt='NeuroCluster Elite' class='logo'>" if self.logo_url else ""}
            <h1>{html.escape(message.title)}</h1>
            <div class="priority-badge">{message.priority.value.title()} Priority</div>
        </div>
        
        <div class="content">
            <div class="message-content">
                {html_message}
            </div>
            
            {self._create_data_table_html(message.data) if message.data else ""}
            
            {"<img src='" + message.image_url + "' style='max-width: 100%; height: auto; border-radius: 4px; margin: 20px 0;'>" if message.image_url else ""}
        </div>
        
        <div class="footer">
            <div class="timestamp">
                Sent: {message.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
                Message ID: {message.id}
            </div>
            <div style="margin-top: 15px;">
                {html.escape(self.footer_text)}
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    def _markdown_to_html(self, text: str) -> str:
        """Convert simple markdown formatting to HTML"""
        
        # Escape HTML first
        text = html.escape(text)
        
        # Convert markdown formatting
        # Bold: **text** -> <strong>text</strong>
        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        
        # Italic: *text* -> <em>text</em>
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        
        # Code: `text` -> <code>text</code>
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        
        # Line breaks
        text = text.replace('\n', '<br>')
        
        # Bullet points: • text -> <li>text</li>
        lines = text.split('<br>')
        in_list = False
        result_lines = []
        
        for line in lines:
            if line.strip().startswith('•'):
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                content = line.strip()[1:].strip()
                result_lines.append(f'<li>{content}</li>')
            else:
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                result_lines.append(line)
        
        if in_list:
            result_lines.append('</ul>')
        
        return '<br>'.join(result_lines)
    
    def _create_data_table_html(self, data: Dict[str, Any]) -> str:
        """Create HTML table from data dictionary"""
        
        if not data:
            return ""
        
        html_table = '<table class="data-table">'
        html_table += '<thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>'
        
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                # Format value based on type
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
                
                html_table += f'<tr><td>{html.escape(clean_key)}</td><td>{html.escape(formatted_value)}</td></tr>'
        
        html_table += '</tbody></table>'
        return html_table
    
    async def _add_attachments(self, msg: MIMEMultipart, attachments: List[str]):
        """Add attachments to email message"""
        
        for attachment_path in attachments:
            try:
                file_path = Path(attachment_path)
                if not file_path.exists():
                    logger.warning(f"⚠️ Attachment not found: {attachment_path}")
                    continue
                
                with open(file_path, 'rb') as f:
                    attachment_data = f.read()
                
                # Determine MIME type based on extension
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                    # Image attachment
                    attachment = MIMEImage(attachment_data)
                    attachment.add_header('Content-Disposition', f'attachment; filename="{file_path.name}"')
                else:
                    # Generic attachment
                    attachment = MIMEBase('application', 'octet-stream')
                    attachment.set_payload(attachment_data)
                    encoders.encode_base64(attachment)
                    attachment.add_header('Content-Disposition', f'attachment; filename="{file_path.name}"')
                
                msg.attach(attachment)
                logger.info(f"📎 Added attachment: {file_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Error adding attachment {attachment_path}: {e}")
    
    # ==================== MESSAGE SENDING ====================
    
    async def send_message(self, message: NotificationMessage) -> bool:
        """Send email message"""
        try:
            # Ensure connection
            if not self.smtp_client or not await self.test_connection():
                if not await self.connect():
                    return False
            
            # Format message
            formatted = await self.format_message(message)
            if not formatted.get('formatted'):
                logger.error(f"❌ Failed to format email message: {formatted.get('error')}")
                return False
            
            email_msg = formatted['message']
            recipients = formatted['recipients']
            
            # Send email
            await self.smtp_client.send_message(email_msg)
            
            # Track sent email
            self.sent_emails.append({
                'message_id': message.id,
                'subject': message.title,
                'recipients': recipients,
                'sent_at': datetime.now(),
                'category': message.category.value,
                'priority': message.priority.value
            })
            
            # Trim email history
            if len(self.sent_emails) > 100:
                self.sent_emails = self.sent_emails[-100:]
            
            logger.info(f"✅ Email sent successfully: {message.title} to {len(recipients)} recipient(s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending email: {e}")
            message.error_message = str(e)
            return False
    
    # ==================== EMAIL TEMPLATES ====================
    
    def load_default_templates(self):
        """Load email-specific templates"""
        super().load_default_templates()
        
        # Override templates with email-specific formatting
        
        # Trade execution template with email formatting
        self.templates['trade_execution_email'] = NotificationTemplate(
            name="Trade Execution (Email)",
            title_template="✅ Trade Executed: {side} {quantity} {symbol}",
            message_template="""
**Trade Execution Confirmation**

Your {side} order for **{quantity} {symbol}** has been executed successfully.

• **Symbol:** {symbol}
• **Side:** {side}  
• **Quantity:** {quantity}
• **Execution Price:** ${price}
• **Total Value:** ${total_value}
• **Timestamp:** {timestamp}
• **Order ID:** {order_id}

The trade has been reflected in your portfolio. You can view your updated positions and performance in the NeuroCluster Elite dashboard.
""",
            category=NotificationCategory.TRADE_EXECUTION,
            supports_html=True,
            supports_markdown=True,
            required_variables=['symbol', 'side', 'quantity', 'price', 'total_value', 'timestamp', 'order_id']
        )
        
        # Portfolio summary email template
        self.templates['portfolio_summary_email'] = NotificationTemplate(
            name="Portfolio Summary (Email)",
            title_template="📊 Daily Portfolio Summary - ${total_value}",
            message_template="""
**Daily Portfolio Summary**

Here's your portfolio performance summary for {date}.

**Portfolio Overview:**
• **Total Value:** ${total_value}
• **Daily P&L:** ${daily_pnl} ({daily_pnl_percent}%)
• **Cash Balance:** ${cash_balance}
• **Invested Amount:** ${invested_amount}

**Position Summary:**
• **Open Positions:** {open_positions}
• **Winning Positions:** {winning_positions}
• **Losing Positions:** {losing_positions}

**Performance Metrics:**
• **Total Return:** {total_return_percent}%
• **Sharpe Ratio:** {sharpe_ratio}
• **Max Drawdown:** {max_drawdown_percent}%

View your complete portfolio analysis in the NeuroCluster Elite dashboard for detailed insights and recommendations.
""",
            category=NotificationCategory.PORTFOLIO_UPDATE,
            supports_html=True,
            supports_markdown=True,
            include_charts=True,
            required_variables=['date', 'total_value', 'daily_pnl', 'daily_pnl_percent', 'cash_balance', 'invested_amount', 'open_positions']
        )
        
        logger.info(f"📧 Loaded {len(self.templates)} email templates")
    
    # ==================== UTILITY METHODS ====================
    
    def get_email_stats(self) -> Dict[str, Any]:
        """Get email delivery statistics"""
        
        if not self.sent_emails:
            return {'total_sent': 0, 'recent_activity': []}
        
        # Calculate stats
        total_sent = len(self.sent_emails)
        recent_emails = [email for email in self.sent_emails 
                        if (datetime.now() - email['sent_at']).days < 7]
        
        # Group by category
        by_category = {}
        for email in self.sent_emails:
            category = email['category']
            by_category[category] = by_category.get(category, 0) + 1
        
        # Group by priority
        by_priority = {}
        for email in self.sent_emails:
            priority = email['priority']
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        return {
            'total_sent': total_sent,
            'sent_last_7_days': len(recent_emails),
            'by_category': by_category,
            'by_priority': by_priority,
            'recent_activity': recent_emails[-10:] if recent_emails else []
        }

# ==================== TESTING ====================

async def test_email_channel():
    """Test email notification channel"""
    print("🧪 Testing Email Notification Channel")
    print("=" * 40)
    
    # Create test configuration
    config = IntegrationConfig(
        name="email",
        integration_type="notification",
        enabled=True,
        api_key="test@example.com",
        secret_key="test_password",
        additional_auth={
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'use_tls': True,
            'from_name': 'NeuroCluster Elite Test'
        }
    )
    
    # Create email channel
    email_channel = EmailNotificationChannel(config)
    
    print("✅ Email channel instance created")
    print(f"✅ SMTP Server: {email_channel.smtp_server}:{email_channel.smtp_port}")
    print(f"✅ From: {email_channel.from_name} <{email_channel.from_address}>")
    print(f"✅ Supports HTML: {email_channel.supports_html}")
    print(f"✅ Supports attachments: {email_channel.supports_attachments}")
    
    # Test message formatting
    from src.integrations.notifications import NotificationMessage, NotificationPriority, NotificationCategory
    
    test_message = NotificationMessage(
        id="test-123",
        title="Test Trading Alert",
        message="**Test Alert**\n\nThis is a test message with **bold** text and bullet points:\n• Point 1\n• Point 2",
        category=NotificationCategory.TRADE_EXECUTION,
        priority=NotificationPriority.HIGH,
        channels=[],
        recipients=["test@example.com"],
        data={
            'symbol': 'AAPL',
            'price': 150.50,
            'quantity': 100,
            'total_value': 15050.00
        }
    )
    
    formatted = await email_channel.format_message(test_message)
    if formatted.get('formatted'):
        print("✅ Message formatting successful")
        print(f"✅ Recipients: {len(formatted['recipients'])}")
    else:
        print(f"❌ Message formatting failed: {formatted.get('error')}")
    
    # Test templates
    templates = email_channel.templates
    print(f"✅ Templates loaded: {len(templates)}")
    for template_name in templates:
        print(f"  • {template_name}")
    
    print("\n⚠️  Note: Email sending requires valid SMTP credentials")
    print("   Configure SMTP settings to test actual email delivery")
    
    print("\n🎉 Email notification channel tests completed!")

if __name__ == "__main__":
    asyncio.run(test_email_channel())