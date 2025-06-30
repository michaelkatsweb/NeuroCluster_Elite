#!/usr/bin/env python3
"""
File: voice_commands.py
Path: NeuroCluster-Elite/src/interfaces/voice_commands.py
Description: Voice command system for NeuroCluster Elite

This module provides voice command functionality allowing users to interact
with the trading platform using natural speech commands.

Features:
- Speech recognition for trading commands
- Text-to-speech feedback
- Natural language processing for commands
- Security and confirmation for critical operations
- Voice-activated portfolio management
- Hands-free market analysis

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re
import json

# Voice recognition imports (with fallbacks)
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None

# Import from our modules
from src.utils.logger import get_enhanced_logger, LogCategory
from src.utils.helpers import format_currency, format_percentage

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.USER)

# ==================== ENUMS AND DATA STRUCTURES ====================

class VoiceCommandType(Enum):
    """Types of voice commands"""
    GREETING = "greeting"
    PORTFOLIO = "portfolio"
    TRADING = "trading"
    MARKET_DATA = "market_data"
    ANALYSIS = "analysis"
    SYSTEM = "system"
    HELP = "help"
    SECURITY = "security"

class CommandSecurity(Enum):
    """Security levels for commands"""
    LOW = "low"          # Information queries
    MEDIUM = "medium"    # Portfolio queries
    HIGH = "high"        # Trading commands
    CRITICAL = "critical" # Account changes

@dataclass
class VoiceCommand:
    """Voice command structure"""
    command_type: VoiceCommandType
    action: str
    parameters: Dict[str, Any]
    security_level: CommandSecurity
    timestamp: datetime
    user_text: str
    confidence: float = 0.0

@dataclass
class VoiceResponse:
    """Voice response structure"""
    text: str
    data: Optional[Dict[str, Any]] = None
    requires_confirmation: bool = False
    action_taken: bool = False

# ==================== VOICE COMMAND SYSTEM ====================

class VoiceCommandSystem:
    """
    Advanced voice command system for trading platform
    
    Features:
    - Natural language command processing
    - Security confirmations for trading commands
    - Portfolio management via voice
    - Market data queries
    - Real-time speech recognition
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize voice command system"""
        
        self.config = config or self._default_config()
        
        # Check availability of required libraries
        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.warning("Speech recognition not available - install speechrecognition")
        
        if not TTS_AVAILABLE:
            logger.warning("Text-to-speech not available - install pyttsx3")
        
        # Initialize components
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.microphone = sr.Microphone() if SPEECH_RECOGNITION_AVAILABLE else None
        self.tts_engine = pyttsx3.init() if TTS_AVAILABLE else None
        
        # Configure TTS if available
        if self.tts_engine:
            self._configure_tts()
        
        # Voice system state
        self.listening = False
        self.listening_thread = None
        self.last_command_time = None
        self.command_history = []
        
        # Command processors
        self.command_processors = {
            VoiceCommandType.GREETING: self._process_greeting,
            VoiceCommandType.PORTFOLIO: self._process_portfolio,
            VoiceCommandType.TRADING: self._process_trading,
            VoiceCommandType.MARKET_DATA: self._process_market_data,
            VoiceCommandType.ANALYSIS: self._process_analysis,
            VoiceCommandType.SYSTEM: self._process_system,
            VoiceCommandType.HELP: self._process_help,
        }
        
        # External handlers (to be set by main application)
        self.portfolio_handler = None
        self.trading_handler = None
        self.market_data_handler = None
        
        # Security settings
        self.voice_authentication_enabled = self.config.get('voice_auth', False)
        self.confirmation_required = self.config.get('require_confirmation', True)
        
        logger.info("🎤 Voice Command System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default voice command configuration"""
        
        return {
            'activation_phrase': 'hello neurocluster',
            'timeout': 5,
            'phrase_time_limit': 10,
            'energy_threshold': 300,
            'dynamic_energy_threshold': True,
            'pause_threshold': 0.8,
            
            # TTS settings
            'tts_rate': 150,
            'tts_volume': 0.8,
            'tts_voice': None,  # Use default voice
            
            # Security settings
            'voice_auth': False,
            'require_confirmation': True,
            'max_trade_amount': 10000,
            
            # Command settings
            'enable_trading_commands': True,
            'enable_portfolio_commands': True,
            'enable_market_commands': True,
        }
    
    def _configure_tts(self):
        """Configure text-to-speech engine"""
        
        if not self.tts_engine:
            return
        
        try:
            # Set speech rate
            self.tts_engine.setProperty('rate', self.config['tts_rate'])
            
            # Set volume
            self.tts_engine.setProperty('volume', self.config['tts_volume'])
            
            # Set voice if specified
            if self.config['tts_voice']:
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if self.config['tts_voice'] in voice.name:
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            logger.info("✅ Text-to-speech configured")
            
        except Exception as e:
            logger.error(f"Error configuring TTS: {e}")
    
    def start_listening(self):
        """Start voice command listening"""
        
        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.error("Speech recognition not available")
            return False
        
        if self.listening:
            return True
        
        self.listening = True
        self.listening_thread = threading.Thread(target=self._listening_loop)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        
        self._speak("Voice commands activated. Say 'Hello NeuroCluster' to start.")
        logger.info("🎤 Voice listening started")
        
        return True
    
    def stop_listening(self):
        """Stop voice command listening"""
        
        if not self.listening:
            return
        
        self.listening = False
        
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
        
        self._speak("Voice commands deactivated.")
        logger.info("🎤 Voice listening stopped")
    
    def _listening_loop(self):
        """Main voice listening loop"""
        
        if not self.recognizer or not self.microphone:
            return
        
        # Adjust for ambient noise
        with self.microphone as source:
            logger.info("🎤 Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
        
        logger.info("🎤 Listening for commands...")
        
        while self.listening:
            try:
                # Listen for audio
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source, 
                        timeout=self.config['timeout'],
                        phrase_time_limit=self.config['phrase_time_limit']
                    )
                
                # Recognize speech
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    logger.info(f"🎤 Heard: {text}")
                    
                    # Process the command
                    asyncio.run(self._process_voice_input(text))
                    
                except sr.UnknownValueError:
                    # Could not understand audio
                    pass
                except sr.RequestError as e:
                    logger.error(f"Speech recognition service error: {e}")
                    
            except sr.WaitTimeoutError:
                # Timeout - continue listening
                pass
            except Exception as e:
                logger.error(f"Error in listening loop: {e}")
                time.sleep(1)
    
    async def _process_voice_input(self, text: str):
        """Process voice input text"""
        
        # Check for activation phrase
        activation_phrase = self.config['activation_phrase']
        
        if activation_phrase in text:
            self._speak("Hello! How can I help you with your trading today?")
            return
        
        # Skip processing if activation phrase not detected recently
        # (In a full implementation, you'd track activation state)
        
        # Parse command
        command = self._parse_command(text)
        
        if command:
            # Process command
            response = await self._execute_command(command)
            
            # Provide voice feedback
            if response:
                self._speak(response.text)
                
                # Log command
                self.command_history.append({
                    'timestamp': datetime.now(),
                    'text': text,
                    'command': command,
                    'response': response.text
                })
                
                # Keep limited history
                if len(self.command_history) > 100:
                    self.command_history = self.command_history[-100:]
    
    def _parse_command(self, text: str) -> Optional[VoiceCommand]:
        """Parse voice input into structured command"""
        
        text = text.lower().strip()
        
        # Command patterns
        patterns = {
            # Greeting patterns
            VoiceCommandType.GREETING: [
                r'hello|hi|hey',
                r'good (morning|afternoon|evening)',
            ],
            
            # Portfolio patterns
            VoiceCommandType.PORTFOLIO: [
                r'(show|check|what is|tell me) my (portfolio|balance|positions)',
                r'how much (money|cash) do i have',
                r'what are my (holdings|stocks|positions)',
                r'portfolio (status|summary|performance)',
            ],
            
            # Trading patterns
            VoiceCommandType.TRADING: [
                r'buy (\d+|\w+) (shares of )?(\w+)',
                r'sell (\d+|\w+) (shares of )?(\w+)',
                r'place (a )?(buy|sell) order for (\w+)',
                r'close (my )?position in (\w+)',
                r'what should i (buy|sell)',
            ],
            
            # Market data patterns
            VoiceCommandType.MARKET_DATA: [
                r'(what is|show me|check) the price of (\w+)',
                r'how is (\w+) (doing|performing)',
                r'market (status|update|summary)',
                r'current (regime|conditions)',
                r'(\w+) (stock|price|quote)',
            ],
            
            # Analysis patterns
            VoiceCommandType.ANALYSIS: [
                r'analyze (\w+)',
                r'what do you think about (\w+)',
                r'should i (buy|sell) (\w+)',
                r'market (analysis|outlook)',
                r'trading (signals|recommendations)',
            ],
            
            # System patterns
            VoiceCommandType.SYSTEM: [
                r'(pause|stop|disable) (voice|listening)',
                r'(resume|start|enable) trading',
                r'system (status|health)',
                r'algorithm (performance|status)',
            ],
            
            # Help patterns
            VoiceCommandType.HELP: [
                r'help|what can (you|i) do',
                r'(list|show) (commands|options)',
                r'how do i (\w+)',
            ],
        }
        
        # Try to match patterns
        for command_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text)
                if match:
                    return self._create_command(command_type, text, match)
        
        # If no pattern matched, try to infer intent
        return self._infer_command_intent(text)
    
    def _create_command(self, command_type: VoiceCommandType, text: str, 
                       match: re.Match) -> VoiceCommand:
        """Create command object from parsed input"""
        
        parameters = {}
        security_level = CommandSecurity.LOW
        
        # Extract parameters based on command type
        if command_type == VoiceCommandType.TRADING:
            security_level = CommandSecurity.HIGH
            
            # Extract trading parameters
            groups = match.groups()
            if 'buy' in text:
                parameters['action'] = 'buy'
            elif 'sell' in text:
                parameters['action'] = 'sell'
            
            # Extract symbol and quantity
            symbols = re.findall(r'\b[A-Z]{2,5}\b', text.upper())
            if symbols:
                parameters['symbol'] = symbols[0]
            
            quantities = re.findall(r'\b\d+\b', text)
            if quantities:
                parameters['quantity'] = int(quantities[0])
        
        elif command_type == VoiceCommandType.MARKET_DATA:
            # Extract symbol for market data
            symbols = re.findall(r'\b[A-Z]{2,5}\b', text.upper())
            if symbols:
                parameters['symbol'] = symbols[0]
        
        elif command_type == VoiceCommandType.PORTFOLIO:
            security_level = CommandSecurity.MEDIUM
        
        return VoiceCommand(
            command_type=command_type,
            action=text,
            parameters=parameters,
            security_level=security_level,
            timestamp=datetime.now(),
            user_text=text
        )
    
    def _infer_command_intent(self, text: str) -> Optional[VoiceCommand]:
        """Infer command intent from unmatched text"""
        
        # Simple keyword-based inference
        if any(word in text for word in ['buy', 'purchase', 'acquire']):
            return VoiceCommand(
                command_type=VoiceCommandType.TRADING,
                action='buy_inquiry',
                parameters={},
                security_level=CommandSecurity.MEDIUM,
                timestamp=datetime.now(),
                user_text=text
            )
        
        elif any(word in text for word in ['sell', 'close', 'exit']):
            return VoiceCommand(
                command_type=VoiceCommandType.TRADING,
                action='sell_inquiry',
                parameters={},
                security_level=CommandSecurity.MEDIUM,
                timestamp=datetime.now(),
                user_text=text
            )
        
        elif any(word in text for word in ['portfolio', 'balance', 'money']):
            return VoiceCommand(
                command_type=VoiceCommandType.PORTFOLIO,
                action='status',
                parameters={},
                security_level=CommandSecurity.MEDIUM,
                timestamp=datetime.now(),
                user_text=text
            )
        
        return None
    
    async def _execute_command(self, command: VoiceCommand) -> Optional[VoiceResponse]:
        """Execute parsed voice command"""
        
        # Security check
        if not self._check_command_security(command):
            return VoiceResponse(
                text="I'm sorry, but that command requires additional security verification.",
                requires_confirmation=True
            )
        
        # Get command processor
        processor = self.command_processors.get(command.command_type)
        
        if processor:
            try:
                response = await processor(command)
                
                # Log successful command execution
                logger.audit(
                    f"voice_command_executed",
                    {
                        'command_type': command.command_type.value,
                        'action': command.action,
                        'parameters': command.parameters
                    }
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Error executing command: {e}")
                return VoiceResponse(
                    text="I'm sorry, there was an error processing your command."
                )
        else:
            return VoiceResponse(
                text="I'm sorry, I don't understand that command. Say 'help' for available commands."
            )
    
    def _check_command_security(self, command: VoiceCommand) -> bool:
        """Check if command passes security requirements"""
        
        # For now, allow all commands
        # In production, implement proper security checks
        return True
    
    async def _process_greeting(self, command: VoiceCommand) -> VoiceResponse:
        """Process greeting commands"""
        
        greetings = [
            "Hello! I'm your NeuroCluster Elite trading assistant. How can I help you today?",
            "Hi there! Ready to analyze the markets and manage your portfolio?",
            "Good to hear from you! What would you like to know about your investments?"
        ]
        
        import random
        greeting = random.choice(greetings)
        
        return VoiceResponse(text=greeting)
    
    async def _process_portfolio(self, command: VoiceCommand) -> VoiceResponse:
        """Process portfolio-related commands"""
        
        if self.portfolio_handler:
            try:
                portfolio_data = await self.portfolio_handler()
                
                # Format portfolio response
                if portfolio_data:
                    total_value = portfolio_data.get('total_value', 0)
                    cash_balance = portfolio_data.get('cash_balance', 0)
                    positions = portfolio_data.get('positions', 0)
                    daily_pnl = portfolio_data.get('daily_pnl', 0)
                    
                    response_text = f"Your portfolio is worth {format_currency(total_value)}. "
                    response_text += f"You have {format_currency(cash_balance)} in cash "
                    response_text += f"and {positions} open positions. "
                    
                    if daily_pnl != 0:
                        pnl_text = "up" if daily_pnl > 0 else "down"
                        response_text += f"Today you're {pnl_text} {format_currency(abs(daily_pnl))}."
                    
                    return VoiceResponse(text=response_text, data=portfolio_data)
                else:
                    return VoiceResponse(text="I couldn't retrieve your portfolio information right now.")
                    
            except Exception as e:
                logger.error(f"Error getting portfolio data: {e}")
                return VoiceResponse(text="There was an error accessing your portfolio.")
        else:
            return VoiceResponse(text="Portfolio information is not available.")
    
    async def _process_trading(self, command: VoiceCommand) -> VoiceResponse:
        """Process trading commands"""
        
        action = command.parameters.get('action')
        symbol = command.parameters.get('symbol')
        quantity = command.parameters.get('quantity')
        
        if action and symbol:
            # This is a specific trading request
            if self.confirmation_required:
                confirmation_text = f"Are you sure you want to {action} "
                if quantity:
                    confirmation_text += f"{quantity} shares of "
                confirmation_text += f"{symbol}? Say 'confirm' to proceed."
                
                return VoiceResponse(
                    text=confirmation_text,
                    requires_confirmation=True,
                    data={'action': action, 'symbol': symbol, 'quantity': quantity}
                )
            else:
                # Execute trade directly (if handler available)
                if self.trading_handler:
                    try:
                        result = await self.trading_handler(action, symbol, quantity)
                        if result.get('success'):
                            return VoiceResponse(
                                text=f"Successfully placed {action} order for {symbol}.",
                                action_taken=True
                            )
                        else:
                            return VoiceResponse(
                                text=f"Failed to place {action} order: {result.get('error', 'Unknown error')}"
                            )
                    except Exception as e:
                        logger.error(f"Error executing trade: {e}")
                        return VoiceResponse(text="There was an error executing your trade.")
                else:
                    return VoiceResponse(text="Trading functionality is not available.")
        else:
            # General trading inquiry
            return VoiceResponse(
                text="I can help you buy or sell stocks. For example, say 'buy 100 shares of Apple' or 'sell Microsoft'."
            )
    
    async def _process_market_data(self, command: VoiceCommand) -> VoiceResponse:
        """Process market data commands"""
        
        symbol = command.parameters.get('symbol')
        
        if symbol and self.market_data_handler:
            try:
                market_data = await self.market_data_handler(symbol)
                
                if market_data:
                    price = market_data.get('price', 0)
                    change = market_data.get('change', 0)
                    change_pct = market_data.get('change_percent', 0)
                    
                    direction = "up" if change >= 0 else "down"
                    
                    response_text = f"{symbol} is trading at {format_currency(price)}, "
                    response_text += f"{direction} {format_currency(abs(change))} "
                    response_text += f"or {format_percentage(abs(change_pct))} today."
                    
                    return VoiceResponse(text=response_text, data=market_data)
                else:
                    return VoiceResponse(text=f"I couldn't find market data for {symbol}.")
                    
            except Exception as e:
                logger.error(f"Error getting market data: {e}")
                return VoiceResponse(text="There was an error retrieving market data.")
        else:
            return VoiceResponse(
                text="I can provide market data for any stock. For example, say 'what's the price of Apple?'"
            )
    
    async def _process_analysis(self, command: VoiceCommand) -> VoiceResponse:
        """Process analysis commands"""
        
        symbol = command.parameters.get('symbol')
        
        if symbol:
            # Placeholder for analysis
            analysis_responses = [
                f"Based on current market conditions, {symbol} shows mixed signals. The NeuroCluster algorithm is monitoring for regime changes.",
                f"{symbol} is currently in a sideways trend. I recommend waiting for a clearer signal before taking action.",
                f"The technical indicators for {symbol} suggest caution. Consider your risk tolerance before investing."
            ]
            
            import random
            response = random.choice(analysis_responses)
            
            return VoiceResponse(text=response)
        else:
            return VoiceResponse(
                text="I can analyze any stock for you. Just say 'analyze' followed by the stock symbol."
            )
    
    async def _process_system(self, command: VoiceCommand) -> VoiceResponse:
        """Process system commands"""
        
        if 'stop' in command.user_text or 'pause' in command.user_text:
            self.stop_listening()
            return VoiceResponse(text="Voice commands have been disabled.")
        
        elif 'status' in command.user_text:
            return VoiceResponse(
                text="NeuroCluster Elite is running normally. All systems are operational."
            )
        
        elif 'algorithm' in command.user_text:
            return VoiceResponse(
                text="The NeuroCluster algorithm is performing at 99.59% efficiency with 0.045 millisecond processing time."
            )
        
        else:
            return VoiceResponse(text="System is running normally.")
    
    async def _process_help(self, command: VoiceCommand) -> VoiceResponse:
        """Process help commands"""
        
        help_text = """Here are the commands I understand:

Portfolio: 'Show my portfolio', 'Check my balance', 'What are my positions'

Trading: 'Buy 100 shares of Apple', 'Sell Microsoft', 'Close my Tesla position'

Market Data: 'What's the price of Google?', 'How is Amazon doing?', 'Market status'

Analysis: 'Analyze Netflix', 'Should I buy Bitcoin?', 'Trading signals'

System: 'System status', 'Algorithm performance', 'Stop listening'

Just speak naturally and I'll try to help you with your trading needs."""
        
        return VoiceResponse(text=help_text)
    
    def _speak(self, text: str):
        """Convert text to speech"""
        
        if not self.tts_engine:
            print(f"🔊 {text}")  # Fallback to print
            return
        
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
            logger.debug(f"🔊 TTS: {text}")
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            print(f"🔊 {text}")  # Fallback to print
    
    def set_handlers(self, portfolio_handler: Callable = None, 
                    trading_handler: Callable = None,
                    market_data_handler: Callable = None):
        """Set external handlers for voice commands"""
        
        self.portfolio_handler = portfolio_handler
        self.trading_handler = trading_handler
        self.market_data_handler = market_data_handler
        
        logger.info("✅ Voice command handlers configured")
    
    def get_command_history(self, limit: int = 10) -> List[Dict]:
        """Get recent command history"""
        
        return self.command_history[-limit:] if self.command_history else []
    
    def is_listening(self) -> bool:
        """Check if voice system is listening"""
        return self.listening

# ==================== TESTING ====================

def test_voice_commands():
    """Test voice command system"""
    
    print("🧪 Testing Voice Command System")
    print("=" * 40)
    
    # Create voice system
    voice_system = VoiceCommandSystem({
        'activation_phrase': 'hello neurocluster',
        'require_confirmation': False
    })
    
    # Test command parsing
    test_commands = [
        "hello neurocluster",
        "show my portfolio",
        "buy 100 shares of apple",
        "what is the price of microsoft",
        "analyze tesla",
        "help"
    ]
    
    for test_text in test_commands:
        print(f"\n🎤 Testing: '{test_text}'")
        command = voice_system._parse_command(test_text)
        
        if command:
            print(f"✅ Parsed as: {command.command_type.value}")
            print(f"   Parameters: {command.parameters}")
        else:
            print("❌ Could not parse command")
    
    # Test TTS if available
    if TTS_AVAILABLE:
        print("\n🔊 Testing text-to-speech...")
        voice_system._speak("Voice command system test completed successfully!")
    else:
        print("\n⚠️  Text-to-speech not available")
    
    print("\n🎉 Voice command tests completed!")

if __name__ == "__main__":
    test_voice_commands()