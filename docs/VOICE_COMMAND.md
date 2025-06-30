# 🎤 NeuroCluster Elite Voice Commands Reference

Complete reference for voice control of the NeuroCluster Elite trading platform. Trade hands-free with natural language commands powered by advanced speech recognition and NLP.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Setup & Configuration](#setup--configuration)
- [Basic Commands](#basic-commands)
- [Portfolio Commands](#portfolio-commands)
- [Trading Commands](#trading-commands)
- [Market Analysis](#market-analysis)
- [Strategy Commands](#strategy-commands)
- [Risk Management](#risk-management)
- [Alert & Notification Commands](#alert--notification-commands)
- [System Commands](#system-commands)
- [Advanced Voice Features](#advanced-voice-features)
- [Troubleshooting](#troubleshooting)
- [Customization](#customization)

---

## 🚀 Getting Started

### Quick Start

```bash
# Enable voice commands in your NeuroCluster Elite installation
python main_console.py --enable-voice

# Or enable in the dashboard
# Settings > Voice Control > Enable Voice Commands
```

### First Commands

Try these basic commands to get started:

- **"Hello NeuroCluster"** - Activate voice control
- **"Show my portfolio"** - Display portfolio status
- **"What's the market doing?"** - Get market overview
- **"Help with voice commands"** - Get voice command help

---

## ⚙️ Setup & Configuration

### Voice Activation

#### Wake Word Commands
| Command | Action |
|---------|--------|
| "Hey NeuroCluster" | Activate voice listening |
| "Hello NeuroCluster" | Activate voice listening |
| "NeuroCluster Elite" | Activate voice listening |
| "Computer" | Activate voice listening (alternative) |

#### Voice Settings Commands
| Command | Action |
|---------|--------|
| "Enable voice commands" | Turn on voice control |
| "Disable voice commands" | Turn off voice control |
| "Set voice language to English" | Change language |
| "Increase voice sensitivity" | Adjust microphone sensitivity |
| "Decrease voice sensitivity" | Lower microphone sensitivity |
| "Calibrate microphone" | Run microphone calibration |

### Audio Configuration

```python
# Voice configuration in settings
VOICE_CONFIG = {
    'wake_words': ['hey neurocluster', 'hello neurocluster'],
    'language': 'en-US',
    'sensitivity': 0.7,
    'timeout': 5,
    'continuous_listening': False,
    'confirmation_sounds': True
}
```

---

## 🔤 Basic Commands

### General Navigation
| Command | Action | Example Response |
|---------|--------|------------------|
| "Show dashboard" | Open main dashboard | "Opening dashboard" |
| "Go to portfolio" | Navigate to portfolio view | "Switching to portfolio view" |
| "Open settings" | Access settings menu | "Opening settings panel" |
| "Help" / "What can I do?" | Show available commands | Lists command categories |
| "Repeat that" | Repeat last response | Repeats previous information |
| "Never mind" / "Cancel" | Cancel current operation | "Operation cancelled" |

### Information Queries
| Command | Action |
|---------|--------|
| "What time is it?" | Current time |
| "What's today's date?" | Current date |
| "Show system status" | System health check |
| "How long have I been trading today?" | Trading session duration |
| "What's my trading performance today?" | Daily P&L summary |

### Voice Feedback Control
| Command | Action |
|---------|--------|
| "Speak faster" | Increase speech rate |
| "Speak slower" | Decrease speech rate |
| "Louder" / "Speak up" | Increase volume |
| "Quieter" / "Lower volume" | Decrease volume |
| "Stop talking" | Silence current response |

---

## 💼 Portfolio Commands

### Portfolio Overview
| Command | Example Response |
|---------|------------------|
| "Show my portfolio" | "Your portfolio is worth $105,250 with $2,500 in unrealized gains" |
| "Portfolio status" | Displays current holdings and performance |
| "How much money do I have?" | "You have $25,000 in cash and $105,250 total portfolio value" |
| "What's my biggest position?" | "Your largest position is AAPL with $15,000 market value" |
| "Show my cash balance" | "Your available cash is $25,000" |

### Position Details
| Command | Action |
|---------|--------|
| "Show my Apple position" | Display AAPL position details |
| "How much Tesla do I own?" | Show TSLA holdings |
| "What's my Amazon profit?" | Show AMZN unrealized P&L |
| "List all my positions" | Show complete position list |
| "Show losing positions" | Display positions with losses |
| "Show winning positions" | Display profitable positions |

### Performance Queries
| Command | Example Response |
|---------|------------------|
| "How did I do today?" | "Today you're up $1,250 or 1.2%" |
| "What's my best performing stock?" | "NVDA is your best performer with +15.2%" |
| "What's my worst position?" | "META is down -2.8% or -$340" |
| "Show my total returns" | "Total portfolio return: +12.5% or +$11,250" |
| "How much have I made this week?" | Weekly performance summary |
| "What's my Sharpe ratio?" | Risk-adjusted return metrics |

---

## 📈 Trading Commands

### Market Orders
| Command | Action |
|---------|--------|
| "Buy 100 shares of Apple" | Place market buy order for AAPL |
| "Sell 50 shares of Tesla" | Place market sell order for TSLA |
| "Buy $1000 worth of Google" | Dollar-based market order |
| "Sell all my Amazon" | Close entire AMZN position |
| "Buy maximum Apple shares" | Use all available cash for AAPL |

### Limit Orders
| Command | Action |
|---------|--------|
| "Buy Apple at $150" | Place limit buy order |
| "Sell Tesla at $250 or better" | Place limit sell order |
| "Set buy limit for Google at $2800" | Limit order with specific price |
| "Cancel my Apple buy order" | Cancel pending order |
| "Show my pending orders" | List all open orders |

### Stop Orders
| Command | Action |
|---------|--------|
| "Set stop loss on Apple at $140" | Place stop loss order |
| "Put a trailing stop on Tesla with 5%" | Trailing stop loss |
| "Set take profit on Google at $3000" | Take profit order |
| "Remove stop loss on Apple" | Cancel stop order |
| "Set protective stops on all positions" | Auto-set stops for all holdings |

### Order Management
| Command | Action |
|---------|--------|
| "Show my recent trades" | Display trade history |
| "Cancel all pending orders" | Cancel all open orders |
| "What orders do I have?" | List active orders |
| "Modify my Apple order to $145" | Change order price |
| "How many shares can I buy?" | Calculate buying power |

---

## 📊 Market Analysis

### Price Quotes
| Command | Example Response |
|---------|------------------|
| "What's Apple trading at?" | "Apple is at $152.30, up $2.75 or 1.8%" |
| "Give me a Tesla quote" | Current TSLA price and change |
| "How's the market doing?" | Overall market summary |
| "Show me the Dow Jones" | Index level and performance |
| "What's Bitcoin's price?" | Cryptocurrency price |

### Technical Analysis
| Command | Action |
|---------|--------|
| "Show me Apple's chart" | Display AAPL price chart |
| "What's Tesla's RSI?" | Technical indicator value |
| "Is Apple overbought?" | Technical analysis summary |
| "Show support and resistance for Google" | Key price levels |
| "What's the moving average for Tesla?" | MA analysis |

### Market Sentiment
| Command | Example Response |
|---------|------------------|
| "What's the market sentiment?" | "Market sentiment is bullish with VIX at 18.5" |
| "How's investor sentiment on Apple?" | Sentiment analysis for AAPL |
| "Show me fear and greed index" | Market emotion indicators |
| "What's the news on Tesla?" | Recent news sentiment |
| "Are people buying or selling?" | Market flow analysis |

### Watchlist Management
| Command | Action |
|---------|--------|
| "Add Apple to my watchlist" | Add AAPL to watchlist |
| "Remove Tesla from watchlist" | Remove TSLA from watchlist |
| "Show my watchlist" | Display watched symbols |
| "Clear my watchlist" | Remove all watched symbols |
| "Add Bitcoin to crypto watchlist" | Cryptocurrency tracking |

---

## 🎯 Strategy Commands

### Strategy Status
| Command | Action |
|---------|--------|
| "What strategy am I using?" | Show active trading strategy |
| "How's my strategy performing?" | Strategy performance metrics |
| "Show strategy signals" | Current trading signals |
| "Switch to bull market strategy" | Change active strategy |
| "Disable automatic trading" | Turn off auto-execution |

### Signal Management
| Command | Example Response |
|---------|------------------|
| "Any new signals?" | "Yes, new BUY signal for AAPL with 85% confidence" |
| "Show all signals" | List current trading signals |
| "Execute the Apple signal" | Trade based on signal |
| "Ignore Tesla signal" | Dismiss signal |
| "What's the confidence on that signal?" | Signal strength details |

### Strategy Configuration
| Command | Action |
|---------|--------|
| "Set strategy to conservative" | Adjust risk parameters |
| "Make strategy more aggressive" | Increase risk tolerance |
| "Enable momentum strategy" | Switch strategy type |
| "Set minimum confidence to 80%" | Adjust signal threshold |
| "Show strategy settings" | Display current parameters |

---

## ⚖️ Risk Management

### Risk Monitoring
| Command | Example Response |
|---------|------------------|
| "What's my portfolio risk?" | "Portfolio beta is 1.15, VaR is 2.1%" |
| "Show my exposure" | Position sizing and concentration |
| "Am I too concentrated?" | Diversification analysis |
| "What's my maximum drawdown?" | Risk metrics |
| "Check my position sizes" | Size limits validation |

### Risk Controls
| Command | Action |
|---------|--------|
| "Set daily loss limit to $1000" | Configure stop loss |
| "Reduce position size limits" | Adjust risk parameters |
| "Enable portfolio protection" | Activate risk controls |
| "Set maximum allocation to 10%" | Position size limits |
| "Stop all trading" | Emergency trading halt |

### Alerts and Warnings
| Command | Action |
|---------|--------|
| "Warn me if I lose more than $500 today" | Set loss alert |
| "Alert me when Apple hits $150" | Price alert |
| "Notify me of margin calls" | Risk notification |
| "Tell me if volatility spikes" | Market condition alert |
| "Warn me about concentration risk" | Diversification alert |

---

## 🔔 Alert & Notification Commands

### Price Alerts
| Command | Action |
|---------|--------|
| "Alert me when Apple hits $160" | Set price alert |
| "Notify me if Tesla drops below $200" | Downside alert |
| "Tell me when Bitcoin reaches $50,000" | Cryptocurrency alert |
| "Alert me if the market drops 2%" | Market-wide alert |
| "Remove Apple price alert" | Cancel alert |

### Performance Alerts
| Command | Action |
|---------|--------|
| "Tell me if I lose more than $1000 today" | Daily loss alert |
| "Alert me when I make $5000 profit" | Profit target alert |
| "Notify me of big moves in my portfolio" | Portfolio movement alert |
| "Tell me if any position moves 5%" | Position volatility alert |
| "Alert me of margin requirements" | Account alert |

### News and Events
| Command | Action |
|---------|--------|
| "Alert me of Apple earnings" | Earnings announcement alert |
| "Notify me of Fed announcements" | Economic event alert |
| "Tell me about Tesla news" | Company-specific news |
| "Alert me of market-moving news" | Breaking news alert |
| "Show my alert settings" | List all active alerts |

---

## 🖥️ System Commands

### System Status
| Command | Example Response |
|---------|------------------|
| "System status" | "All systems operational, 99.8% uptime" |
| "How's the algorithm performing?" | "Algorithm efficiency: 99.59%" |
| "Check data feeds" | Market data connection status |
| "Is everything working?" | Complete health check |
| "Show system metrics" | Performance statistics |

### Data and Connectivity
| Command | Action |
|---------|--------|
| "Refresh market data" | Update price feeds |
| "Check broker connection" | Verify trading connection |
| "Reconnect to data feed" | Reset data connections |
| "Update prices" | Force price refresh |
| "Test trading connection" | Verify order execution |

### Settings and Preferences
| Command | Action |
|---------|--------|
| "Switch to dark mode" | Change interface theme |
| "Show me in voice mode" | Enable audio responses |
| "Set currency to USD" | Change display currency |
| "Enable paper trading" | Switch to simulation mode |
| "Save current settings" | Persist configuration |

---

## 🎛️ Advanced Voice Features

### Natural Language Processing

The voice system understands natural variations of commands:

#### Synonyms and Variations
- **"Buy"** = "Purchase", "Get", "Acquire", "Add"
- **"Sell"** = "Dispose", "Exit", "Close", "Liquidate"
- **"Show"** = "Display", "Tell me", "What is", "Give me"
- **"Portfolio"** = "Holdings", "Positions", "Investments"

#### Context Awareness
```
User: "Show my Apple position"
System: "You own 100 shares of AAPL worth $15,230"

User: "Sell half of it"  # System remembers "it" = Apple
System: "Selling 50 shares of AAPL at market price"

User: "What's the profit?"  # System knows context
System: "Your AAPL position has $1,230 unrealized gain"
```

### Multi-Step Commands
| Command | System Response |
|---------|-----------------|
| "If Apple drops to $140, sell all my shares and buy Tesla" | Creates conditional order |
| "Show me tech stocks under $100 with good momentum" | Complex market scan |
| "Calculate how much I need to invest to make $10,000" | Investment calculation |

### Voice Macros

Create custom voice commands:

```python
# Custom voice macro example
{
    "command": "morning routine",
    "actions": [
        "show portfolio status",
        "check overnight news",
        "review pending orders",
        "show market sentiment"
    ]
}
```

### Conversation Mode

Enable continuous conversation:

```
User: "Enable conversation mode"
System: "Conversation mode enabled. I'm listening..."

User: "What's my Tesla position?"
System: "You own 50 shares worth $12,500"

User: "What's the profit on that?"
System: "Tesla position shows $625 unrealized gain"

User: "Should I sell?"
System: "Based on momentum indicators, consider holding. Current signal is neutral."
```

---

## 🔧 Troubleshooting

### Common Issues

#### Voice Not Recognized
**Problem**: Commands not being understood
**Solutions**:
- "Calibrate microphone"
- "Increase voice sensitivity"
- "Test microphone"
- Speak clearly and at normal pace
- Reduce background noise

#### No Audio Response
**Problem**: System not speaking back
**Solutions**:
- "Enable voice responses"
- "Check audio output"
- "Increase volume"
- "Test speakers"

#### Wrong Commands Executed
**Problem**: System misinterprets commands
**Solutions**:
- "Show last command"
- "Undo last action"
- Use more specific language
- Pause between words

### Voice System Diagnostics

#### Test Commands
| Command | Purpose |
|---------|---------|
| "Test microphone" | Audio input test |
| "Test speakers" | Audio output test |
| "Voice system status" | Complete diagnostics |
| "Show recognition accuracy" | Speech recognition metrics |
| "Calibrate voice system" | Full recalibration |

#### Debug Mode
```
User: "Enable voice debug mode"
System: "Debug mode enabled. Showing recognition confidence scores."

User: "Show my portfolio"
System: [Confidence: 95%] "Displaying portfolio overview"
```

### Performance Optimization

#### Improve Recognition Accuracy
1. **Training**: "Train voice recognition" - Improves accuracy for your voice
2. **Background**: Reduce background noise
3. **Distance**: Stay 1-2 feet from microphone
4. **Speed**: Speak at normal conversational pace

#### Reduce Latency
- Use local processing: "Enable offline mode"
- Reduce command complexity
- Clear audio cache: "Clear voice cache"

---

## 🎨 Customization

### Custom Wake Words

```python
# Add custom wake words
CUSTOM_WAKE_WORDS = [
    "hey trader",
    "computer trade", 
    "trading assistant"
]
```

### Personalized Responses

```python
# Customize system responses
RESPONSE_STYLE = {
    "formal": "Your portfolio value is $105,250",
    "casual": "You've got $105K in your portfolio",
    "brief": "$105,250 portfolio value"
}
```

### Voice Shortcuts

Create personalized shortcuts:

| Your Command | System Action |
|--------------|---------------|
| "My usual" | Execute your most common trading strategy |
| "Emergency exit" | Close all positions immediately |
| "Check the big three" | Show AAPL, GOOGL, MSFT status |
| "Power hour" | Enable aggressive trading mode |

### Language Localization

Support for multiple languages:

```python
SUPPORTED_LANGUAGES = {
    'en-US': 'English (United States)',
    'en-GB': 'English (United Kingdom)', 
    'es-ES': 'Spanish (Spain)',
    'fr-FR': 'French (France)',
    'de-DE': 'German (Germany)',
    'ja-JP': 'Japanese (Japan)',
    'zh-CN': 'Chinese (Simplified)'
}
```

### Voice Themes

Different voice personalities:

- **Professional**: Formal, detailed responses
- **Casual**: Friendly, conversational tone
- **Brief**: Concise, minimal responses
- **Educational**: Explains reasoning behind responses

---

## 📱 Mobile Voice Commands

### Mobile-Specific Commands
| Command | Action |
|---------|--------|
| "Read my portfolio out loud" | Audio portfolio summary |
| "Text me when Apple hits $150" | SMS alert |
| "Call my broker" | Phone integration |
| "Share my performance" | Social sharing |
| "Take a screenshot" | Capture current screen |

### Driving Mode
Safe commands for hands-free use:

| Command | Response |
|---------|----------|
| "Driving mode on" | Enables hands-free operation |
| "Just the essentials" | Provides only critical information |
| "Any urgent alerts?" | Priority notifications only |
| "Portfolio summary" | Brief audio update |

---

## 🚨 Emergency Commands

### Immediate Actions
| Command | Action |
|---------|--------|
| "Emergency stop" | Halt all trading immediately |
| "Close everything" | Liquidate all positions |
| "Cancel all orders" | Cancel pending orders |
| "Safe mode" | Enable protective settings |
| "Lock account" | Disable trading until unlocked |

### Crisis Management
| Command | Action |
|---------|--------|
| "Market crash protocol" | Execute predefined crisis response |
| "Risk off" | Switch to defensive positioning |
| "Hedge everything" | Add protective hedges |
| "Call emergency contact" | Contact designated person |

---

## 📊 Voice Analytics

### Usage Statistics
Monitor your voice command usage:

- Most used commands
- Recognition accuracy rates
- Response times
- Error frequencies

### Command History
| Command | Access Method |
|---------|---------------|
| "Show voice history" | Last 50 voice commands |
| "Voice command stats" | Usage analytics |
| "Export voice data" | Download usage data |
| "Clear voice history" | Reset command history |

---

## 🔮 Future Voice Features

### Upcoming Capabilities
- **Predictive Commands**: "I think you want to check Apple"
- **Emotional Recognition**: Detect stress in voice and adjust recommendations
- **Multi-Language**: Real-time language switching
- **Voice Biometrics**: Voice-based authentication
- **AI Conversations**: Full natural language discussions about trading

### Beta Features
Enable experimental features:
```
User: "Enable beta voice features"
System: "Beta features enabled. You now have access to experimental commands."
```

---

## 📚 Voice Command Cheat Sheet

### Quick Reference Card

```
WAKE WORDS:          "Hey NeuroCluster" / "Hello NeuroCluster"

PORTFOLIO:           "Show portfolio" / "Portfolio status"
POSITIONS:           "Show [stock] position" / "List positions"
TRADING:             "Buy/Sell [amount] [stock]"
QUOTES:              "What's [stock] trading at?"
ALERTS:              "Alert me when [stock] hits [price]"
ORDERS:              "Show pending orders" / "Cancel orders"
RISK:                "What's my portfolio risk?"
NEWS:                "Any news on [stock]?"
HELP:                "Help" / "What can I do?"
EMERGENCY:           "Emergency stop" / "Close everything"

EXAMPLES:
✓ "Buy 100 shares of Apple"
✓ "What's Tesla trading at?"
✓ "Show my Google position"
✓ "Alert me when Bitcoin hits 50000"
✓ "How did I do today?"
✓ "Set stop loss on Apple at 140"
```

---

## 📞 Voice Support

### Getting Help
- **Voice Help**: Say "Help with voice commands"
- **Documentation**: https://docs.neurocluster-elite.com/voice
- **Video Tutorials**: https://youtube.com/neurocluster-voice
- **Community**: https://discord.gg/neurocluster-elite

### Feedback
Help improve voice recognition:
- **Report Issues**: "Report voice problem"
- **Suggest Commands**: "I want to suggest a command"
- **Rate Experience**: "Rate voice experience"

---

**Voice Commands Reference Last Updated:** June 30, 2025  
**Version:** 1.0.0  
**Supported Languages:** English (US/UK), Spanish, French, German, Japanese, Chinese  
**Next Update:** July 30, 2025

---

*"The future of trading is voice-controlled. Trade at the speed of thought."*