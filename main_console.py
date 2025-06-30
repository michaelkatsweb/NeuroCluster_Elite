#!/usr/bin/env python3
"""
File: main_console.py
Path: NeuroCluster-Elite/main_console.py
Description: Console interface entry point for NeuroCluster Elite

This is the main command-line interface for the NeuroCluster Elite trading platform.
It provides various modes of operation including demo, live trading, backtesting,
and system management.

Usage:
    python main_console.py --help
    neurocluster --help
    neurocluster --demo
    neurocluster --continuous --symbols AAPL,GOOGL,MSFT
    neurocluster --backtest --start 2023-01-01 --end 2023-12-31
    neurocluster --paper-trading --capital 100000
    neurocluster --live-trading --broker alpaca

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import asyncio
import argparse
import sys
import os
import signal
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/neurocluster_console.log')
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from src.core.neurocluster_elite import NeuroClusterElite, RegimeType, AssetType
    from src.data.multi_asset_manager import MultiAssetDataManager
    from src.trading.trading_engine import AdvancedTradingEngine
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import setup_logging
    from src.utils.helpers import print_banner, print_table, format_currency, format_percentage
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

# ==================== GLOBAL STATE ====================

class ConsoleState:
    """Global state for console application"""
    def __init__(self):
        self.running = False
        self.trading_engine = None
        self.config_manager = None
        self.start_time = None
        self.interrupt_count = 0

console_state = ConsoleState()

# ==================== SIGNAL HANDLERS ====================

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    console_state.interrupt_count += 1
    
    if console_state.interrupt_count == 1:
        print("\n🛑 Received shutdown signal. Stopping gracefully...")
        console_state.running = False
        
        if console_state.trading_engine:
            console_state.trading_engine.stop_trading()
    else:
        print("\n🚨 Force shutdown requested. Exiting immediately!")
        sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ==================== CONSOLE INTERFACE ====================

class NeuroClusterConsole:
    """Main console interface for NeuroCluster Elite"""
    
    def __init__(self, args):
        self.args = args
        self.config_manager = ConfigManager()
        self.trading_engine = None
        self.data_manager = None
        self.neurocluster = None
        
        # Setup logging based on arguments
        log_level = logging.DEBUG if args.verbose else logging.INFO
        setup_logging(log_level, args.log_file)
        
        logger.info("🚀 NeuroCluster Elite Console Interface Starting")
    
    async def run(self):
        """Main entry point for console application"""
        
        console_state.running = True
        console_state.start_time = datetime.now()
        
        try:
            # Print banner
            self.print_startup_banner()
            
            # Load configuration
            await self.load_configuration()
            
            # Initialize components
            await self.initialize_components()
            
            # Run the requested mode
            if self.args.mode == 'demo':
                await self.run_demo_mode()
            elif self.args.mode == 'continuous':
                await self.run_continuous_mode()
            elif self.args.mode == 'backtest':
                await self.run_backtest_mode()
            elif self.args.mode == 'status':
                await self.show_status()
            elif self.args.mode == 'test':
                await self.run_tests()
            elif self.args.mode == 'setup':
                await self.run_setup_wizard()
            else:
                await self.run_interactive_mode()
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            await self.cleanup()
    
    def print_startup_banner(self):
        """Print startup banner with system information"""
        
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🚀 NEUROCLUSTER ELITE v1.0.0                          ║
║                Ultimate AI-Powered Multi-Asset Trading Platform             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • 99.59% Algorithm Efficiency  • 0.045ms Processing Time                   ║
║  • Multi-Asset Support         • Real-time Regime Detection               ║
║  • Advanced Risk Management    • Voice Commands Support                   ║
║  • Professional Analytics     • Enterprise Security                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        
        print(banner)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💻 Mode: {self.args.mode.upper()}")
        print(f"📊 Paper Trading: {'Enabled' if not self.args.live_trading else 'Disabled'}")
        print(f"🔧 Config: {self.args.config}")
        print("=" * 80)
    
    async def load_configuration(self):
        """Load configuration from files and command line"""
        
        print("📋 Loading configuration...")
        
        # Load base configuration
        config = self.config_manager.load_config(self.args.config)
        
        # Apply command line overrides
        if self.args.symbols:
            symbols = self.args.symbols.split(',')
            config['symbols'] = {'stocks': symbols}
        
        if self.args.capital:
            config['trading']['initial_capital'] = self.args.capital
        
        if self.args.live_trading:
            config['trading']['paper_trading'] = False
        
        if self.args.risk_level:
            risk_levels = {
                'conservative': 0.01,
                'moderate': 0.02,
                'aggressive': 0.05
            }
            config['risk']['max_portfolio_risk'] = risk_levels.get(self.args.risk_level, 0.02)
        
        self.config = config
        print("✅ Configuration loaded successfully")
    
    async def initialize_components(self):
        """Initialize core components"""
        
        print("🔧 Initializing components...")
        
        # Initialize NeuroCluster algorithm
        self.neurocluster = NeuroClusterElite(self.config.get('algorithm', {}))
        print("✅ NeuroCluster algorithm initialized")
        
        # Initialize data manager
        self.data_manager = MultiAssetDataManager(self.config.get('data', {}))
        print("✅ Multi-asset data manager initialized")
        
        # Initialize trading engine if needed
        if self.args.mode in ['continuous', 'backtest', 'demo']:
            self.trading_engine = AdvancedTradingEngine(self.config)
            console_state.trading_engine = self.trading_engine
            print("✅ Trading engine initialized")
        
        print("🎉 All components ready!")
    
    async def run_demo_mode(self):
        """Run demonstration mode with simulated data"""
        
        print("\n🎬 Starting Demo Mode")
        print("=" * 50)
        print("This demo shows NeuroCluster Elite capabilities using simulated market data.")
        print("Perfect for learning the system without real market exposure.\n")
        
        demo_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'BTC-USD', 'ETH-USD']
        cycle_count = 0
        
        while console_state.running and cycle_count < (self.args.demo_cycles or 10):
            try:
                print(f"📊 Demo Cycle {cycle_count + 1}")
                print("-" * 30)
                
                # Generate simulated market data
                market_data = self.generate_demo_data(demo_symbols)
                
                # Show market data
                self.display_market_data(market_data)
                
                # Extract features and detect regime
                features = self.neurocluster.extract_enhanced_features(market_data)
                regime, confidence = self.neurocluster.detect_regime(features)
                
                # Display regime detection
                self.display_regime_info(regime, confidence, features)
                
                # Show algorithm performance
                self.display_performance_metrics()
                
                cycle_count += 1
                
                if cycle_count < (self.args.demo_cycles or 10):
                    print(f"\n⏱️  Next update in {self.args.interval or 5} seconds...")
                    await asyncio.sleep(self.args.interval or 5)
                
            except Exception as e:
                logger.error(f"Error in demo cycle: {e}")
                break
        
        print("\n🎬 Demo completed!")
        self.display_final_summary()
    
    async def run_continuous_mode(self):
        """Run continuous trading mode"""
        
        print("\n🔄 Starting Continuous Trading Mode")
        print("=" * 50)
        print("Real-time market analysis and trading with live data feeds.")
        print("Press Ctrl+C to stop gracefully.\n")
        
        if self.args.live_trading:
            print("⚠️  LIVE TRADING MODE ENABLED - Real money at risk!")
            confirmation = input("Type 'CONFIRM' to proceed with live trading: ")
            if confirmation != 'CONFIRM':
                print("❌ Live trading cancelled. Use --paper-trading for safe mode.")
                return
        
        # Start trading engine
        trading_task = asyncio.create_task(self.trading_engine.start_trading())
        
        # Status update task
        status_task = asyncio.create_task(self.continuous_status_updates())
        
        try:
            # Run until interrupted
            await asyncio.gather(trading_task, status_task)
        except asyncio.CancelledError:
            print("📋 Continuous mode stopped")
    
    async def continuous_status_updates(self):
        """Provide periodic status updates in continuous mode"""
        
        update_interval = 30  # Update every 30 seconds
        
        while console_state.running:
            try:
                await asyncio.sleep(update_interval)
                
                if self.trading_engine:
                    status = self.trading_engine.get_status()
                    self.display_trading_status(status)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in status updates: {e}")
    
    async def run_backtest_mode(self):
        """Run backtesting mode"""
        
        print("\n📈 Starting Backtest Mode")
        print("=" * 50)
        
        start_date = self.args.start_date or '2023-01-01'
        end_date = self.args.end_date or '2023-12-31'
        
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💰 Initial Capital: ${self.args.capital or 100000:,}")
        print(f"📊 Symbols: {self.args.symbols or 'Default portfolio'}")
        
        # TODO: Implement backtesting
        print("\n🚧 Backtesting implementation coming soon!")
        print("For now, use demo mode to see algorithm capabilities.")
    
    async def show_status(self):
        """Show current system status"""
        
        print("\n📊 NeuroCluster Elite System Status")
        print("=" * 50)
        
        # System information
        uptime = datetime.now() - console_state.start_time if console_state.start_time else timedelta(0)
        
        status_data = [
            ["System Status", "🟢 Online"],
            ["Uptime", str(uptime)],
            ["Version", "1.0.0"],
            ["Mode", self.args.mode.title()],
            ["Paper Trading", "Yes" if not self.args.live_trading else "No"],
        ]
        
        print_table(status_data, headers=["Component", "Status"])
        
        # Component status
        if self.neurocluster:
            perf = self.neurocluster.get_performance_summary()
            
            print(f"\n🧠 NeuroCluster Algorithm:")
            print(f"   Efficiency: {perf['efficiency_rate']:.2f}%")
            print(f"   Processing Time: {perf['avg_processing_time_ms']:.3f}ms")
            print(f"   Total Processed: {perf['total_processed']:,}")
            print(f"   Active Clusters: {perf['cluster_count']}")
        
        if self.trading_engine:
            engine_status = self.trading_engine.get_status()
            
            print(f"\n💰 Trading Engine:")
            print(f"   Portfolio Value: {format_currency(engine_status['portfolio_value'])}")
            print(f"   Cash Balance: {format_currency(engine_status['cash_balance'])}")
            print(f"   Active Positions: {engine_status['num_positions']}")
            print(f"   Total Trades: {engine_status['performance_metrics']['total_trades']}")
    
    async def run_tests(self):
        """Run system tests"""
        
        print("\n🧪 Running NeuroCluster Elite Tests")
        print("=" * 50)
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Algorithm performance
        print("🔬 Test 1: Algorithm Performance")
        try:
            from src.core.neurocluster_elite import run_performance_test
            
            test_results = run_performance_test(self.neurocluster, 500)
            
            if (test_results['efficiency_rate'] >= 99.0 and 
                test_results['avg_processing_time_ms'] <= 0.1):
                print("✅ Algorithm performance test PASSED")
                tests_passed += 1
            else:
                print("❌ Algorithm performance test FAILED")
            
            tests_total += 1
            
        except Exception as e:
            print(f"❌ Algorithm test failed with error: {e}")
            tests_total += 1
        
        # Test 2: Data fetching
        print("\n🔬 Test 2: Data Fetching")
        try:
            test_symbols = ['AAPL', 'GOOGL']
            data = await self.data_manager.fetch_market_data(test_symbols, AssetType.STOCK)
            
            if len(data) > 0:
                print("✅ Data fetching test PASSED")
                tests_passed += 1
            else:
                print("❌ Data fetching test FAILED - No data returned")
            
            tests_total += 1
            
        except Exception as e:
            print(f"❌ Data fetching test failed: {e}")
            tests_total += 1
        
        # Test 3: Trading engine initialization
        print("\n🔬 Test 3: Trading Engine")
        try:
            if self.trading_engine:
                status = self.trading_engine.get_status()
                
                if status['portfolio_value'] > 0:
                    print("✅ Trading engine test PASSED")
                    tests_passed += 1
                else:
                    print("❌ Trading engine test FAILED")
            else:
                print("❌ Trading engine not initialized")
            
            tests_total += 1
            
        except Exception as e:
            print(f"❌ Trading engine test failed: {e}")
            tests_total += 1
        
        # Summary
        print(f"\n📊 Test Results: {tests_passed}/{tests_total} tests passed")
        
        if tests_passed == tests_total:
            print("🎉 All tests passed! System is ready for trading.")
        else:
            print("⚠️  Some tests failed. Check configuration and API keys.")
    
    async def run_setup_wizard(self):
        """Run interactive setup wizard"""
        
        print("\n🧙 NeuroCluster Elite Setup Wizard")
        print("=" * 50)
        print("This wizard will help you configure NeuroCluster Elite for first use.\n")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print("❌ Python 3.8 or higher required")
            return
        else:
            print(f"✅ Python {python_version.major}.{python_version.minor} detected")
        
        # Check dependencies
        print("\n📦 Checking dependencies...")
        
        required_packages = [
            'streamlit', 'plotly', 'pandas', 'numpy', 'scikit-learn',
            'yfinance', 'aiohttp', 'ta', 'python-dotenv'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} - Missing")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Run: pip install -r requirements.txt")
            return
        
        # Create configuration
        print("\n⚙️  Setting up configuration...")
        
        # Get user preferences
        capital = input("💰 Initial capital (default: 100000): ") or "100000"
        risk_level = input("⚖️  Risk level (conservative/moderate/aggressive, default: moderate): ") or "moderate"
        symbols = input("📊 Symbols to track (comma-separated, default: AAPL,GOOGL,MSFT): ") or "AAPL,GOOGL,MSFT"
        
        # Create .env file
        env_file = Path(".env")
        if not env_file.exists():
            print("📝 Creating .env configuration file...")
            
            env_content = f"""# NeuroCluster Elite Configuration
# Generated by setup wizard on {datetime.now().isoformat()}

# Basic Settings
PAPER_TRADING=true
INITIAL_CAPITAL={capital}
DEFAULT_STOCKS={symbols}

# Risk Management
RISK_LEVEL={risk_level}

# Add your API keys here:
# ALPHA_VANTAGE_API_KEY=your_key_here
# POLYGON_API_KEY=your_key_here

# Notification settings (optional):
# DISCORD_WEBHOOK_URL=your_webhook_here
# TELEGRAM_BOT_TOKEN=your_token_here
"""
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            print("✅ Configuration file created: .env")
        
        # Create directories
        directories = ['data', 'logs', 'config', 'data/cache', 'data/exports']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file and add your API keys")
        print("2. Run: neurocluster --demo (to test the system)")
        print("3. Run: neurocluster-web (to launch web interface)")
        print("4. Run: neurocluster --continuous (for live trading)")
    
    async def run_interactive_mode(self):
        """Run interactive command mode"""
        
        print("\n💬 Interactive Mode")
        print("=" * 50)
        print("Type 'help' for available commands, 'quit' to exit.\n")
        
        while console_state.running:
            try:
                command = input("neurocluster> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                elif command == 'help':
                    self.show_interactive_help()
                elif command == 'status':
                    await self.show_status()
                elif command.startswith('demo'):
                    await self.run_demo_mode()
                elif command == 'test':
                    await self.run_tests()
                elif command == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands.")
                
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit gracefully.")
    
    def show_interactive_help(self):
        """Show help for interactive mode"""
        
        commands = [
            ["help", "Show this help message"],
            ["status", "Show system status"],
            ["demo", "Run demonstration mode"],
            ["test", "Run system tests"],
            ["clear", "Clear screen"],
            ["quit", "Exit interactive mode"],
        ]
        
        print("\n📚 Available Commands:")
        print_table(commands, headers=["Command", "Description"])
    
    def generate_demo_data(self, symbols: List[str]) -> Dict:
        """Generate realistic demo market data"""
        
        import random
        
        market_data = {}
        
        for symbol in symbols:
            # Determine asset type
            if symbol.endswith('-USD'):
                asset_type = AssetType.CRYPTO
                base_price = random.uniform(100, 50000)
                volatility = random.uniform(2, 8)
            else:
                asset_type = AssetType.STOCK
                base_price = random.uniform(50, 500)
                volatility = random.uniform(0.5, 3)
            
            # Generate realistic price movement
            change_pct = random.gauss(0, volatility)
            change = base_price * (change_pct / 100)
            
            from src.core.neurocluster_elite import MarketData
            
            market_data[symbol] = MarketData(
                symbol=symbol,
                asset_type=asset_type,
                price=base_price + change,
                change=change,
                change_percent=change_pct,
                volume=random.uniform(100000, 10000000),
                timestamp=datetime.now(),
                rsi=random.uniform(20, 80),
                volatility=volatility,
                liquidity_score=random.uniform(0.3, 1.0),
                sentiment_score=random.uniform(-0.5, 0.5)
            )
        
        return market_data
    
    def display_market_data(self, market_data: Dict):
        """Display market data in a formatted table"""
        
        data = []
        for symbol, md in market_data.items():
            emoji = "📈" if md.change >= 0 else "📉"
            data.append([
                f"{emoji} {symbol}",
                f"${md.price:.2f}",
                f"{md.change:+.2f}",
                f"{md.change_percent:+.2f}%",
                f"{md.volume:,.0f}",
                f"{md.rsi:.1f}" if md.rsi else "N/A"
            ])
        
        print_table(data, headers=["Symbol", "Price", "Change", "Change %", "Volume", "RSI"])
    
    def display_regime_info(self, regime: RegimeType, confidence: float, features):
        """Display regime detection information"""
        
        print(f"\n🧠 Market Regime Analysis:")
        print(f"   Current Regime: {regime.value}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Algorithm Processing: ⚡ Ultra-fast (target: 0.045ms)")
        
        # Show key features
        feature_names = [
            "Price Momentum", "Momentum Consistency", "Volatility", "Vol Clustering",
            "Volume", "Correlation", "Liquidity", "Sentiment"
        ]
        
        print(f"\n📊 Key Features:")
        for i, (name, value) in enumerate(zip(feature_names[:4], features[:4])):
            print(f"   {name}: {value:.3f}")
    
    def display_performance_metrics(self):
        """Display algorithm performance metrics"""
        
        if self.neurocluster:
            perf = self.neurocluster.get_performance_summary()
            
            metrics = [
                ["Efficiency Rate", f"{perf['efficiency_rate']:.2f}%"],
                ["Processing Time", f"{perf['avg_processing_time_ms']:.3f}ms"],
                ["Total Processed", f"{perf['total_processed']:,}"],
                ["Active Clusters", str(perf['cluster_count'])],
                ["Target Efficiency", f"{perf['target_efficiency']:.2f}%"],
                ["Target Time", f"{perf['target_processing_time_ms']:.3f}ms"]
            ]
            
            print(f"\n⚡ Algorithm Performance:")
            print_table(metrics, headers=["Metric", "Value"])
    
    def display_trading_status(self, status: Dict):
        """Display trading engine status"""
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n📊 Trading Status Update ({timestamp})")
        print("-" * 50)
        print(f"💰 Portfolio Value: {format_currency(status['portfolio_value'])}")
        print(f"💵 Cash Balance: {format_currency(status['cash_balance'])}")
        print(f"📈 Positions: {status['num_positions']}")
        
        perf = status['performance_metrics']
        print(f"📊 Total Trades: {perf['total_trades']}")
        print(f"📊 Win Rate: {format_percentage(perf['win_rate'])}")
        print(f"📊 Total Return: {format_percentage(perf['total_return'])}")
    
    def display_final_summary(self):
        """Display final summary"""
        
        runtime = datetime.now() - console_state.start_time if console_state.start_time else timedelta(0)
        
        print(f"\n📋 Session Summary:")
        print(f"   Runtime: {runtime}")
        print(f"   Mode: {self.args.mode.title()}")
        
        if self.neurocluster:
            perf = self.neurocluster.get_performance_summary()
            print(f"   Data Points Processed: {perf['total_processed']:,}")
            print(f"   Algorithm Efficiency: {perf['efficiency_rate']:.2f}%")
    
    async def cleanup(self):
        """Cleanup resources"""
        
        print("\n🧹 Cleaning up...")
        
        if self.trading_engine:
            self.trading_engine.stop_trading()
        
        if self.data_manager:
            # Clear cache if needed
            pass
        
        print("✅ Cleanup completed")

# ==================== ARGUMENT PARSER ====================

def create_argument_parser():
    """Create command line argument parser"""
    
    parser = argparse.ArgumentParser(
        prog='neurocluster',
        description='NeuroCluster Elite - Ultimate AI-Powered Multi-Asset Trading Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neurocluster --demo                           # Run demonstration mode
  neurocluster --continuous --symbols AAPL,GOOGL,MSFT  # Live trading mode
  neurocluster --backtest --start 2023-01-01 --end 2023-12-31  # Backtest
  neurocluster --status                         # Show system status
  neurocluster --test                           # Run system tests
  neurocluster --setup                          # Run setup wizard

For more information, visit: https://neurocluster-elite.readthedocs.io/
        """
    )
    
    # Operation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--demo', action='store_const', const='demo', dest='mode',
                           help='Run demonstration mode with simulated data')
    mode_group.add_argument('--continuous', action='store_const', const='continuous', dest='mode',
                           help='Run continuous trading mode')
    mode_group.add_argument('--backtest', action='store_const', const='backtest', dest='mode',
                           help='Run backtesting mode')
    mode_group.add_argument('--status', action='store_const', const='status', dest='mode',
                           help='Show system status and exit')
    mode_group.add_argument('--test', action='store_const', const='test', dest='mode',
                           help='Run system tests')
    mode_group.add_argument('--setup', action='store_const', const='setup', dest='mode',
                           help='Run interactive setup wizard')
    
    # Trading options
    trading_group = parser.add_argument_group('Trading Options')
    trading_group.add_argument('--live-trading', action='store_true',
                              help='Enable live trading (default: paper trading)')
    trading_group.add_argument('--paper-trading', action='store_true', default=True,
                              help='Enable paper trading (default)')
    trading_group.add_argument('--capital', type=float, default=100000,
                              help='Initial capital (default: 100000)')
    trading_group.add_argument('--risk-level', choices=['conservative', 'moderate', 'aggressive'],
                              default='moderate', help='Risk level (default: moderate)')
    
    # Market data options
    data_group = parser.add_argument_group('Market Data Options')
    data_group.add_argument('--symbols', type=str,
                           help='Comma-separated list of symbols to track')
    data_group.add_argument('--interval', type=int, default=10,
                           help='Update interval in seconds (default: 10)')
    
    # Backtesting options
    backtest_group = parser.add_argument_group('Backtesting Options')
    backtest_group.add_argument('--start-date', type=str,
                               help='Backtest start date (YYYY-MM-DD)')
    backtest_group.add_argument('--end-date', type=str,
                               help='Backtest end date (YYYY-MM-DD)')
    
    # Demo options
    demo_group = parser.add_argument_group('Demo Options')
    demo_group.add_argument('--demo-cycles', type=int, default=10,
                           help='Number of demo cycles to run (default: 10)')
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument('--config', type=str, default='config/default_config.yaml',
                             help='Configuration file path')
    config_group.add_argument('--log-file', type=str, default='logs/neurocluster_console.log',
                             help='Log file path')
    config_group.add_argument('--verbose', '-v', action='store_true',
                             help='Enable verbose logging')
    
    # Set default mode
    parser.set_defaults(mode='interactive')
    
    return parser

# ==================== MAIN ENTRY POINT ====================

async def main():
    """Main entry point for console application"""
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create and run console interface
    console = NeuroClusterConsole(args)
    await console.run()

def sync_main():
    """Synchronous wrapper for main function"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sync_main()