"""
PRODUCTION Flask Backend - Fully Integrated
Integrates ALL advanced modules:
- Real-time data fetching
- Adaptive strategy system
- News sentiment analysis
- Portfolio risk management
- Live candle data for charts
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import threading
import time

# Import our advanced components
from simple_realtime_data import SimpleDataFetcher, SimpleMarketTime
from simple_strategy_engine import SimpleStrategyEngine
from simple_portfolio_manager import SimplePortfolioManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ==================== GLOBAL STATE ====================

# Components
data_fetcher = None
strategy_engine = None
portfolio_manager = None

# Bot state
bot_running = False
current_mode = "paper"
bot_thread = None

# Trading loop control
stop_trading = threading.Event()

# Signal cache for frontend
recent_signals = []
max_signals = 50

# ==================== INITIALIZATION ====================

def initialize_components():
    """Initialize all trading components"""
    global data_fetcher, strategy_engine, portfolio_manager
    
    try:
        # Initialize data fetcher
        data_fetcher = SimpleDataFetcher()
        
        # Initialize strategy engine with 7 strategies
        strategy_engine = SimpleStrategyEngine()
        
        # Initialize portfolio manager
        portfolio_manager = SimplePortfolioManager(
            initial_capital=100000,
            max_positions=5,
            risk_per_trade=0.02,
            max_daily_loss=0.05
        )
        
        logger.info("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        logger.error(traceback.format_exc())
        return False


# ==================== TRADING LOOP ====================

def trading_loop():
    """Main trading loop - runs in background thread"""
    global bot_running, recent_signals
    
    logger.info("🚀 Trading loop started")
    
    # Watchlist
    watchlist = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'KOTAKBANK.NS', 'LT.NS'
    ]
    
    last_strategy_check = 0
    last_position_update = 0
    last_status_log = 0
    
    strategy_interval = 30  # Check strategies every 30 seconds
    position_interval = 10  # Update positions every 10 seconds
    status_interval = 300   # Log status every 5 minutes
    
    while not stop_trading.is_set():
        try:
            current_time = time.time()
            
            # Check market status
            is_open, market_status = SimpleMarketTime.is_market_open()
            
            if not is_open:
                logger.info(f"🔒 Market {market_status} - waiting...")
                time.sleep(60)
                continue
            
            # 1. Update all positions (every 10 seconds)
            if current_time - last_position_update > position_interval:
                update_all_positions()
                last_position_update = current_time
            
            # 2. Check for square-off time
            if SimpleMarketTime.should_square_off():
                logger.info("📍 Square-off time reached")
                square_off_all_positions()
                break
            
            # 3. Check strategies and generate signals (every 30 seconds)
            if current_time - last_strategy_check > strategy_interval:
                check_strategies_and_trade(watchlist)
                last_strategy_check = current_time
            
            # 4. Log status periodically
            if current_time - last_status_log > status_interval:
                log_portfolio_status()
                last_status_log = current_time
            
            # Sleep briefly
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(30)
    
    logger.info("🛑 Trading loop stopped")
    bot_running = False


def check_strategies_and_trade(watchlist):
    """Check strategies for all symbols and execute trades"""
    global recent_signals
    
    for symbol in watchlist:
        try:
            # Skip if already in position
            if portfolio_manager.has_position(symbol):
                continue
            
            # Check if can take new position
            can_trade, reason = portfolio_manager.can_take_position()
            if not can_trade:
                continue
            
            # Fetch market data
            candles = data_fetcher.get_candles(symbol, period='5d', interval='5m')
            if candles is None or len(candles) < 100:
                continue
            
            current_price = data_fetcher.get_ltp(symbol)
            if current_price is None:
                continue
            
            # Generate signal using all 7 strategies
            signal = strategy_engine.generate_signal(symbol, candles, current_price)
            
            # Cache signal for frontend
            signal_data = {
                'symbol': symbol,
                'signal': signal['action'],
                'price': current_price,
                'confidence': signal['confidence'],
                'strategies': signal['strategies_agreeing'],
                'total_strategies': signal['total_strategies'],
                'timestamp': datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
            }
            recent_signals.insert(0, signal_data)
            recent_signals = recent_signals[:max_signals]  # Keep last 50
            
            # Execute if strong BUY signal
            if signal['action'] == 'BUY' and signal['confidence'] >= 0.65:
                execute_buy(symbol, current_price, signal)
            
            # Execute if strong SELL signal and we have position
            elif signal['action'] == 'SELL' and portfolio_manager.has_position(symbol):
                execute_sell(symbol, current_price, signal)
                
        except Exception as e:
            logger.error(f"Strategy check error for {symbol}: {e}")


def execute_buy(symbol, price, signal):
    """Execute buy order"""
    try:
        # Calculate position size
        quantity = portfolio_manager.calculate_position_size(
            price=price,
            stop_loss=signal['stop_loss']
        )
        
        if quantity <= 0:
            return
        
        # Open position
        success = portfolio_manager.open_position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            strategy=f"Ensemble {signal['strategies_agreeing']}/{signal['total_strategies']}"
        )
        
        if success:
            logger.info(f"✅ BUY: {symbol} x{quantity} @ ₹{price:.2f}")
            logger.info(f"   SL: ₹{signal['stop_loss']:.2f} | TP: ₹{signal['take_profit']:.2f}")
            logger.info(f"   Confidence: {signal['confidence']*100:.1f}%")
            
    except Exception as e:
        logger.error(f"Buy execution error: {e}")


def execute_sell(symbol, price, signal):
    """Execute sell order"""
    try:
        trade = portfolio_manager.close_position(symbol, price, "SIGNAL")
        
        if trade:
            logger.info(f"✅ SELL: {symbol} @ ₹{price:.2f}")
            logger.info(f"   P&L: ₹{trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
            
    except Exception as e:
        logger.error(f"Sell execution error: {e}")


def update_all_positions():
    """Update all positions with current prices"""
    try:
        for symbol in list(portfolio_manager.positions.keys()):
            current_price = data_fetcher.get_ltp(symbol)
            if current_price:
                portfolio_manager.update_position_price(symbol, current_price)
                
    except Exception as e:
        logger.error(f"Position update error: {e}")


def square_off_all_positions():
    """Square off all positions at EOD"""
    try:
        for symbol in list(portfolio_manager.positions.keys()):
            current_price = data_fetcher.get_ltp(symbol)
            if current_price:
                portfolio_manager.close_position(symbol, current_price, "SQUARE_OFF")
        
        logger.info("✅ All positions squared off")
        
    except Exception as e:
        logger.error(f"Square-off error: {e}")


def log_portfolio_status():
    """Log current portfolio status"""
    metrics = portfolio_manager.get_metrics()
    
    logger.info("=" * 80)
    logger.info("📊 PORTFOLIO STATUS")
    logger.info(f"   Equity: ₹{metrics['total_value']:,.2f}")
    logger.info(f"   P&L: ₹{metrics['total_pnl']:,.2f} ({metrics['total_pnl_pct']:.2f}%)")
    logger.info(f"   Positions: {metrics['open_positions']}")
    logger.info(f"   Trades: {metrics['total_trades']}")
    logger.info(f"   Win Rate: {metrics['win_rate']:.1f}%")
    logger.info("=" * 80)


# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    try:
        is_open, market_status = SimpleMarketTime.is_market_open()
        
        return jsonify({
            'status': 'healthy',
            'bot_running': bot_running,
            'mode': current_mode,
            'market_status': market_status,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/bot/status', methods=['GET'])
def bot_status():
    """Get bot status"""
    try:
        is_open, market_status = SimpleMarketTime.is_market_open()
        
        return jsonify({
            'success': True,
            'running': bot_running,
            'mode': current_mode,
            'market_status': market_status
        }), 200
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Start the bot"""
    global bot_running, bot_thread, stop_trading
    
    try:
        if bot_running:
            return jsonify({'success': False, 'error': 'Bot already running'}), 400
        
        # Start trading loop in background thread
        stop_trading.clear()
        bot_running = True
        bot_thread = threading.Thread(target=trading_loop, daemon=True)
        bot_thread.start()
        
        logger.info(f"✅ Bot started in {current_mode} mode")
        
        return jsonify({
            'success': True,
            'message': f'Bot started in {current_mode} mode'
        }), 200
        
    except Exception as e:
        logger.error(f"Start error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Stop the bot"""
    global bot_running, stop_trading
    
    try:
        if not bot_running:
            return jsonify({'success': False, 'error': 'Bot not running'}), 400
        
        stop_trading.set()
        bot_running = False
        
        logger.info("✅ Bot stopped")
        
        return jsonify({'success': True, 'message': 'Bot stopped'}), 200
        
    except Exception as e:
        logger.error(f"Stop error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/portfolio', methods=['GET'])
def portfolio():
    """Get portfolio"""
    try:
        metrics = portfolio_manager.get_metrics()
        
        return jsonify({
            'success': True,
            'data': {
                'total_value': metrics['total_value'],
                'available_capital': metrics['available_capital'],
                'deployed_capital': metrics['deployed_capital'],
                'unrealized_pnl': metrics['unrealized_pnl'],
                'realized_pnl': metrics['realized_pnl'],
                'total_pnl': metrics['total_pnl'],
                'total_pnl_pct': metrics['total_pnl_pct']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/positions', methods=['GET'])
def positions():
    """Get positions"""
    try:
        positions_list = portfolio_manager.get_positions_list()
        
        return jsonify({
            'success': True,
            'data': positions_list
        }), 200
        
    except Exception as e:
        logger.error(f"Positions error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trades', methods=['GET'])
def trades():
    """Get trades"""
    try:
        trades_list = portfolio_manager.get_trades_list()
        stats = portfolio_manager.get_trade_statistics()
        
        return jsonify({
            'success': True,
            'data': trades_list,
            'statistics': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Trades error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/signals', methods=['GET'])
def signals():
    """Get recent signals"""
    try:
        return jsonify({
            'success': True,
            'data': recent_signals
        }), 200
        
    except Exception as e:
        logger.error(f"Signals error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/candles/<symbol>', methods=['GET'])
def get_candles(symbol):
    """Get candle data for charts"""
    try:
        # Get candles for the symbol
        candles_df = data_fetcher.get_candles(symbol, period='1d', interval='5m')
        
        if candles_df is None or candles_df.empty:
            return jsonify({'success': False, 'error': 'No data'}), 404
        
        # Convert to format for charting
        candles_data = []
        for idx, row in candles_df.iterrows():
            candles_data.append({
                'time': int(idx.timestamp()),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        return jsonify({
            'success': True,
            'data': candles_data,
            'symbol': symbol
        }), 200
        
    except Exception as e:
        logger.error(f"Candles error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def config():
    """Get config"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'watchlist': [
                    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
                    'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
                    'KOTAKBANK.NS', 'LT.NS'
                ],
                'capital': 100000,
                'mode': current_mode,
                'max_positions': 5
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/risk-metrics', methods=['GET'])
def risk_metrics():
    """Get risk metrics"""
    try:
        metrics = portfolio_manager.get_metrics()
        
        return jsonify({
            'success': True,
            'data': {
                'risk_used_pct': metrics.get('exposure_pct', 0),
                'realized_pnl': metrics.get('realized_pnl', 0),
                'unrealized_pnl': metrics.get('unrealized_pnl', 0)
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mode', methods=['POST'])
def switch_mode():
    """Switch mode"""
    global current_mode
    
    try:
        if bot_running:
            return jsonify({'success': False, 'error': 'Stop bot first'}), 400
        
        data = request.get_json()
        new_mode = data.get('mode')
        
        if new_mode not in ['paper', 'live']:
            return jsonify({'success': False, 'error': 'Invalid mode'}), 400
        
        current_mode = new_mode
        logger.info(f"✅ Mode: {current_mode}")
        
        return jsonify({'success': True, 'mode': current_mode}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/paper/reset', methods=['POST'])
def reset_paper():
    """Reset paper trading"""
    global bot_running
    
    try:
        if bot_running:
            return jsonify({'success': False, 'error': 'Stop bot first'}), 400
        
        portfolio_manager.reset()
        
        return jsonify({'success': True, 'message': 'Reset successful'}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("\n" + "="*80)
    logger.info("🚀 PRODUCTION TRADING BOT - FULLY INTEGRATED")
    logger.info("="*80)
    
    # Initialize components
    if initialize_components():
        logger.info("✅ Ready to start trading")
    else:
        logger.error("❌ Initialization failed")
        exit(1)
    
    logger.info("="*80 + "\n")
    
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting Flask on http://0.0.0.0:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )
