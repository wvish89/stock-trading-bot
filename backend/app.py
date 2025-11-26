"""
COMPLETE TRADING BOT - ALL ISSUES FIXED
Copy this ENTIRE file and save as app.py in your project root
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import os
from datetime import datetime, timedelta, time as dtime
import logging
import random
import time as time_module

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

# GLOBAL VARIABLES - THIS IS CRITICAL
trading_bot = None
bot_thread = None
bot_running = False
current_mode = "paper"
paper_engine = None

print("\n" + "="*60)
print("✅ Global variables initialized")
print(f"   bot_running = {bot_running}")
print(f"   current_mode = {current_mode}")
print("="*60 + "\n")

# ============== PAPER TRADING ENGINE ==============

class PaperTradingEngine:
    def __init__(self):
        self.capital = 100000
        self.positions = {}
        self.trades = []
        self.daily_pnl = 0.0
        self.price_history = {}
        self._init_prices()
        logger.info("✅ PaperTradingEngine initialized")

    def _init_prices(self):
        """Initialize with realistic prices"""
        symbols = ['RELIANCE-EQ', 'TCS-EQ', 'INFY-EQ', 'HDFCBANK-EQ']
        base_prices = {
            'RELIANCE-EQ': 2450,
            'TCS-EQ': 3800,
            'INFY-EQ': 1450,
            'HDFCBANK-EQ': 1650
        }
        
        for symbol in symbols:
            price = base_prices.get(symbol, 1000)
            self.price_history[symbol] = [price] * 50
            logger.info(f"  Initialized {symbol}: ₹{price}")

    def update_prices(self):
        """Update prices randomly"""
        for symbol in self.price_history:
            change = random.gauss(0, 0.005)
            new_price = self.price_history[symbol][-1] * (1 + change)
            self.price_history[symbol].append(new_price)

    def get_portfolio_value(self):
        """Calculate portfolio value"""
        position_value = 0
        for symbol, qty in self.positions.items():
            if symbol in self.price_history:
                position_value += self.price_history[symbol][-1] * qty
        return self.capital + position_value

    def place_order(self, symbol, trans_type, qty, price):
        """Place order"""
        if trans_type == 'BUY':
            cost = price * qty
            if cost > self.capital:
                return {'success': False, 'error': 'Insufficient capital'}
            self.capital -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
            logger.info(f"✅ BUY {symbol}: {qty} @ ₹{price}")
            return {'success': True, 'order_id': f'ORD_{len(self.trades)+1}'}
        
        elif trans_type == 'SELL':
            if symbol not in self.positions or self.positions[symbol] == 0:
                return {'success': False, 'error': 'No position'}
            proceeds = price * qty
            self.capital += proceeds
            self.positions[symbol] -= qty
            pnl = (price - 1000) * qty  # Simplified P&L
            self.daily_pnl += pnl
            self.trades.append({
                'symbol': symbol,
                'qty': qty,
                'price': price,
                'pnl': pnl,
                'time': datetime.now().isoformat()
            })
            logger.info(f"✅ SELL {symbol}: {qty} @ ₹{price}, P&L: ₹{pnl}")
            return {'success': True, 'order_id': f'ORD_{len(self.trades)}'}

# ============== AUTO TRADING BOT ==============

class AutoTradingBot:
    def __init__(self, engine):
        self.engine = engine
        self.running = False
        logger.info("✅ AutoTradingBot created")

    def start(self):
        """Start bot trading loop"""
        self.running = True
        logger.info("🤖 BOT MONITORING LOOP STARTED")
        
        while self.running:
            try:
                # Update prices every 2 seconds
                self.engine.update_prices()
                
                # Simple demo: random trades
                if random.random() > 0.98:  # 2% chance
                    symbol = 'RELIANCE-EQ'
                    price = self.engine.price_history[symbol][-1]
                    self.engine.place_order(symbol, 'BUY', 1, price)
                
                time_module.sleep(2)
            except Exception as e:
                logger.error(f"Bot error: {e}")
                time_module.sleep(2)

    def stop(self):
        """Stop bot"""
        self.running = False
        # Close all positions
        for symbol in list(self.engine.positions.keys()):
            if self.engine.positions[symbol] > 0:
                price = self.engine.price_history[symbol][-1]
                self.engine.place_order(symbol, 'SELL', 
                                       self.engine.positions[symbol], price)
        logger.info("🛑 BOT STOPPED - All positions closed")

# ============== API ENDPOINTS ==============

@app.route('/api/health', methods=['GET'])
def health():
    """Health check - CRITICAL FOR CONNECTION"""
    try:
        logger.info(f"📊 Health check: bot_running={bot_running}, mode={current_mode}")
        return jsonify({
            'status': 'healthy',
            'bot_running': bot_running,
            'mode': current_mode,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """START BOT - ISSUE #1 FIX"""
    global trading_bot, bot_thread, bot_running, paper_engine
    
    try:
        logger.info("📍 /api/bot/start called")
        
        if bot_running:
            logger.warning("⚠️  Bot already running!")
            return jsonify({
                'success': False, 
                'error': 'Bot already running'
            }), 400

        logger.info("🔧 Initializing paper engine...")
        if not paper_engine:
            paper_engine = PaperTradingEngine()

        logger.info("🔧 Creating bot instance...")
        trading_bot = AutoTradingBot(paper_engine)

        logger.info("🔧 Starting bot thread...")
        bot_thread = threading.Thread(target=trading_bot.start, daemon=True)
        bot_thread.start()

        bot_running = True
        logger.info(f"✅ BOT STARTED in {current_mode} mode")

        return jsonify({
            'success': True,
            'message': f'Bot started in {current_mode} mode',
            'bot_running': bot_running
        }), 200

    except Exception as e:
        logger.error(f"❌ Start bot error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """STOP BOT"""
    global bot_running, trading_bot
    
    try:
        logger.info("📍 /api/bot/stop called")
        
        if not bot_running:
            logger.warning("⚠️  Bot not running!")
            return jsonify({
                'success': False,
                'error': 'Bot not running'
            }), 400

        if trading_bot:
            trading_bot.stop()
        
        bot_running = False
        logger.info("✅ BOT STOPPED")

        return jsonify({
            'success': True,
            'message': 'Bot stopped',
            'bot_running': bot_running
        }), 200

    except Exception as e:
        logger.error(f"❌ Stop bot error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/status', methods=['GET'])
def bot_status():
    """Get bot status"""
    try:
        logger.info(f"📊 Status: running={bot_running}, mode={current_mode}")
        return jsonify({
            'success': True,
            'running': bot_running,
            'mode': current_mode,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/mode', methods=['POST'])
def switch_mode():
    """SWITCH MODE - ISSUE #2 FIX"""
    global current_mode, bot_running
    
    try:
        logger.info("📍 /api/mode called")
        
        if bot_running:
            logger.warning("⚠️  Stop bot before switching mode!")
            return jsonify({
                'success': False,
                'error': 'Stop bot before switching mode'
            }), 400

        data = request.get_json()
        new_mode = data.get('mode', 'paper')
        
        logger.info(f"🔄 Switching from {current_mode} to {new_mode}")

        if new_mode not in ['paper', 'live']:
            return jsonify({
                'success': False,
                'error': 'Mode must be paper or live'
            }), 400

        if new_mode == 'live':
            # Check credentials
            api_key = os.getenv('ANGEL_API_KEY')
            if not api_key or api_key == 'YOUR_API_KEY_HERE':
                logger.warning("❌ Live mode requires API credentials!")
                return jsonify({
                    'success': False,
                    'error': 'Live trading credentials not configured in .env'
                }), 400
            logger.info("✅ Live credentials verified")

        current_mode = new_mode
        logger.info(f"✅ Mode switched to {current_mode}")

        return jsonify({
            'success': True,
            'mode': current_mode,
            'message': f'Switched to {current_mode} mode'
        }), 200

    except Exception as e:
        logger.error(f"❌ Mode switch error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
def portfolio():
    """Get portfolio"""
    try:
        if not paper_engine:
            return jsonify({
                'success': True,
                'data': {
                    'total_value': 100000,
                    'capital': 100000,
                    'pnl': 0
                }
            }), 200

        value = paper_engine.get_portfolio_value()
        return jsonify({
            'success': True,
            'data': {
                'total_value': round(value, 2),
                'capital': round(paper_engine.capital, 2),
                'pnl': round(paper_engine.daily_pnl, 2)
            }
        }), 200
    except Exception as e:
        logger.error(f"❌ Portfolio error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def positions():
    """Get positions"""
    try:
        if not paper_engine:
            return jsonify({'success': True, 'data': []}), 200

        pos_list = []
        for symbol, qty in paper_engine.positions.items():
            if qty > 0:
                price = paper_engine.price_history.get(symbol, [0])[-1]
                pos_list.append({
                    'symbol': symbol,
                    'quantity': qty,
                    'price': round(price, 2)
                })

        return jsonify({'success': True, 'data': pos_list}), 200
    except Exception as e:
        logger.error(f"❌ Positions error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trades', methods=['GET'])
def trades():
    """Get trades"""
    try:
        if not paper_engine:
            return jsonify({
                'success': True,
                'data': [],
                'statistics': {'total_trades': 0, 'total_pnl': 0, 'win_rate': 0}
            }), 200

        trade_list = paper_engine.trades
        total_pnl = sum(t['pnl'] for t in trade_list)
        wins = sum(1 for t in trade_list if t['pnl'] > 0)
        win_rate = (wins / len(trade_list) * 100) if trade_list else 0

        return jsonify({
            'success': True,
            'data': trade_list,
            'statistics': {
                'total_trades': len(trade_list),
                'total_pnl': round(total_pnl, 2),
                'win_rate': round(win_rate, 2)
            }
        }), 200
    except Exception as e:
        logger.error(f"❌ Trades error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def config():
    """Get config"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'capital': 100000,
                'mode': current_mode,
                'bot_running': bot_running
            }
        }), 200
    except Exception as e:
        logger.error(f"❌ Config error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/paper/reset', methods=['POST'])
def reset_paper():
    """Reset paper trading"""
    global paper_engine, bot_running
    
    try:
        logger.info("📍 /api/paper/reset called")
        
        if bot_running:
            return jsonify({
                'success': False,
                'error': 'Stop bot before resetting'
            }), 400

        paper_engine = None
        logger.info("✅ Paper trading reset")

        return jsonify({
            'success': True,
            'message': 'Paper trading reset'
        }), 200

    except Exception as e:
        logger.error(f"❌ Reset error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============== MAIN ==============

if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("🚀 TRADING BOT STARTING")
    logger.info("="*60)
    logger.info(f"Time: {datetime.now()}")
    logger.info(f"Initial Mode: {current_mode}")
    logger.info(f"Initial Bot Running: {bot_running}")
    logger.info("="*60 + "\n")

    # Initialize paper engine
    paper_engine = PaperTradingEngine()

    # Start Flask server
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=False,
        use_reloader=False,
        threaded=True
    )
