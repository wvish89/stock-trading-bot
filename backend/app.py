"""
Trading Bot Backend API - Flask Server
Works with Railway deployment
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global state
trading_bot = None
bot_thread = None
trading_mode = "paper"
paper_engine = None


# ==================== SIMPLE CONFIG ====================

class Config:
    def __init__(self):
        self.API_KEY = os.getenv('ANGEL_API_KEY', '')
        self.CLIENT_ID = os.getenv('ANGEL_CLIENT_ID', '')
        self.PASSWORD = os.getenv('ANGEL_PASSWORD', '')
        self.TOTP_SECRET = os.getenv('ANGEL_TOTP_SECRET', '')
        self.CAPITAL = float(os.getenv('TRADING_CAPITAL', 100000))
        self.RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 0.05))
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))
        
        watchlist_str = os.getenv('WATCHLIST', 'RELIANCE-EQ,TCS-EQ,INFY-EQ,HDFCBANK-EQ,SBIN-EQ')
        self.WATCHLIST = [s.strip() for s in watchlist_str.split(',')]
        self.SQUARE_OFF_TIME = os.getenv('SQUARE_OFF_TIME', '15:15')


# ==================== PAPER TRADING ENGINE ====================

class PaperTradingEngine:
    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.daily_pnl = 0.0
        self.signals = []
    
    def place_order(self, symbol: str, transaction_type: str, quantity: int, price: float) -> Dict:
        order_id = f"PAPER_{len(self.orders) + 1}_{datetime.now().strftime('%H%M%S')}"
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'transaction_type': transaction_type,
            'quantity': quantity,
            'price': price,
            'status': 'COMPLETE',
            'timestamp': datetime.now().isoformat()
        }
        self.orders.append(order)
        
        if transaction_type == 'BUY':
            if symbol in self.positions:
                old_qty = self.positions[symbol]['quantity']
                old_price = self.positions[symbol]['avg_price']
                new_qty = old_qty + quantity
                self.positions[symbol]['quantity'] = new_qty
                self.positions[symbol]['avg_price'] = (old_price * old_qty + price * quantity) / new_qty
            else:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_time': datetime.now().isoformat(),
                    'stop_loss': price * 0.98,
                    'target': price * 1.04
                }
            self.capital -= (price * quantity)
            
        elif transaction_type == 'SELL':
            if symbol in self.positions:
                entry_price = self.positions[symbol]['avg_price']
                pnl = (price - entry_price) * quantity
                self.daily_pnl += pnl
                self.capital += (price * quantity)
                
                self.trades.append({
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'quantity': quantity,
                    'pnl': pnl,
                    'entry_time': self.positions[symbol]['entry_time'],
                    'exit_time': datetime.now().isoformat(),
                    'mode': 'paper'
                })
                
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] <= 0:
                    del self.positions[symbol]
        
        return order
    
    def add_signal(self, symbol: str, signal_type: str, price: float, confidence: float = 0.5):
        signal = {
            'symbol': symbol,
            'signal_type': signal_type,
            'price': price,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'indicators': {'rsi': 50, 'macd': 0, 'adx': 25}
        }
        self.signals.append(signal)
        if len(self.signals) > 100:
            self.signals = self.signals[-100:]
        return signal
    
    def get_positions(self) -> List[Dict]:
        return [
            {
                'symbol': symbol,
                'quantity': data['quantity'],
                'avg_price': data['avg_price'],
                'entry_time': data['entry_time'],
                'stop_loss': data.get('stop_loss', 0),
                'target': data.get('target', 0)
            }
            for symbol, data in self.positions.items()
        ]
    
    def get_portfolio_value(self) -> float:
        position_value = sum(
            data['quantity'] * data['avg_price']
            for data in self.positions.values()
        )
        return self.capital + position_value
    
    def reset(self):
        self.capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.daily_pnl = 0.0
        self.signals = []


# ==================== SIMPLE TRADING BOT ====================

class SimpleTradingBot:
    def __init__(self, paper_engine: PaperTradingEngine, config: Config):
        self.paper_engine = paper_engine
        self.config = config
        self.running = False
        self.daily_pnl = 0.0
        self.positions_count = 0
    
    def start(self):
        self.running = True
        logger.info("Trading bot started")
        self._monitor_loop()
    
    def stop(self):
        self.running = False
        logger.info("Trading bot stopped")
    
    def _monitor_loop(self):
        import time
        import random
        
        while self.running:
            try:
                for symbol in self.config.WATCHLIST:
                    # Generate random signal for demo
                    signal_types = ['BUY', 'SELL', 'HOLD', 'HOLD', 'HOLD']
                    signal_type = random.choice(signal_types)
                    price = random.uniform(1000, 5000)
                    confidence = random.uniform(0.3, 0.9)
                    
                    self.paper_engine.add_signal(symbol, signal_type, price, confidence)
                    logger.info(f"Signal generated: {signal_type} for {symbol} at {price:.2f}")
                    
                    # Execute trade if strong signal
                    if signal_type == 'BUY' and confidence > 0.7 and len(self.paper_engine.positions) < self.config.MAX_POSITIONS:
                        qty = int((self.config.CAPITAL * 0.1) / price)
                        if qty > 0:
                            self.paper_engine.place_order(symbol, 'BUY', qty, price)
                            logger.info(f"BUY executed: {symbol} x {qty} @ {price:.2f}")
                    
                    elif signal_type == 'SELL' and symbol in self.paper_engine.positions:
                        pos = self.paper_engine.positions[symbol]
                        self.paper_engine.place_order(symbol, 'SELL', pos['quantity'], price)
                        logger.info(f"SELL executed: {symbol} @ {price:.2f}")
                
                time.sleep(60)  # Wait 60 seconds
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(60)


# ==================== API ROUTES ====================

@app.route('/')
def index():
    return jsonify({
        'name': 'Trading Bot API',
        'version': '2.0.0',
        'status': 'running',
        'mode': trading_mode,
        'message': 'Welcome to Trading Bot API. Use /api/health to check status.'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'mode': trading_mode,
        'bot_running': trading_bot.running if trading_bot else False
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    try:
        config = Config()
        return jsonify({
            'success': True,
            'data': {
                'capital': config.CAPITAL,
                'risk_per_trade': config.RISK_PER_TRADE,
                'max_daily_loss': config.MAX_DAILY_LOSS,
                'max_positions': config.MAX_POSITIONS,
                'watchlist': config.WATCHLIST,
                'square_off_time': config.SQUARE_OFF_TIME,
                'mode': trading_mode,
                'api_configured': bool(config.API_KEY)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/mode', methods=['GET'])
def get_mode():
    return jsonify({
        'mode': trading_mode, 
        'bot_running': trading_bot.running if trading_bot else False
    })

@app.route('/api/mode', methods=['POST'])
def set_mode():
    global trading_mode, paper_engine
    
    try:
        data = request.json or {}
        new_mode = data.get('mode', 'paper')
        
        if new_mode not in ['live', 'paper']:
            return jsonify({'success': False, 'error': 'Invalid mode'}), 400
        
        if trading_bot and trading_bot.running:
            return jsonify({'success': False, 'error': 'Stop the bot before changing mode'}), 400
        
        trading_mode = new_mode
        
        if new_mode == 'paper' and paper_engine is None:
            config = Config()
            paper_engine = PaperTradingEngine(config.CAPITAL)
        
        return jsonify({'success': True, 'mode': trading_mode})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    global trading_bot, bot_thread, paper_engine
    
    try:
        if trading_bot and trading_bot.running:
            return jsonify({'success': False, 'error': 'Bot is already running'}), 400
        
        config = Config()
        
        if paper_engine is None:
            paper_engine = PaperTradingEngine(config.CAPITAL)
        
        trading_bot = SimpleTradingBot(paper_engine, config)
        
        bot_thread = threading.Thread(target=trading_bot.start, daemon=True)
        bot_thread.start()
        
        return jsonify({'success': True, 'message': f'Bot started in {trading_mode} mode'})
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    global trading_bot
    
    try:
        if not trading_bot or not trading_bot.running:
            return jsonify({'success': False, 'error': 'Bot is not running'}), 400
        
        trading_bot.stop()
        return jsonify({'success': True, 'message': 'Bot stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/status', methods=['GET'])
def bot_status():
    running = trading_bot.running if trading_bot else False
    daily_pnl = paper_engine.daily_pnl if paper_engine else 0
    positions_count = len(paper_engine.positions) if paper_engine else 0
    
    return jsonify({
        'running': running,
        'mode': trading_mode,
        'positions_count': positions_count,
        'daily_pnl': daily_pnl
    })

@app.route('/api/positions', methods=['GET'])
def get_positions():
    try:
        if paper_engine:
            positions = paper_engine.get_positions()
        else:
            positions = []
        return jsonify({'success': True, 'data': positions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trades', methods=['GET'])
def get_trades():
    try:
        if paper_engine and paper_engine.trades:
            trades = paper_engine.trades
            winning = [t for t in trades if t.get('pnl', 0) > 0]
            losing = [t for t in trades if t.get('pnl', 0) < 0]
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            
            return jsonify({
                'success': True,
                'data': trades,
                'statistics': {
                    'total_trades': len(trades),
                    'winning_trades': len(winning),
                    'losing_trades': len(losing),
                    'win_rate': (len(winning) / len(trades) * 100) if trades else 0,
                    'total_pnl': total_pnl
                }
            })
        else:
            return jsonify({
                'success': True,
                'data': [],
                'statistics': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0
                }
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals', methods=['GET'])
def get_signals():
    try:
        if paper_engine and paper_engine.signals:
            return jsonify({'success': True, 'data': paper_engine.signals[-50:]})
        return jsonify({'success': True, 'data': []})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    try:
        config = Config()
        
        if paper_engine:
            total_value = paper_engine.get_portfolio_value()
            daily_pnl = paper_engine.daily_pnl
            cash = paper_engine.capital
        else:
            total_value = config.CAPITAL
            daily_pnl = 0.0
            cash = config.CAPITAL
        
        return jsonify({
            'success': True,
            'data': {
                'total_value': total_value,
                'cash': cash,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': (daily_pnl / config.CAPITAL * 100) if config.CAPITAL > 0 else 0,
                'invested_value': total_value - cash
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    try:
        config = Config()
        return jsonify({'success': True, 'data': config.WATCHLIST})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/paper/reset', methods=['POST'])
def reset_paper_trading():
    global paper_engine
    
    try:
        if trading_bot and trading_bot.running:
            return jsonify({'success': False, 'error': 'Stop the bot before resetting'}), 400
        
        config = Config()
        paper_engine = PaperTradingEngine(config.CAPITAL)
        
        return jsonify({'success': True, 'message': 'Paper trading account reset'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== MAIN ====================

# Initialize on startup
config = Config()
paper_engine = PaperTradingEngine(config.CAPITAL)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Mode: {trading_mode}")
    logger.info(f"Watchlist: {config.WATCHLIST}")
    app.run(host='0.0.0.0', port=port, debug=False)
