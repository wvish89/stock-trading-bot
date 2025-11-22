"""
Trading Bot Backend API - Flask Server
Provides REST API for frontend dashboard
Supports both Live and Paper Trading modes
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import sqlite3

# Import trading bot components
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot import (
    TradingBot, Config, BrokerAPI, TradingStrategy,
    Backtester, DatabaseManager, RiskManager,
    TradingSignal, SignalType
)

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
trading_bot = None
bot_thread = None
trading_mode = "paper"  # "paper" or "live"
paper_trading_engine = None


# ==================== PAPER TRADING ENGINE ====================

class PaperTradingEngine:
    """Simulates trading without real orders"""
    
    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.daily_pnl = 0.0
        
    def place_order(self, symbol: str, transaction_type: str, 
                   quantity: int, price: float) -> Dict:
        """Simulate order placement"""
        order_id = f"PAPER_{len(self.orders) + 1}"
        
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
        
        # Update positions
        if transaction_type == 'BUY':
            if symbol in self.positions:
                self.positions[symbol]['quantity'] += quantity
                self.positions[symbol]['avg_price'] = (
                    (self.positions[symbol]['avg_price'] * 
                     (self.positions[symbol]['quantity'] - quantity) +
                     price * quantity) / self.positions[symbol]['quantity']
                )
            else:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_time': datetime.now()
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
                    'exit_time': datetime.now()
                })
                
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] <= 0:
                    del self.positions[symbol]
        
        return order
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        return [
            {
                'symbol': symbol,
                'quantity': data['quantity'],
                'avg_price': data['avg_price'],
                'entry_time': data['entry_time'].isoformat()
            }
            for symbol, data in self.positions.items()
        ]
    
    def get_portfolio_value(self, current_prices: Dict) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            data['quantity'] * current_prices.get(symbol, data['avg_price'])
            for symbol, data in self.positions.items()
        )
        return self.capital + position_value
    
    def reset(self):
        """Reset paper trading account"""
        self.capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.daily_pnl = 0.0


# ==================== API ROUTES ====================

@app.route('/')
def serve():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'mode': trading_mode
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
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
                'mode': trading_mode
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        data = request.json
        config = Config()
        
        # Load current config
        with open('config.json', 'r') as f:
            config_data = json.load(f)
        
        # Update fields
        if 'capital' in data:
            config_data['capital'] = data['capital']
        if 'risk_per_trade' in data:
            config_data['risk_per_trade'] = data['risk_per_trade']
        if 'max_daily_loss' in data:
            config_data['max_daily_loss'] = data['max_daily_loss']
        if 'max_positions' in data:
            config_data['max_positions'] = data['max_positions']
        if 'watchlist' in data:
            config_data['watchlist'] = data['watchlist']
        
        # Save config
        with open('config.json', 'w') as f:
            json.dump(config_data, f, indent=4)
        
        return jsonify({'success': True, 'message': 'Configuration updated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/mode', methods=['GET'])
def get_mode():
    """Get current trading mode"""
    return jsonify({'mode': trading_mode})

@app.route('/api/mode', methods=['POST'])
def set_mode():
    """Set trading mode (live or paper)"""
    global trading_mode, paper_trading_engine
    
    try:
        data = request.json
        new_mode = data.get('mode', 'paper')
        
        if new_mode not in ['live', 'paper']:
            return jsonify({'success': False, 'error': 'Invalid mode'}), 400
        
        # Stop bot if running
        if trading_bot and trading_bot.running:
            return jsonify({
                'success': False, 
                'error': 'Stop the bot before changing mode'
            }), 400
        
        trading_mode = new_mode
        
        # Initialize paper trading if needed
        if new_mode == 'paper' and paper_trading_engine is None:
            config = Config()
            paper_trading_engine = PaperTradingEngine(config.CAPITAL)
        
        return jsonify({
            'success': True, 
            'mode': trading_mode,
            'message': f'Trading mode set to {trading_mode}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Start trading bot"""
    global trading_bot, bot_thread
    
    try:
        if trading_bot and trading_bot.running:
            return jsonify({
                'success': False, 
                'error': 'Bot is already running'
            }), 400
        
        # Create bot instance based on mode
        if trading_mode == 'paper':
            trading_bot = PaperTradingBot()
        else:
            trading_bot = TradingBot()
        
        # Start bot in separate thread
        bot_thread = threading.Thread(target=trading_bot.start)
        bot_thread.daemon = True
        bot_thread.start()
        
        socketio.emit('bot_status', {'status': 'running', 'mode': trading_mode})
        
        return jsonify({
            'success': True, 
            'message': f'Bot started in {trading_mode} mode'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Stop trading bot"""
    global trading_bot
    
    try:
        if not trading_bot or not trading_bot.running:
            return jsonify({
                'success': False, 
                'error': 'Bot is not running'
            }), 400
        
        trading_bot.stop()
        socketio.emit('bot_status', {'status': 'stopped'})
        
        return jsonify({'success': True, 'message': 'Bot stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/status', methods=['GET'])
def bot_status():
    """Get bot status"""
    if trading_bot and trading_bot.running:
        return jsonify({
            'running': True,
            'mode': trading_mode,
            'positions_count': len(trading_bot.positions),
            'daily_pnl': trading_bot.risk_manager.daily_pnl
        })
    return jsonify({'running': False, 'mode': trading_mode})

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get current positions"""
    try:
        if trading_mode == 'paper' and paper_trading_engine:
            positions = paper_trading_engine.get_positions()
        elif trading_bot:
            positions = list(trading_bot.positions.values())
        else:
            positions = []
        
        return jsonify({'success': True, 'data': positions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trade history"""
    try:
        db = DatabaseManager()
        trades_df = db.get_trade_history()
        
        if len(trades_df) > 0:
            trades = trades_df.to_dict('records')
            
            # Calculate statistics
            total_pnl = trades_df['pnl'].sum()
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = (winning_trades / len(trades_df) * 100) if len(trades_df) > 0 else 0
            
            return jsonify({
                'success': True,
                'data': trades,
                'statistics': {
                    'total_trades': len(trades_df),
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
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

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get portfolio summary"""
    try:
        config = Config()
        
        if trading_mode == 'paper' and paper_trading_engine:
            current_value = paper_trading_engine.get_portfolio_value({})
            daily_pnl = paper_trading_engine.daily_pnl
            capital = paper_trading_engine.capital
        elif trading_bot:
            current_value = config.CAPITAL + trading_bot.risk_manager.daily_pnl
            daily_pnl = trading_bot.risk_manager.daily_pnl
            capital = config.CAPITAL
        else:
            current_value = config.CAPITAL
            daily_pnl = 0.0
            capital = config.CAPITAL
        
        return jsonify({
            'success': True,
            'data': {
                'total_value': current_value,
                'cash': capital,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': (daily_pnl / config.CAPITAL * 100),
                'invested_value': current_value - capital
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get recent trading signals"""
    try:
        conn = sqlite3.connect('trading_bot.db')
        query = """
            SELECT * FROM signals 
            ORDER BY timestamp DESC 
            LIMIT 50
        """
        signals_df = pd.read_sql_query(query, conn)
        conn.close()
        
        signals = signals_df.to_dict('records') if len(signals_df) > 0 else []
        
        return jsonify({'success': True, 'data': signals})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest on historical data"""
    try:
        data = request.json
        symbol = data.get('symbol', 'RELIANCE-EQ')
        days = data.get('days', 30)
        
        config = Config()
        broker = BrokerAPI(config)
        backtester = Backtester(config)
        
        # Fetch historical data
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
        
        df = broker.get_historical_data(
            symbol=symbol,
            exchange="NSE",
            interval="FIVE_MINUTE",
            from_date=from_date,
            to_date=to_date
        )
        
        if df is None or len(df) < 50:
            return jsonify({
                'success': False, 
                'error': 'Insufficient data for backtesting'
            }), 400
        
        # Run backtest
        results = backtester.run_backtest(df, symbol)
        
        return jsonify({
            'success': True,
            'data': {
                'metrics': results['metrics'],
                'equity_curve': results['equity_curve'],
                'trades_count': len(results['trades'])
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    """Get watchlist symbols"""
    try:
        config = Config()
        return jsonify({'success': True, 'data': config.WATCHLIST})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist', methods=['POST'])
def update_watchlist():
    """Update watchlist"""
    try:
        data = request.json
        symbols = data.get('symbols', [])
        
        with open('config.json', 'r') as f:
            config_data = json.load(f)
        
        config_data['watchlist'] = symbols
        
        with open('config.json', 'w') as f:
            json.dump(config_data, f, indent=4)
        
        return jsonify({'success': True, 'message': 'Watchlist updated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/market-data/<symbol>', methods=['GET'])
def get_market_data(symbol):
    """Get live market data for a symbol"""
    try:
        config = Config()
        broker = BrokerAPI(config)
        
        ltp = broker.get_ltp(symbol)
        
        if ltp:
            return jsonify({
                'success': True,
                'data': {
                    'symbol': symbol,
                    'ltp': ltp,
                    'timestamp': datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                'success': False, 
                'error': 'Failed to fetch market data'
            }), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/paper/reset', methods=['POST'])
def reset_paper_trading():
    """Reset paper trading account"""
    global paper_trading_engine
    
    try:
        if trading_mode != 'paper':
            return jsonify({
                'success': False, 
                'error': 'Not in paper trading mode'
            }), 400
        
        if paper_trading_engine:
            paper_trading_engine.reset()
            return jsonify({
                'success': True, 
                'message': 'Paper trading account reset'
            })
        else:
            return jsonify({
                'success': False, 
                'error': 'Paper trading engine not initialized'
            }), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== PAPER TRADING BOT ====================

class PaperTradingBot(TradingBot):
    """Extended TradingBot for paper trading"""
    
    def __init__(self):
        super().__init__()
        self.paper_engine = paper_trading_engine
    
    def execute_buy(self, signal: TradingSignal):
        """Execute paper buy order"""
        try:
            quantity = self.risk_manager.calculate_position_size(
                signal.price, signal.stop_loss
            )
            
            if quantity <= 0:
                return
            
            # Place paper order
            order = self.paper_engine.place_order(
                symbol=signal.symbol,
                transaction_type='BUY',
                quantity=quantity,
                price=signal.price
            )
            
            self.positions[signal.symbol] = {
                'quantity': quantity,
                'entry_price': signal.price,
                'stop_loss': signal.stop_loss,
                'target': signal.target,
                'order_id': order['order_id']
            }
            
            # Emit to frontend
            socketio.emit('new_trade', {
                'type': 'BUY',
                'symbol': signal.symbol,
                'price': signal.price,
                'quantity': quantity
            })
            
            self.notifier.notify_signal(signal)
            
        except Exception as e:
            print(f"Error executing paper buy: {e}")
    
    def execute_sell(self, signal: TradingSignal):
        """Execute paper sell order"""
        try:
            position = self.positions.get(signal.symbol)
            if not position:
                return
            
            # Place paper order
            order = self.paper_engine.place_order(
                symbol=signal.symbol,
                transaction_type='SELL',
                quantity=position['quantity'],
                price=signal.price
            )
            
            pnl = (signal.price - position['entry_price']) * position['quantity']
            self.risk_manager.update_daily_pnl(pnl)
            
            # Emit to frontend
            socketio.emit('new_trade', {
                'type': 'SELL',
                'symbol': signal.symbol,
                'price': signal.price,
                'quantity': position['quantity'],
                'pnl': pnl
            })
            
            # Save to database
            self.db.save_trade({
                'symbol': signal.symbol,
                'entry_time': datetime.now() - timedelta(minutes=30),
                'exit_time': datetime.now(),
                'entry_price': position['entry_price'],
                'exit_price': signal.price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'strategy': 'Paper Trading',
                'notes': f'Paper trade. P&L: ₹{pnl:.2f}'
            })
            
            del self.positions[signal.symbol]
            
        except Exception as e:
            print(f"Error executing paper sell: {e}")


# ==================== WEBSOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('subscribe_updates')
def handle_subscribe():
    """Subscribe to real-time updates"""
    emit('subscribed', {'message': 'Subscribed to updates'})


# ==================== STARTUP ====================

if __name__ == '__main__':
    print("=" * 60)
    print("TRADING BOT BACKEND SERVER")
    print("=" * 60)
    print(f"Server starting on http://localhost:5000")
    print(f"Trading Mode: {trading_mode}")
    print("=" * 60)
    
    # Initialize paper trading engine
    config = Config()
    paper_trading_engine = PaperTradingEngine(config.CAPITAL)
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)