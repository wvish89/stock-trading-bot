"""
Smart Trading Bot Backend API v3.0
Enhanced with: Technical Analysis, Risk Management, Stock Selection, Trading Rules
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import os
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional
import logging
import random
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS - Allow Netlify frontend
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://vermillion-kheer-9eeb5f.netlify.app",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "*"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Global state
trading_bot = None
bot_thread = None
trading_mode = "paper"
paper_engine = None


class Config:
    """Configuration with all strategy parameters"""
    def __init__(self):
        self.API_KEY = os.getenv('ANGEL_API_KEY', '')
        self.CLIENT_ID = os.getenv('ANGEL_CLIENT_ID', '')
        self.PASSWORD = os.getenv('ANGEL_PASSWORD', '')
        self.TOTP_SECRET = os.getenv('ANGEL_TOTP_SECRET', '')
        
        # Capital & Risk Management
        self.CAPITAL = float(os.getenv('TRADING_CAPITAL', 100000))
        self.RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.01))  # 1% max risk per trade
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 0.03))  # 3% max daily loss
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))
        self.MIN_RISK_REWARD = float(os.getenv('MIN_RISK_REWARD', 2.0))  # Minimum 1:2 RR ratio
        
        # Trading Hours
        self.MARKET_OPEN = dtime(9, 15)
        self.MARKET_CLOSE = dtime(15, 30)
        self.AVOID_OPENING_MINUTES = int(os.getenv('AVOID_OPENING_MINUTES', 30))  # Skip first 30 mins
        self.SQUARE_OFF_TIME = os.getenv('SQUARE_OFF_TIME', '15:15')
        
        # Technical Indicators
        self.RSI_PERIOD = 14
        self.RSI_OVERBOUGHT = 70
        self.RSI_OVERSOLD = 30
        self.EMA_FAST = 9
        self.EMA_SLOW = 21
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # Watchlist - High liquidity, large-cap stocks
        watchlist_str = os.getenv('WATCHLIST', 
            'RELIANCE-EQ,TCS-EQ,INFY-EQ,HDFCBANK-EQ,ICICIBANK-EQ,SBIN-EQ,BHARTIARTL-EQ,ITC-EQ,KOTAKBANK-EQ,LT-EQ')
        self.WATCHLIST = [s.strip() for s in watchlist_str.split(',')]
        
        # Stock selection criteria
        self.MIN_VOLUME = 100000  # Minimum daily volume
        self.MAX_SPREAD_PCT = 0.5  # Max bid-ask spread %


class TechnicalAnalyzer:
    """Technical and Chart Analysis"""
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        if len(prices) < period:
            return prices
        multiplier = 2 / (period + 1)
        ema = [sum(prices[:period]) / period]
        for price in prices[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        return [None] * (period - 1) + ema
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> List[float]:
        sma = []
        for i in range(len(prices)):
            if i < period - 1:
                sma.append(None)
            else:
                sma.append(sum(prices[i-period+1:i+1]) / period)
        return sma
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict:
        ema12 = TechnicalAnalyzer.calculate_ema(prices, 12)
        ema26 = TechnicalAnalyzer.calculate_ema(prices, 26)
        macd_line = []
        for i in range(len(prices)):
            if ema12[i] is not None and ema26[i] is not None:
                macd_line.append(ema12[i] - ema26[i])
            else:
                macd_line.append(None)
        valid_macd = [m for m in macd_line if m is not None]
        signal_line = TechnicalAnalyzer.calculate_ema(valid_macd, 9) if len(valid_macd) >= 9 else []
        return {
            'macd': macd_line[-1] if macd_line and macd_line[-1] else 0,
            'signal': signal_line[-1] if signal_line else 0,
            'histogram': (macd_line[-1] or 0) - (signal_line[-1] if signal_line else 0)
        }
    
    @staticmethod
    def detect_trend(prices: List[float]) -> str:
        if len(prices) < 21:
            return 'sideways'
        ema9 = TechnicalAnalyzer.calculate_ema(prices, 9)
        ema21 = TechnicalAnalyzer.calculate_ema(prices, 21)
        if ema9[-1] and ema21[-1]:
            if ema9[-1] > ema21[-1] * 1.002:
                return 'uptrend'
            elif ema9[-1] < ema21[-1] * 0.998:
                return 'downtrend'
        return 'sideways'
    
    @staticmethod
    def find_support_resistance(prices: List[float]) -> Dict:
        if len(prices) < 20:
            return {'support': prices[-1] * 0.98, 'resistance': prices[-1] * 1.02}
        recent = prices[-20:]
        return {
            'support': min(recent),
            'resistance': max(recent)
        }
    
    @staticmethod
    def detect_breakout(prices: List[float], volume: List[float]) -> Optional[str]:
        if len(prices) < 20 or len(volume) < 20:
            return None
        sr = TechnicalAnalyzer.find_support_resistance(prices[:-1])
        current_price = prices[-1]
        avg_volume = sum(volume[-20:-1]) / 19
        current_volume = volume[-1]
        
        # Breakout with volume confirmation
        if current_price > sr['resistance'] and current_volume > avg_volume * 1.5:
            return 'bullish_breakout'
        elif current_price < sr['support'] and current_volume > avg_volume * 1.5:
            return 'bearish_breakout'
        return None


class RiskManager:
    """Risk Management System"""
    
    def __init__(self, config: Config):
        self.config = config
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 10
    
    def can_trade(self) -> tuple:
        """Check if trading is allowed based on risk rules"""
        # Check daily loss limit
        if self.daily_pnl <= -self.config.CAPITAL * self.config.MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"
        
        # Check max trades
        if self.daily_trades >= self.max_daily_trades:
            return False, "Max daily trades reached"
        
        return True, "OK"
    
    def calculate_position_size(self, entry: float, stop_loss: float) -> int:
        """Calculate position size based on risk per trade"""
        risk_amount = self.config.CAPITAL * self.config.RISK_PER_TRADE
        risk_per_share = abs(entry - stop_loss)
        if risk_per_share == 0:
            return 0
        quantity = int(risk_amount / risk_per_share)
        # Also limit by capital available
        max_by_capital = int((self.config.CAPITAL * 0.2) / entry)  # Max 20% per position
        return min(quantity, max_by_capital, 100)  # Cap at 100 shares
    
    def validate_risk_reward(self, entry: float, stop_loss: float, target: float) -> bool:
        """Ensure trade meets minimum risk-reward ratio"""
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        if risk == 0:
            return False
        rr_ratio = reward / risk
        return rr_ratio >= self.config.MIN_RISK_REWARD
    
    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl
        self.daily_trades += 1
    
    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0


class MarketTimeManager:
    """Trading Time Management"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def is_market_open(self) -> bool:
        now = datetime.now()
        current_time = now.time()
        # Check if weekday
        if now.weekday() >= 5:
            return False
        return self.config.MARKET_OPEN <= current_time <= self.config.MARKET_CLOSE
    
    def should_avoid_trading(self) -> tuple:
        """Check if we should avoid trading (opening rush, closing time)"""
        now = datetime.now()
        current_time = now.time()
        
        # Avoid first 30 minutes (opening volatility)
        avoid_until = (datetime.combine(now.date(), self.config.MARKET_OPEN) + 
                      timedelta(minutes=self.config.AVOID_OPENING_MINUTES)).time()
        if current_time < avoid_until:
            return True, f"Avoiding first {self.config.AVOID_OPENING_MINUTES} mins"
        
        # Square off time
        sq_hour, sq_min = map(int, self.config.SQUARE_OFF_TIME.split(':'))
        square_off = dtime(sq_hour, sq_min)
        if current_time >= square_off:
            return True, "Square-off time reached"
        
        return False, "OK"
    
    def get_market_status(self) -> str:
        if not self.is_market_open():
            return 'closed'
        avoid, _ = self.should_avoid_trading()
        if avoid:
            return 'restricted'
        return 'open'


class PaperTradingEngine:
    """Paper Trading Simulation Engine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.capital = config.CAPITAL
        self.initial_capital = config.CAPITAL
        self.positions: Dict[str, Dict] = {}
        self.orders = []
        self.trades = []
        self.daily_pnl = 0.0
        self.signals = []
        self.risk_manager = RiskManager(config)
        self.time_manager = MarketTimeManager(config)
        self.analyzer = TechnicalAnalyzer()
        
        # Simulated price data
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self._init_price_data()
    
    def _init_price_data(self):
        """Initialize simulated price data for each stock"""
        base_prices = {
            'RELIANCE-EQ': 2450, 'TCS-EQ': 3800, 'INFY-EQ': 1450,
            'HDFCBANK-EQ': 1650, 'ICICIBANK-EQ': 1050, 'SBIN-EQ': 620,
            'BHARTIARTL-EQ': 1150, 'ITC-EQ': 440, 'KOTAKBANK-EQ': 1750, 'LT-EQ': 3200
        }
        for symbol in self.config.WATCHLIST:
            base = base_prices.get(symbol, 1000)
            # Generate 50 candles of historical data
            prices = [base]
            volumes = [random.randint(100000, 500000)]
            for _ in range(49):
                change = random.gauss(0, 0.01)  # 1% std dev
                prices.append(prices[-1] * (1 + change))
                volumes.append(random.randint(100000, 500000))
            self.price_history[symbol] = prices
            self.volume_history[symbol] = volumes
    
    def _update_prices(self):
        """Simulate price movement"""
        for symbol in self.price_history:
            change = random.gauss(0, 0.005)  # 0.5% std dev per update
            new_price = self.price_history[symbol][-1] * (1 + change)
            self.price_history[symbol].append(new_price)
            self.volume_history[symbol].append(random.randint(100000, 500000))
            # Keep last 100 candles
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
                self.volume_history[symbol] = self.volume_history[symbol][-100:]
    
    def analyze_stock(self, symbol: str) -> Dict:
        """Perform technical analysis on a stock"""
        prices = self.price_history.get(symbol, [])
        volumes = self.volume_history.get(symbol, [])
        if len(prices) < 30:
            return {'signal': 'HOLD', 'confidence': 0}
        
        current_price = prices[-1]
        rsi = self.analyzer.calculate_rsi(prices)
        macd = self.analyzer.calculate_macd(prices)
        trend = self.analyzer.detect_trend(prices)
        sr = self.analyzer.find_support_resistance(prices)
        breakout = self.analyzer.detect_breakout(prices, volumes)
        
        # Scoring system
        score = 0
        signals = []
        
        # RSI signals
        if rsi < 30:
            score += 2
            signals.append('RSI oversold')
        elif rsi > 70:
            score -= 2
            signals.append('RSI overbought')
        elif 40 < rsi < 60:
            score += 1
            signals.append('RSI neutral')
        
        # MACD signals
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            score += 2
            signals.append('MACD bullish')
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            score -= 2
            signals.append('MACD bearish')
        
        # Trend
        if trend == 'uptrend':
            score += 1
            signals.append('Uptrend')
        elif trend == 'downtrend':
            score -= 1
            signals.append('Downtrend')
        
        # Breakout
        if breakout == 'bullish_breakout':
            score += 3
            signals.append('Bullish breakout!')
        elif breakout == 'bearish_breakout':
            score -= 3
            signals.append('Bearish breakout!')
        
        # Determine signal
        if score >= 3:
            signal_type = 'BUY'
            stop_loss = sr['support'] * 0.99
            target = current_price + (current_price - stop_loss) * 2  # 1:2 RR
        elif score <= -3:
            signal_type = 'SELL'
            stop_loss = sr['resistance'] * 1.01
            target = current_price - (stop_loss - current_price) * 2
        else:
            signal_type = 'HOLD'
            stop_loss = 0
            target = 0
        
        confidence = min(abs(score) / 6, 1.0)
        risk = abs(current_price - stop_loss) if stop_loss else 0
        reward = abs(target - current_price) if target else 0
        rr_ratio = f"1:{reward/risk:.1f}" if risk > 0 else "N/A"
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'price': round(current_price, 2),
            'confidence': round(confidence, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'risk_reward': rr_ratio,
            'indicators': {
                'rsi': round(rsi, 2),
                'macd': round(macd['macd'], 4),
                'macd_signal': round(macd['signal'], 4),
                'trend': trend,
                'support': round(sr['support'], 2),
                'resistance': round(sr['resistance'], 2)
            },
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }
    
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
                pos = self.positions[symbol]
                old_qty = pos['quantity']
                old_price = pos['avg_price']
                new_qty = old_qty + quantity
                pos['quantity'] = new_qty
                pos['avg_price'] = (old_price * old_qty + price * quantity) / new_qty
            else:
                analysis = self.analyze_stock(symbol)
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_time': datetime.now().isoformat(),
                    'stop_loss': analysis['stop_loss'],
                    'target': analysis['target']
                }
            self.capital -= (price * quantity)
            
        elif transaction_type == 'SELL':
            if symbol in self.positions:
                pos = self.positions[symbol]
                entry_price = pos['avg_price']
                pnl = (price - entry_price) * quantity
                self.daily_pnl += pnl
                self.risk_manager.update_daily_pnl(pnl)
                self.capital += (price * quantity)
                
                self.trades.append({
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'quantity': quantity,
                    'pnl': round(pnl, 2),
                    'entry_time': pos['entry_time'],
                    'exit_time': datetime.now().isoformat(),
                    'mode': 'paper'
                })
                
                pos['quantity'] -= quantity
                if pos['quantity'] <= 0:
                    del self.positions[symbol]
        
        return order
    
    def add_signal(self, analysis: Dict):
        self.signals.append(analysis)
        if len(self.signals) > 100:
            self.signals = self.signals[-100:]
    
    def get_positions(self) -> List[Dict]:
        return [
            {
                'symbol': symbol,
                'quantity': data['quantity'],
                'avg_price': data['avg_price'],
                'entry_time': data['entry_time'],
                'stop_loss': data.get('stop_loss', 0),
                'target': data.get('target', 0),
                'current_price': self.price_history.get(symbol, [0])[-1],
                'unrealized_pnl': round((self.price_history.get(symbol, [0])[-1] - data['avg_price']) * data['quantity'], 2)
            }
            for symbol, data in self.positions.items()
        ]
    
    def get_portfolio_value(self) -> float:
        position_value = sum(
            self.price_history.get(symbol, [data['avg_price']])[-1] * data['quantity']
            for symbol, data in self.positions.items()
        )
        return self.capital + position_value
    
    def check_stop_loss_target(self):
        """Auto-exit positions that hit SL or target"""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            current_price = self.price_history.get(symbol, [pos['avg_price']])[-1]
            
            if current_price <= pos.get('stop_loss', 0):
                logger.info(f"STOP LOSS hit for {symbol}")
                self.place_order(symbol, 'SELL', pos['quantity'], current_price)
            elif current_price >= pos.get('target', float('inf')):
                logger.info(f"TARGET hit for {symbol}")
                self.place_order(symbol, 'SELL', pos['quantity'], current_price)
    
    def reset(self):
        self.capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.daily_pnl = 0.0
        self.signals = []
        self.risk_manager.reset_daily()
        self._init_price_data()


class SmartTradingBot:
    """Enhanced Trading Bot with Strategy Rules"""
    
    def __init__(self, paper_engine: PaperTradingEngine, config: Config):
        self.paper_engine = paper_engine
        self.config = config
        self.running = False
    
    def start(self):
        self.running = True
        logger.info("Smart Trading Bot started")
        self._monitor_loop()
    
    def stop(self):
        self.running = False
        # Square off all positions
        for symbol in list(self.paper_engine.positions.keys()):
            pos = self.paper_engine.positions[symbol]
            price = self.paper_engine.price_history.get(symbol, [pos['avg_price']])[-1]
            self.paper_engine.place_order(symbol, 'SELL', pos['quantity'], price)
        logger.info("Smart Trading Bot stopped")
    
    def _monitor_loop(self):
        import time
        
        while self.running:
            try:
                # Update simulated prices
                self.paper_engine._update_prices()
                
                # Check stop loss and targets
                self.paper_engine.check_stop_loss_target()
                
                # Check if we can trade
                can_trade, reason = self.paper_engine.risk_manager.can_trade()
                avoid, avoid_reason = self.paper_engine.time_manager.should_avoid_trading()
                
                if not can_trade:
                    logger.info(f"Trading paused: {reason}")
                    time.sleep(30)
                    continue
                
                # Analyze each stock
                for symbol in self.config.WATCHLIST:
                    analysis = self.paper_engine.analyze_stock(symbol)
                    self.paper_engine.add_signal(analysis)
                    
                    # Skip if avoiding trading times (but still generate signals)
                    if avoid:
                        continue
                    
                    # Execute trades based on signals
                    if analysis['signal_type'] == 'BUY' and analysis['confidence'] >= 0.5:
                        # Check if we can take more positions
                        if len(self.paper_engine.positions) >= self.config.MAX_POSITIONS:
                            continue
                        if symbol in self.paper_engine.positions:
                            continue
                        
                        # Validate risk-reward
                        if not self.paper_engine.risk_manager.validate_risk_reward(
                            analysis['price'], analysis['stop_loss'], analysis['target']
                        ):
                            logger.info(f"Skipping {symbol}: Poor risk-reward")
                            continue
                        
                        # Calculate position size
                        qty = self.paper_engine.risk_manager.calculate_position_size(
                            analysis['price'], analysis['stop_loss']
                        )
                        
                        if qty > 0:
                            self.paper_engine.place_order(symbol, 'BUY', qty, analysis['price'])
                            logger.info(f"BUY {symbol} x {qty} @ {analysis['price']}")
                    
                    elif analysis['signal_type'] == 'SELL' and symbol in self.paper_engine.positions:
                        pos = self.paper_engine.positions[symbol]
                        self.paper_engine.place_order(symbol, 'SELL', pos['quantity'], analysis['price'])
                        logger.info(f"SELL {symbol} @ {analysis['price']}")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(30)


# Initialize
config = Config()
paper_engine = PaperTradingEngine(config)


# ==================== API ROUTES ====================

@app.route('/')
def index():
    return jsonify({
        'name': 'Smart Trading Bot API',
        'version': '3.0.0',
        'status': 'running',
        'mode': trading_mode
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    time_mgr = MarketTimeManager(config)
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'mode': trading_mode,
        'market_status': time_mgr.get_market_status(),
        'bot_running': trading_bot.running if trading_bot else False
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'success': True,
        'data': {
            'capital': config.CAPITAL,
            'risk_per_trade': config.RISK_PER_TRADE,
            'max_daily_loss': config.MAX_DAILY_LOSS,
            'max_positions': config.MAX_POSITIONS,
            'min_risk_reward': config.MIN_RISK_REWARD,
            'watchlist': config.WATCHLIST,
            'square_off_time': config.SQUARE_OFF_TIME,
            'avoid_opening_minutes': config.AVOID_OPENING_MINUTES,
            'mode': trading_mode
        }
    })

@app.route('/api/mode', methods=['GET', 'POST'])
def handle_mode():
    global trading_mode, paper_engine
    
    if request.method == 'GET':
        return jsonify({'mode': trading_mode})
    
    data = request.json or {}
    new_mode = data.get('mode', 'paper')
    
    if new_mode not in ['live', 'paper']:
        return jsonify({'success': False, 'error': 'Invalid mode'}), 400
    
    if trading_bot and trading_bot.running:
        return jsonify({'success': False, 'error': 'Stop bot first'}), 400
    
    trading_mode = new_mode
    return jsonify({'success': True, 'mode': trading_mode})

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    global trading_bot, bot_thread, paper_engine
    
    if trading_bot and trading_bot.running:
        return jsonify({'success': False, 'error': 'Already running'}), 400
    
    trading_bot = SmartTradingBot(paper_engine, config)
    bot_thread = threading.Thread(target=trading_bot.start, daemon=True)
    bot_thread.start()
    
    return jsonify({'success': True, 'message': f'Started in {trading_mode} mode'})

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    global trading_bot
    
    if not trading_bot or not trading_bot.running:
        return jsonify({'success': False, 'error': 'Not running'}), 400
    
    trading_bot.stop()
    return jsonify({'success': True, 'message': 'Stopped'})

@app.route('/api/bot/status', methods=['GET'])
def bot_status():
    time_mgr = MarketTimeManager(config)
    return jsonify({
        'running': trading_bot.running if trading_bot else False,
        'mode': trading_mode,
        'market_status': time_mgr.get_market_status(),
        'positions_count': len(paper_engine.positions),
        'daily_pnl': round(paper_engine.daily_pnl, 2)
    })

@app.route('/api/positions', methods=['GET'])
def get_positions():
    return jsonify({'success': True, 'data': paper_engine.get_positions()})

@app.route('/api/trades', methods=['GET'])
def get_trades():
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
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(sum(t['pnl'] for t in winning) / len(winning), 2) if winning else 0,
            'avg_loss': round(sum(t['pnl'] for t in losing) / len(losing), 2) if losing else 0
        }
    })

@app.route('/api/signals', methods=['GET'])
def get_signals():
    return jsonify({'success': True, 'data': paper_engine.signals[-50:]})

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    total_value = paper_engine.get_portfolio_value()
    return jsonify({
        'success': True,
        'data': {
            'total_value': round(total_value, 2),
            'cash': round(paper_engine.capital, 2),
            'daily_pnl': round(paper_engine.daily_pnl, 2),
            'daily_pnl_pct': round((paper_engine.daily_pnl / config.CAPITAL * 100), 2),
            'invested_value': round(total_value - paper_engine.capital, 2),
            'total_return_pct': round(((total_value - config.CAPITAL) / config.CAPITAL * 100), 2)
        }
    })

@app.route('/api/risk-metrics', methods=['GET'])
def get_risk_metrics():
    rm = paper_engine.risk_manager
    positions_value = sum(
        paper_engine.price_history.get(s, [d['avg_price']])[-1] * d['quantity']
        for s, d in paper_engine.positions.items()
    )
    return jsonify({
        'success': True,
        'data': {
            'daily_pnl': round(rm.daily_pnl, 2),
            'daily_trades': rm.daily_trades,
            'max_daily_trades': rm.max_daily_trades,
            'risk_per_trade_pct': config.RISK_PER_TRADE * 100,
            'max_daily_loss_pct': config.MAX_DAILY_LOSS * 100,
            'current_exposure': round(positions_value, 2),
            'risk_used_pct': round((positions_value / config.CAPITAL * 100), 2) if config.CAPITAL > 0 else 0,
            'can_trade': rm.can_trade()[0],
            'trade_status': rm.can_trade()[1]
        }
    })

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    return jsonify({'success': True, 'data': config.WATCHLIST})

@app.route('/api/analysis/<symbol>', methods=['GET'])
def get_analysis(symbol):
    """Get detailed analysis for a specific stock"""
    if symbol not in paper_engine.price_history:
        return jsonify({'success': False, 'error': 'Symbol not found'}), 404
    
    analysis = paper_engine.analyze_stock(symbol)
    return jsonify({'success': True, 'data': analysis})

@app.route('/api/paper/reset', methods=['POST'])
def reset_paper():
    global paper_engine
    
    if trading_bot and trading_bot.running:
        return jsonify({'success': False, 'error': 'Stop bot first'}), 400
    
    paper_engine.reset()
    return jsonify({'success': True, 'message': 'Paper trading reset'})


# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Smart Trading Bot API on port {port}")
    logger.info(f"Watchlist: {config.WATCHLIST}")
    logger.info(f"Risk per trade: {config.RISK_PER_TRADE * 100}%")
    logger.info(f"Min Risk-Reward: 1:{config.MIN_RISK_REWARD}")
    app.run(host='0.0.0.0', port=port, debug=False)
