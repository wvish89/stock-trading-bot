"""
Advanced Intelligent Trading Bot v5.0 - Ensemble Strategy System
Automatically combines all 7 strategies with intelligent selection
Auto-trades based on market conditions + strategy performance
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import os
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple
import logging
import random
import math
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://vermillion-kheer-9eeb5f.netlify.app",
            "http://localhost:3000",
            "*"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

trading_bot = None
bot_thread = None
paper_engine = None

class StrategyEnum:
    ORB = "opening_range_breakout"
    NEWS = "news_driven"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    MA_CROSSOVER = "ma_crossover"
    PIVOT_POINTS = "pivot_points"
    ALL = [ORB, NEWS, MOMENTUM, BREAKOUT, SCALPING, MA_CROSSOVER, PIVOT_POINTS]

class Config:
    def __init__(self):
        self.API_KEY = os.getenv('ANGEL_API_KEY', '')
        self.CLIENT_ID = os.getenv('ANGEL_CLIENT_ID', '')
        self.PASSWORD = os.getenv('ANGEL_PASSWORD', '')
        self.TOTP_SECRET = os.getenv('ANGEL_TOTP_SECRET', '')

        # Capital & Risk Management
        self.CAPITAL = float(os.getenv('TRADING_CAPITAL', 100000))
        self.RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.01))
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 0.03))
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))
        self.MIN_RISK_REWARD = float(os.getenv('MIN_RISK_REWARD', 2.0))

        # Trading Hours
        self.MARKET_OPEN = dtime(9, 15)
        self.MARKET_CLOSE = dtime(15, 30)
        self.AVOID_OPENING_MINUTES = int(os.getenv('AVOID_OPENING_MINUTES', 45))
        self.SQUARE_OFF_TIME = os.getenv('SQUARE_OFF_TIME', '15:15')

        # Ensemble Settings
        self.ENSEMBLE_MODE = True  # Use all strategies together
        self.STRATEGY_CONFIDENCE_THRESHOLD = 0.5
        self.USE_BEST_STRATEGY_ONLY = False  # If False, uses ensemble voting
        self.ENSEMBLE_VOTING_THRESHOLD = 3  # Min strategies agreeing to trade

        # Strategy-specific parameters
        self.ORB_PERIOD_MINUTES = 15
        self.MOMENTUM_PERIOD = 14
        self.MOMENTUM_VOLUME_MULTIPLIER = 1.5
        self.BREAKOUT_LOOKBACK_BARS = 20
        self.BREAKOUT_VOLUME_CONFIRMATION = 1.5
        self.SCALP_TARGET_POINTS = 5
        self.SCALP_SL_POINTS = 2
        self.SCALP_MIN_VOLUME = 100000
        self.MA_FAST_PERIOD = 9
        self.MA_SLOW_PERIOD = 21

        # Watchlist
        watchlist_str = os.getenv('WATCHLIST',
            'RELIANCE-EQ,TCS-EQ,INFY-EQ,HDFCBANK-EQ,ICICIBANK-EQ,SBIN-EQ,BHARTIARTL-EQ,ITC-EQ,KOTAKBANK-EQ,LT-EQ')
        self.WATCHLIST = [s.strip() for s in watchlist_str.split(',')]

class EnsembleStrategyAnalyzer:
    """Combines all 7 strategies with intelligent voting system"""

    def __init__(self, config: Config):
        self.config = config
        self.strategy_scores = {s: {'wins': 0, 'losses': 0, 'score': 0.5} for s in StrategyEnum.ALL}

    def analyze_orb(self, prices: List[float], volumes: List[float]) -> Dict:
        """Opening Range Breakout"""
        if len(prices) < self.config.ORB_PERIOD_MINUTES:
            return {'signal': 'HOLD', 'confidence': 0}
        
        opening_range = prices[:self.config.ORB_PERIOD_MINUTES]
        opening_high = max(opening_range)
        opening_low = min(opening_range)
        range_width = opening_high - opening_low
        current_price = prices[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else current_volume
        
        if current_price > opening_high and current_volume > avg_volume * 1.2:
            return {
                'signal': 'BUY',
                'confidence': 0.8,
                'price': current_price,
                'stop_loss': opening_low - (range_width * 0.1),
                'target': current_price + (range_width * 2),
                'strategy': 'ORB',
                'reason': f'ORB Bullish breakout'
            }
        elif current_price < opening_low and current_volume > avg_volume * 1.2:
            return {
                'signal': 'SELL',
                'confidence': 0.8,
                'price': current_price,
                'stop_loss': opening_high + (range_width * 0.1),
                'target': current_price - (range_width * 2),
                'strategy': 'ORB',
                'reason': f'ORB Bearish breakdown'
            }
        return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'ORB'}

    def analyze_news(self, prices: List[float], volumes: List[float]) -> Dict:
        """News-Driven Trading"""
        if len(prices) < 10:
            return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'NEWS'}
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        avg_price = np.mean(prices[-10:])
        avg_volume = np.mean(volumes[-10:])
        
        price_change_pct = ((current_price - avg_price) / avg_price) * 100
        volume_spike = current_volume / avg_volume
        
        if abs(price_change_pct) > 1.5 and volume_spike > 2.0:
            if price_change_pct > 0:
                return {
                    'signal': 'BUY',
                    'confidence': 0.7,
                    'price': current_price,
                    'stop_loss': current_price * 0.97,
                    'target': current_price * 1.04,
                    'strategy': 'NEWS',
                    'reason': f'NEWS Positive reaction'
                }
            else:
                return {
                    'signal': 'SELL',
                    'confidence': 0.7,
                    'price': current_price,
                    'stop_loss': current_price * 1.03,
                    'target': current_price * 0.96,
                    'strategy': 'NEWS',
                    'reason': f'NEWS Negative reaction'
                }
        return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'NEWS'}

    def analyze_momentum(self, prices: List[float], volumes: List[float]) -> Dict:
        """Momentum Trading"""
        if len(prices) < self.config.MOMENTUM_PERIOD:
            return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'MOMENTUM'}
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        ma_fast = np.mean(prices[-5:])
        ma_slow = np.mean(prices[-self.config.MOMENTUM_PERIOD:])
        avg_volume = np.mean(volumes[-20:])
        
        if current_price > ma_slow and current_volume > avg_volume * self.config.MOMENTUM_VOLUME_MULTIPLIER and ma_fast > ma_slow:
            momentum_strength = ((current_price - ma_slow) / ma_slow) * 100
            confidence = min(0.85, 0.5 + (momentum_strength / 10))
            return {
                'signal': 'BUY',
                'confidence': confidence,
                'price': current_price,
                'stop_loss': ma_slow * 0.98,
                'target': current_price * 1.05,
                'strategy': 'MOMENTUM',
                'reason': f'MOMENTUM Bullish'
            }
        elif current_price < ma_slow and current_volume > avg_volume * self.config.MOMENTUM_VOLUME_MULTIPLIER and ma_fast < ma_slow:
            momentum_strength = ((ma_slow - current_price) / ma_slow) * 100
            confidence = min(0.85, 0.5 + (momentum_strength / 10))
            return {
                'signal': 'SELL',
                'confidence': confidence,
                'price': current_price,
                'stop_loss': ma_slow * 1.02,
                'target': current_price * 0.95,
                'strategy': 'MOMENTUM',
                'reason': f'MOMENTUM Bearish'
            }
        return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'MOMENTUM'}

    def analyze_breakout(self, prices: List[float], volumes: List[float]) -> Dict:
        """Breakout Trading"""
        lookback = self.config.BREAKOUT_LOOKBACK_BARS
        if len(prices) < lookback:
            return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'BREAKOUT'}
        
        recent_prices = prices[-lookback:]
        resistance = max(recent_prices)
        support = min(recent_prices)
        current_price = prices[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:])
        
        if current_price > resistance and current_volume > avg_volume * self.config.BREAKOUT_VOLUME_CONFIRMATION:
            return {
                'signal': 'BUY',
                'confidence': 0.8,
                'price': current_price,
                'stop_loss': support,
                'target': current_price + (resistance - support) * 1.5,
                'strategy': 'BREAKOUT',
                'reason': f'BREAKOUT Bullish'
            }
        elif current_price < support and current_volume > avg_volume * self.config.BREAKOUT_VOLUME_CONFIRMATION:
            return {
                'signal': 'SELL',
                'confidence': 0.8,
                'price': current_price,
                'stop_loss': resistance,
                'target': current_price - (resistance - support) * 1.5,
                'strategy': 'BREAKOUT',
                'reason': f'BREAKOUT Bearish'
            }
        return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'BREAKOUT'}

    def analyze_scalping(self, prices: List[float], volumes: List[float]) -> Dict:
        """Scalping"""
        if len(prices) < 5:
            return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'SCALPING'}
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:])
        
        if current_volume < self.config.SCALP_MIN_VOLUME:
            return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'SCALPING'}
        
        recent_high = max(prices[-5:])
        recent_low = min(prices[-5:])
        range_width = recent_high - recent_low
        
        if current_price <= recent_low + (range_width * 0.3) and current_volume > avg_volume:
            return {
                'signal': 'BUY',
                'confidence': 0.65,
                'price': current_price,
                'stop_loss': current_price - self.config.SCALP_SL_POINTS,
                'target': current_price + self.config.SCALP_TARGET_POINTS,
                'strategy': 'SCALPING',
                'reason': f'SCALP Low bounce'
            }
        elif current_price >= recent_high - (range_width * 0.3) and current_volume > avg_volume:
            return {
                'signal': 'SELL',
                'confidence': 0.65,
                'price': current_price,
                'stop_loss': current_price + self.config.SCALP_SL_POINTS,
                'target': current_price - self.config.SCALP_TARGET_POINTS,
                'strategy': 'SCALPING',
                'reason': f'SCALP High pullback'
            }
        return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'SCALPING'}

    def analyze_ma_crossover(self, prices: List[float]) -> Dict:
        """Moving Average Crossover"""
        if len(prices) < self.config.MA_SLOW_PERIOD:
            return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'MA_CROSSOVER'}
        
        current_price = prices[-1]
        ma_fast = np.mean(prices[-self.config.MA_FAST_PERIOD:])
        ma_slow = np.mean(prices[-self.config.MA_SLOW_PERIOD:])
        ma_fast_prev = np.mean(prices[-(self.config.MA_FAST_PERIOD+1):-1])
        ma_slow_prev = np.mean(prices[-(self.config.MA_SLOW_PERIOD+1):-1])
        
        if ma_fast_prev <= ma_slow_prev and ma_fast > ma_slow:
            return {
                'signal': 'BUY',
                'confidence': 0.75,
                'price': current_price,
                'stop_loss': ma_slow * 0.98,
                'target': current_price * 1.04,
                'strategy': 'MA_CROSSOVER',
                'reason': f'MA Golden cross'
            }
        elif ma_fast_prev >= ma_slow_prev and ma_fast < ma_slow:
            return {
                'signal': 'SELL',
                'confidence': 0.75,
                'price': current_price,
                'stop_loss': ma_slow * 1.02,
                'target': current_price * 0.96,
                'strategy': 'MA_CROSSOVER',
                'reason': f'MA Death cross'
            }
        return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'MA_CROSSOVER'}

    def analyze_pivot_points(self, prices: List[float], volumes: List[float],
                             prev_high: float, prev_low: float, prev_close: float) -> Dict:
        """Pivot Point Trading"""
        current_price = prices[-1]
        current_volume = volumes[-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        resistance1 = (2 * pivot) - prev_low
        support1 = (2 * pivot) - prev_high
        resistance2 = pivot + (prev_high - prev_low)
        support2 = pivot - (prev_high - prev_low)
        
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else current_volume
        
        if current_price > resistance1 and current_volume > avg_volume * 1.2:
            return {
                'signal': 'BUY',
                'confidence': 0.75,
                'price': current_price,
                'stop_loss': support1,
                'target': resistance2,
                'strategy': 'PIVOT_POINTS',
                'reason': f'PIVOT Above R1'
            }
        elif current_price < support1 and current_volume > avg_volume * 1.2:
            return {
                'signal': 'SELL',
                'confidence': 0.75,
                'price': current_price,
                'stop_loss': resistance1,
                'target': support2,
                'strategy': 'PIVOT_POINTS',
                'reason': f'PIVOT Below S1'
            }
        return {'signal': 'HOLD', 'confidence': 0, 'strategy': 'PIVOT_POINTS'}

    def ensemble_analysis(self, symbol: str, prices: List[float], volumes: List[float]) -> Dict:
        """
        Analyze all strategies and combine results intelligently
        Returns consensus signal with confidence
        """
        analyses = []
        
        # Run all strategy analyses
        analyses.append(self.analyze_orb(prices, volumes))
        analyses.append(self.analyze_news(prices, volumes))
        analyses.append(self.analyze_momentum(prices, volumes))
        analyses.append(self.analyze_breakout(prices, volumes))
        analyses.append(self.analyze_scalping(prices, volumes))
        analyses.append(self.analyze_ma_crossover(prices))
        
        prev_high = max(prices[-40:-20]) if len(prices) >= 40 else prices[-1] * 1.02
        prev_low = min(prices[-40:-20]) if len(prices) >= 40 else prices[-1] * 0.98
        prev_close = prices[-21] if len(prices) >= 21 else prices[-1]
        analyses.append(self.analyze_pivot_points(prices, volumes, prev_high, prev_low, prev_close))
        
        # Count signals
        buy_signals = [a for a in analyses if a['signal'] == 'BUY']
        sell_signals = [a for a in analyses if a['signal'] == 'SELL']
        hold_signals = [a for a in analyses if a['signal'] == 'HOLD']
        
        buy_confidence = np.mean([a['confidence'] for a in buy_signals]) if buy_signals else 0
        sell_confidence = np.mean([a['confidence'] for a in sell_signals]) if sell_signals else 0
        
        # Ensemble voting
        if len(buy_signals) >= self.config.ENSEMBLE_VOTING_THRESHOLD and buy_confidence >= self.config.STRATEGY_CONFIDENCE_THRESHOLD:
            # Strong BUY consensus
            best_buy = max(buy_signals, key=lambda x: x['confidence'])
            return {
                'signal': 'BUY',
                'confidence': min(1.0, buy_confidence + (len(buy_signals) * 0.05)),  # Boost by number agreeing
                'price': prices[-1],
                'stop_loss': np.mean([a.get('stop_loss', prices[-1] * 0.98) for a in buy_signals]),
                'target': np.mean([a.get('target', prices[-1] * 1.02) for a in buy_signals]),
                'strategy': 'ENSEMBLE',
                'reason': f'Ensemble consensus: {len(buy_signals)}/7 strategies voting BUY',
                'voting': {
                    'buy_votes': len(buy_signals),
                    'sell_votes': len(sell_signals),
                    'hold_votes': len(hold_signals),
                    'buy_strategies': [a['strategy'] for a in buy_signals],
                    'sell_strategies': [a['strategy'] for a in sell_signals]
                },
                'all_signals': analyses
            }
        
        elif len(sell_signals) >= self.config.ENSEMBLE_VOTING_THRESHOLD and sell_confidence >= self.config.STRATEGY_CONFIDENCE_THRESHOLD:
            # Strong SELL consensus
            best_sell = max(sell_signals, key=lambda x: x['confidence'])
            return {
                'signal': 'SELL',
                'confidence': min(1.0, sell_confidence + (len(sell_signals) * 0.05)),
                'price': prices[-1],
                'stop_loss': np.mean([a.get('stop_loss', prices[-1] * 1.02) for a in sell_signals]),
                'target': np.mean([a.get('target', prices[-1] * 0.98) for a in sell_signals]),
                'strategy': 'ENSEMBLE',
                'reason': f'Ensemble consensus: {len(sell_signals)}/7 strategies voting SELL',
                'voting': {
                    'buy_votes': len(buy_signals),
                    'sell_votes': len(sell_signals),
                    'hold_votes': len(hold_signals),
                    'buy_strategies': [a['strategy'] for a in buy_signals],
                    'sell_strategies': [a['strategy'] for a in sell_signals]
                },
                'all_signals': analyses
            }
        
        else:
            # No consensus - HOLD
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'strategy': 'ENSEMBLE',
                'reason': f'No consensus: B={len(buy_signals)}, S={len(sell_signals)}, H={len(hold_signals)}',
                'voting': {
                    'buy_votes': len(buy_signals),
                    'sell_votes': len(sell_signals),
                    'hold_votes': len(hold_signals),
                    'buy_strategies': [a['strategy'] for a in buy_signals],
                    'sell_strategies': [a['strategy'] for a in sell_signals]
                },
                'all_signals': analyses
            }

    def update_strategy_scores(self, strategy: str, won: bool):
        """Update strategy performance tracking"""
        if strategy == 'ENSEMBLE':
            return
        if won:
            self.strategy_scores[strategy]['wins'] += 1
        else:
            self.strategy_scores[strategy]['losses'] += 1
        
        total = self.strategy_scores[strategy]['wins'] + self.strategy_scores[strategy]['losses']
        if total > 0:
            self.strategy_scores[strategy]['score'] = self.strategy_scores[strategy]['wins'] / total

class RiskManager:
    def __init__(self, config: Config):
        self.config = config
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 20

    def can_trade(self) -> tuple:
        if self.daily_pnl <= -self.config.CAPITAL * self.config.MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"
        if self.daily_trades >= self.max_daily_trades:
            return False, "Max daily trades reached"
        return True, "OK"

    def calculate_position_size(self, entry: float, stop_loss: float) -> int:
        risk_amount = self.config.CAPITAL * self.config.RISK_PER_TRADE
        risk_per_share = abs(entry - stop_loss)
        if risk_per_share == 0:
            return 0
        quantity = int(risk_amount / risk_per_share)
        max_by_capital = int((self.config.CAPITAL * 0.2) / entry)
        return min(quantity, max_by_capital, 100)

    def validate_risk_reward(self, entry: float, stop_loss: float, target: float) -> bool:
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
    def __init__(self, config: Config):
        self.config = config

    def is_market_open(self) -> bool:
    """Check if NSE market is currently open"""
    now = datetime.now()
    current_time = now.time()
    
    # Check if weekday
    if now.weekday() >= 5:
        return False
    
    # Market hours
    market_start = dtime(9, 15, 0)
    lunch_start = dtime(11, 40, 0)
    lunch_end = dtime(12, 30, 0)
    market_end = dtime(15, 30, 0)
    
    # Before market open
    if current_time < market_start:
        return False
    
    # After market close (FIXED: Using >= instead of >)
    if current_time >= market_end:
        return False
    
    # During lunch break
    if lunch_start <= current_time < lunch_end:
        return False
    
    return True

    def should_avoid_trading(self) -> tuple:
        now = datetime.now()
        current_time = now.time()
        avoid_until = (datetime.combine(now.date(), self.config.MARKET_OPEN) +
            timedelta(minutes=self.config.AVOID_OPENING_MINUTES)).time()
        if current_time < avoid_until:
            return True, f"Avoiding first {self.config.AVOID_OPENING_MINUTES} mins"
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
        self.analyzer = EnsembleStrategyAnalyzer(config)
        
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self._init_price_data()

    def _init_price_data(self):
        base_prices = {
            'RELIANCE-EQ': 2450, 'TCS-EQ': 3800, 'INFY-EQ': 1450,
            'HDFCBANK-EQ': 1650, 'ICICIBANK-EQ': 1050, 'SBIN-EQ': 620,
            'BHARTIARTL-EQ': 1150, 'ITC-EQ': 440, 'KOTAKBANK-EQ': 1750, 'LT-EQ': 3200
        }
        for symbol in self.config.WATCHLIST:
            base = base_prices.get(symbol, 1000)
            prices = [base]
            volumes = [random.randint(100000, 500000)]
            for _ in range(49):
                change = random.gauss(0, 0.01)
                prices.append(prices[-1] * (1 + change))
                volumes.append(random.randint(100000, 500000))
            self.price_history[symbol] = prices
            self.volume_history[symbol] = volumes

    def _update_prices(self):
        for symbol in self.price_history:
            change = random.gauss(0, 0.005)
            new_price = self.price_history[symbol][-1] * (1 + change)
            self.price_history[symbol].append(new_price)
            self.volume_history[symbol].append(random.randint(100000, 500000))
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
                self.volume_history[symbol] = self.volume_history[symbol][-100:]

    def analyze_stock_ensemble(self, symbol: str) -> Dict:
        """Get ensemble analysis for stock"""
        prices = self.price_history.get(symbol, [])
        volumes = self.volume_history.get(symbol, [])
        if len(prices) < 30:
            return {'signal': 'HOLD', 'confidence': 0}
        
        analysis = self.analyzer.ensemble_analysis(symbol, prices, volumes)
        return analysis

    def place_order(self, symbol: str, transaction_type: str, quantity: int, price: float) -> Dict:
        if quantity <= 0:
            return {'order_id': None, 'status': 'REJECTED', 'error': 'Invalid quantity'}
        
        order_cost = price * quantity
        if transaction_type == 'BUY' and order_cost > self.capital:
            return {'order_id': None, 'status': 'REJECTED', 'error': 'Insufficient capital'}
        
        order_id = f"PAPER_{len(self.orders) + 1}_{datetime.now().strftime('%H%M%S')}"
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'transaction_type': transaction_type,
            'quantity': quantity,
            'price': price,
            'status': 'EXECUTED',
            'timestamp': datetime.now().isoformat(),
            'order_cost': round(order_cost, 2)
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
                analysis = self.analyze_stock_ensemble(symbol)
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_time': datetime.now().isoformat(),
                    'stop_loss': analysis.get('stop_loss', price * 0.98),
                    'target': analysis.get('target', price * 1.02),
                    'strategy': 'ENSEMBLE'
                }
            
            old_capital = self.capital
            self.capital -= order_cost
            logger.info(f"✅ ENSEMBLE BUY {symbol}: ₹{old_capital:.2f} → ₹{self.capital:.2f}")
            
        elif transaction_type == 'SELL':
            if symbol not in self.positions:
                return {'order_id': None, 'status': 'REJECTED', 'error': 'No open position'}
            
            pos = self.positions[symbol]
            entry_price = pos['avg_price']
            pnl = (price - entry_price) * quantity
            self.capital += order_cost
            self.daily_pnl += pnl
            self.risk_manager.update_daily_pnl(pnl)
            
            won = pnl > 0
            self.analyzer.update_strategy_scores('ENSEMBLE', won)
            
            self.trades.append({
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': price,
                'quantity': quantity,
                'pnl': round(pnl, 2),
                'entry_time': pos['entry_time'],
                'exit_time': datetime.now().isoformat(),
                'strategy': 'ENSEMBLE',
                'mode': 'paper'
            })
            
            pos['quantity'] -= quantity
            if pos['quantity'] <= 0:
                del self.positions[symbol]
            
            logger.info(f"✅ ENSEMBLE SELL {symbol}: P&L ₹{pnl:.2f}")
        
        return order

    def get_positions(self) -> List[Dict]:
        return [
            {
                'symbol': symbol,
                'quantity': data['quantity'],
                'avg_price': data['avg_price'],
                'strategy': 'ENSEMBLE',
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
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            current_price = self.price_history.get(symbol, [pos['avg_price']])[-1]
            if current_price <= pos.get('stop_loss', 0):
                logger.info(f"🛑 SL hit: {symbol} @ ₹{current_price:.2f}")
                self.place_order(symbol, 'SELL', pos['quantity'], current_price)
            elif current_price >= pos.get('target', float('inf')):
                logger.info(f"🎯 TARGET hit: {symbol} @ ₹{current_price:.2f}")
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

class AutoTradingBot:
    """Autonomous bot using ensemble strategy"""
    def __init__(self, paper_engine: PaperTradingEngine, config: Config):
        self.paper_engine = paper_engine
        self.config = config
        self.running = False

    def start(self):
        self.running = True
        logger.info("🤖 Autonomous Ensemble Trading Bot started!")
        logger.info("📊 Using all 7 strategies with intelligent voting")
        self._monitor_loop()

    def stop(self):
        self.running = False
        for symbol in list(self.paper_engine.positions.keys()):
            pos = self.paper_engine.positions[symbol]
            price = self.paper_engine.price_history.get(symbol, [pos['avg_price']])[-1]
            self.paper_engine.place_order(symbol, 'SELL', pos['quantity'], price)
        logger.info("🛑 Bot stopped - all positions closed")

    def _monitor_loop(self):
        import time
        while self.running:
            try:
                self.paper_engine._update_prices()
                self.paper_engine.check_stop_loss_target()
                
                can_trade, reason = self.paper_engine.risk_manager.can_trade()
                avoid, avoid_reason = self.paper_engine.time_manager.should_avoid_trading()
                
                if not can_trade:
                    time.sleep(30)
                    continue
                
                for symbol in self.config.WATCHLIST:
                    analysis = self.paper_engine.analyze_stock_ensemble(symbol)
                    self.paper_engine.signals.append(analysis)
                    
                    if avoid:
                        continue
                    
                    # Execute on ensemble signal
                    if analysis['signal'] == 'BUY' and analysis.get('confidence', 0) >= self.config.STRATEGY_CONFIDENCE_THRESHOLD:
                        if len(self.paper_engine.positions) >= self.config.MAX_POSITIONS:
                            continue
                        if symbol in self.paper_engine.positions:
                            continue
                        
                        required_capital = analysis['price'] * 1
                        if required_capital > self.paper_engine.capital * 0.8:
                            continue
                        
                        if not self.paper_engine.risk_manager.validate_risk_reward(
                            analysis['price'], analysis.get('stop_loss', analysis['price'] * 0.98),
                            analysis.get('target', analysis['price'] * 1.02)
                        ):
                            continue
                        
                        qty = self.paper_engine.risk_manager.calculate_position_size(
                            analysis['price'], analysis.get('stop_loss', analysis['price'] * 0.98)
                        )
                        
                        if qty > 0 and qty * analysis['price'] <= self.paper_engine.capital * 0.9:
                            order = self.paper_engine.place_order(symbol, 'BUY', qty, analysis['price'])
                            if order['status'] == 'EXECUTED':
                                voting_info = analysis.get('voting', {})
                                logger.info(f"✅ BUY {symbol}: {voting_info.get('buy_votes', 0)}/7 strategies agreed | Confidence: {analysis['confidence']:.2f}")
                    
                    elif analysis['signal'] == 'SELL' and symbol in self.paper_engine.positions:
                        pos = self.paper_engine.positions[symbol]
                        self.paper_engine.place_order(symbol, 'SELL', pos['quantity'], analysis['price'])
                        voting_info = analysis.get('voting', {})
                        logger.info(f"✅ SELL {symbol}: {voting_info.get('sell_votes', 0)}/7 strategies agreed")
                
                time.sleep(10)
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                time.sleep(30)

config = Config()
paper_engine = PaperTradingEngine(config)

# ==================== API ROUTES ====================

@app.route('/')
def index():
    return jsonify({
        'name': 'Autonomous Ensemble Trading Bot v5.0',
        'version': '5.0.0',
        'mode': 'ENSEMBLE - All 7 Strategies Combined',
        'strategies_count': 7
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    time_mgr = MarketTimeManager(config)
    return jsonify({
        'status': 'healthy',
        'mode': 'ENSEMBLE AUTONOMOUS',
        'market_status': time_mgr.get_market_status(),
        'bot_running': trading_bot.running if trading_bot else False,
        'ensemble_mode': True,
        'voting_threshold': config.ENSEMBLE_VOTING_THRESHOLD
    })

@app.route('/api/ensemble/config', methods=['GET'])
def get_ensemble_config():
    return jsonify({
        'success': True,
        'ensemble_settings': {
            'mode': 'AUTONOMOUS - All Strategies Combined',
            'strategies': StrategyEnum.ALL,
            'voting_threshold': config.ENSEMBLE_VOTING_THRESHOLD,
            'confidence_threshold': config.STRATEGY_CONFIDENCE_THRESHOLD,
            'auto_strategy_selection': True,
            'description': 'Bot automatically combines all 7 strategies and trades on consensus'
        },
        'strategy_performance': paper_engine.analyzer.strategy_scores
    })

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    global trading_bot, bot_thread, paper_engine
    if trading_bot and trading_bot.running:
        return jsonify({'success': False, 'error': 'Already running'}), 400
    
    trading_bot = AutoTradingBot(paper_engine, config)
    bot_thread = threading.Thread(target=trading_bot.start, daemon=True)
    bot_thread.start()
    
    return jsonify({'success': True, 'message': 'Autonomous Ensemble Bot started!', 'mode': 'All 7 strategies active'})

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    global trading_bot
    if not trading_bot or not trading_bot.running:
        return jsonify({'success': False, 'error': 'Not running'}), 400
    
    trading_bot.stop()
    return jsonify({'success': True, 'message': 'Bot stopped'})

@app.route('/api/bot/status', methods=['GET'])
def bot_status():
    time_mgr = MarketTimeManager(config)
    analyzer = paper_engine.analyzer
    return jsonify({
        'running': trading_bot.running if trading_bot else False,
        'mode': 'ENSEMBLE AUTONOMOUS',
        'market_status': time_mgr.get_market_status(),
        'positions_count': len(paper_engine.positions),
        'daily_pnl': round(paper_engine.daily_pnl, 2),
        'daily_trades': paper_engine.risk_manager.daily_trades,
        'strategy_scores': analyzer.strategy_scores,
        'voting_threshold': config.ENSEMBLE_VOTING_THRESHOLD,
        'strategies_active': 7
    })

@app.route('/api/signals', methods=['GET'])
def get_signals():
    return jsonify({'success': True, 'data': paper_engine.signals[-50:]})

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
            'avg_loss': round(sum(t['pnl'] for t in losing) / len(losing), 2) if losing else 0,
            'strategy_used': 'ENSEMBLE (All 7 strategies)'
        }
    })

@app.route('/api/positions', methods=['GET'])
def get_positions():
    return jsonify({'success': True, 'data': paper_engine.get_positions()})

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    total_value = paper_engine.get_portfolio_value()
    unrealized_pnl = 0
    
    for symbol, pos in paper_engine.positions.items():
        current_price = paper_engine.price_history.get(symbol, [pos['avg_price']])[-1]
        unrealized_pnl += (current_price - pos['avg_price']) * pos['quantity']
    
    total_pnl = paper_engine.daily_pnl + unrealized_pnl
    
    return jsonify({
        'success': True,
        'data': {
            'initial_capital': round(config.CAPITAL, 2),
            'total_value': round(total_value, 2),
            'cash': round(paper_engine.capital, 2),
            'invested_value': round(total_value - paper_engine.capital, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'realized_pnl': round(paper_engine.daily_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'daily_pnl_pct': round((total_pnl / config.CAPITAL * 100), 2),
            'total_return_pct': round((total_pnl / config.CAPITAL * 100), 2),
            'positions_count': len(paper_engine.positions),
            'trades_count': len(paper_engine.trades),
            'trading_mode': 'ENSEMBLE AUTONOMOUS'
        }
    })

@app.route('/api/paper/reset', methods=['POST'])
def reset_paper():
    if trading_bot and trading_bot.running:
        return jsonify({'success': False, 'error': 'Stop bot first'}), 400
    paper_engine.reset()
    return jsonify({'success': True, 'message': 'Reset complete'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Autonomous Ensemble Trading Bot v5.0 on port {port}")
    logger.info("📊 ENSEMBLE MODE: All 7 strategies combined with intelligent voting")
    logger.info("🤖 BOT AUTONOMOUSLY TRADES ON CONSENSUS")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
