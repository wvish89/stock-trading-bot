"""
Indian Stock Trading Bot - Complete System
Version: 2.0.0 (Cloud Ready)
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

# API Integration
try:
    from SmartApi import SmartConnect
except ImportError:
    from smartapi import SmartConnect
import pyotp

# Notifications
import requests

# Database
import sqlite3

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


# ==================== ENUMS AND DATA CLASSES ====================

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "STOPLOSS"
    SL_M = "STOPLOSS_MARKET"

class TransactionType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class ProductType(Enum):
    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    symbol: str
    signal: SignalType
    price: float
    timestamp: datetime
    indicators: Dict
    confidence: float
    stop_loss: float
    target: float


# ==================== CONFIGURATION ====================

class Config:
    """Cloud-ready configuration loader"""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """Load from environment variables or config.json"""
        config_data = {}
        if os.path.exists('config.json'):
            try:
                with open('config.json', 'r') as f:
                    config_data = json.load(f)
            except:
                pass
        
        # API credentials
        self.API_KEY = os.getenv('ANGEL_API_KEY', config_data.get('api_key', ''))
        self.CLIENT_ID = os.getenv('ANGEL_CLIENT_ID', config_data.get('client_id', ''))
        self.PASSWORD = os.getenv('ANGEL_PASSWORD', config_data.get('password', ''))
        self.TOTP_SECRET = os.getenv('ANGEL_TOTP_SECRET', config_data.get('totp_secret', ''))
        
        # Trading parameters
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', config_data.get('max_positions', 5)))
        self.RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', config_data.get('risk_per_trade', 0.02)))
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', config_data.get('max_daily_loss', 0.05)))
        self.CAPITAL = float(os.getenv('TRADING_CAPITAL', config_data.get('capital', 100000)))
        
        # Strategy parameters
        self.RSI_PERIOD = int(os.getenv('RSI_PERIOD', config_data.get('rsi_period', 14)))
        self.RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', config_data.get('rsi_overbought', 70)))
        self.RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', config_data.get('rsi_oversold', 30)))
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # Notifications
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', config_data.get('telegram_bot_token', ''))
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', config_data.get('telegram_chat_id', ''))
        
        # Watchlist
        watchlist_env = os.getenv('WATCHLIST')
        if watchlist_env:
            self.WATCHLIST = watchlist_env.split(',')
        else:
            self.WATCHLIST = config_data.get('watchlist', ['RELIANCE-EQ', 'TCS-EQ', 'INFY-EQ', 'HDFCBANK-EQ', 'SBIN-EQ'])
        
        self.SQUARE_OFF_TIME = os.getenv('SQUARE_OFF_TIME', config_data.get('square_off_time', '15:15'))


# ==================== TECHNICAL INDICATORS ====================

class TechnicalIndicators:
    """Technical indicators without TA-Lib dependency"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return np.array([50] * len(prices))
        
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        result = np.zeros(len(prices))
        result[1:] = rsi.values
        result[0] = 50
        return result
    
    @staticmethod
    def calculate_ema(prices, period):
        """Calculate EMA"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def calculate_sma(prices, period):
        """Calculate SMA"""
        return pd.Series(prices).rolling(window=period).mean().values
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = TechnicalIndicators.calculate_sma(prices, period)
        std = pd.Series(prices).rolling(window=period).std().values
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Calculate ATR"""
        tr1 = high - low
        tr2 = abs(high - np.roll(close, 1))
        tr3 = abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period).mean().values
        return atr
    
    @staticmethod
    def calculate_supertrend(df, period=10, multiplier=3.0):
        """Calculate Supertrend"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        supertrend = np.zeros(len(df))
        direction = np.ones(len(df))
        
        for i in range(1, len(df)):
            if close[i] > upperband[i-1]:
                direction[i] = 1
            elif close[i] < lowerband[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            supertrend[i] = lowerband[i] if direction[i] == 1 else upperband[i]
        
        return supertrend, direction
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX"""
        plus_dm = np.maximum(np.diff(high, prepend=high[0]), 0)
        minus_dm = np.maximum(-np.diff(low, prepend=low[0]), 0)
        
        tr = TechnicalIndicators.calculate_atr(high, low, close, 1)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / (pd.Series(tr).rolling(period).mean() + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / (pd.Series(tr).rolling(period).mean() + 1e-10)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean().values
        
        return adx


# ==================== BROKER API ====================

class BrokerAPI:
    """Angel Broking SmartAPI integration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.smart_api = None
        self.connected = False
        self.connect()
    
    def connect(self):
        """Connect to Angel Broking API"""
        if not self.config.API_KEY or not self.config.TOTP_SECRET:
            logger.warning("API credentials not configured")
            return False
        
        try:
            self.smart_api = SmartConnect(api_key=self.config.API_KEY)
            totp = pyotp.TOTP(self.config.TOTP_SECRET).now()
            
            data = self.smart_api.generateSession(
                self.config.CLIENT_ID,
                self.config.PASSWORD,
                totp
            )
            
            if data.get('status'):
                self.connected = True
                logger.info("Connected to Angel Broking API")
                return True
            else:
                logger.error(f"Login failed: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def get_ltp(self, symbol: str, exchange: str = "NSE") -> Optional[float]:
        """Get Last Traded Price"""
        if not self.connected:
            return None
        try:
            data = self.smart_api.ltpData(exchange, symbol, symbol)
            if data.get('status'):
                return float(data['data']['ltp'])
            return None
        except Exception as e:
            logger.error(f"Error fetching LTP: {e}")
            return None
    
    def get_historical_data(self, symbol: str, exchange: str, interval: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        """Fetch OHLC historical data"""
        if not self.connected:
            return None
        try:
            params = {
                "exchange": exchange,
                "symboltoken": symbol,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            data = self.smart_api.getCandleData(params)
            
            if data.get('status'):
                df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            return None
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def place_order(self, symbol: str, exchange: str, transaction_type: TransactionType, quantity: int, order_type: OrderType, price: float = 0, product_type: ProductType = ProductType.INTRADAY) -> Optional[str]:
        """Place an order"""
        if not self.connected:
            return None
        try:
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": symbol,
                "transactiontype": transaction_type.value,
                "exchange": exchange,
                "ordertype": order_type.value,
                "producttype": product_type.value,
                "duration": "DAY",
                "price": price,
                "quantity": quantity
            }
            
            response = self.smart_api.placeOrder(order_params)
            
            if response.get('status'):
                order_id = response['data']['orderid']
                logger.info(f"Order placed: {order_id}")
                return order_id
            return None
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.connected:
            return []
        try:
            data = self.smart_api.position()
            return data.get('data', []) if data.get('status') else []
        except:
            return []


# ==================== TRADING STRATEGY ====================

class TradingStrategy:
    """Main trading strategy"""
    
    def __init__(self, config: Config):
        self.config = config
        self.indicators = TechnicalIndicators()
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Analyze and generate trading signal"""
        if len(df) < 50:
            return self._hold_signal(df, symbol)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Calculate indicators
        rsi = self.indicators.calculate_rsi(close, self.config.RSI_PERIOD)
        macd, macd_signal, macd_hist = self.indicators.calculate_macd(close)
        supertrend, st_direction = self.indicators.calculate_supertrend(df)
        ema_20 = self.indicators.calculate_ema(close, 20)
        ema_50 = self.indicators.calculate_ema(close, 50)
        adx = self.indicators.calculate_adx(high, low, close)
        
        current_price = close[-1]
        
        # Get latest values
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
        current_macd = macd[-1] if not np.isnan(macd[-1]) else 0
        current_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
        current_st = st_direction[-1] if not np.isnan(st_direction[-1]) else 0
        current_adx = adx[-1] if not np.isnan(adx[-1]) else 20
        
        indicators_dict = {
            'rsi': round(current_rsi, 2),
            'macd': round(current_macd, 4),
            'macd_signal': round(current_macd_signal, 4),
            'supertrend': int(current_st),
            'adx': round(current_adx, 2),
            'ema_20': round(ema_20[-1], 2),
            'ema_50': round(ema_50[-1], 2)
        }
        
        # Signal logic
        buy_signals = 0
        sell_signals = 0
        
        # RSI
        if current_rsi < self.config.RSI_OVERSOLD:
            buy_signals += 1
        elif current_rsi > self.config.RSI_OVERBOUGHT:
            sell_signals += 1
        
        # MACD
        if current_macd > current_macd_signal and macd[-2] <= macd_signal[-2]:
            buy_signals += 1
        elif current_macd < current_macd_signal and macd[-2] >= macd_signal[-2]:
            sell_signals += 1
        
        # Supertrend
        if current_st == 1:
            buy_signals += 1
        elif current_st == -1:
            sell_signals += 1
        
        # EMA crossover
        if ema_20[-1] > ema_50[-1] and ema_20[-2] <= ema_50[-2]:
            buy_signals += 1
        elif ema_20[-1] < ema_50[-1] and ema_20[-2] >= ema_50[-2]:
            sell_signals += 1
        
        # ADX trend strength
        trend_strong = current_adx > 25
        
        # Determine signal
        confidence = max(buy_signals, sell_signals) / 4.0
        
        if buy_signals >= 3 and trend_strong:
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                price=current_price,
                timestamp=datetime.now(),
                indicators=indicators_dict,
                confidence=confidence,
                stop_loss=current_price * 0.98,
                target=current_price * 1.04
            )
        elif sell_signals >= 3 and trend_strong:
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                price=current_price,
                timestamp=datetime.now(),
                indicators=indicators_dict,
                confidence=confidence,
                stop_loss=current_price * 1.02,
                target=current_price * 0.96
            )
        
        return self._hold_signal(df, symbol, indicators_dict)
    
    def _hold_signal(self, df, symbol, indicators=None):
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.HOLD,
            price=df['close'].iloc[-1] if len(df) > 0 else 0,
            timestamp=datetime.now(),
            indicators=indicators or {},
            confidence=0.0,
            stop_loss=0.0,
            target=0.0
        )


# ==================== RISK MANAGEMENT ====================

class RiskManager:
    """Risk management system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.daily_pnl = 0.0
        self.positions_count = 0
    
    def can_take_position(self) -> bool:
        if self.positions_count >= self.config.MAX_POSITIONS:
            return False
        if self.daily_pnl <= -self.config.MAX_DAILY_LOSS * self.config.CAPITAL:
            return False
        return True
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        risk_amount = self.config.CAPITAL * self.config.RISK_PER_TRADE
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0
        return max(1, int(risk_amount / risk_per_share))
    
    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl
        logger.info(f"Daily P&L: ₹{self.daily_pnl:.2f}")
    
    def reset_daily(self):
        self.daily_pnl = 0.0


# ==================== NOTIFICATIONS ====================

class NotificationManager:
    """Send notifications via Telegram"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def send_telegram(self, message: str):
        if not self.config.TELEGRAM_BOT_TOKEN or not self.config.TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": self.config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    def notify_signal(self, signal: TradingSignal):
        message = f"""
🤖 <b>Trading Signal</b>

📊 Symbol: {signal.symbol}
🎯 Signal: {signal.signal.value}
💰 Price: ₹{signal.price:.2f}
📈 Confidence: {signal.confidence*100:.1f}%
🛡️ Stop Loss: ₹{signal.stop_loss:.2f}
🎯 Target: ₹{signal.target:.2f}

RSI: {signal.indicators.get('rsi', 0)}
ADX: {signal.indicators.get('adx', 0)}
"""
        self.send_telegram(message)


# ==================== DATABASE ====================

class DatabaseManager:
    """Database management"""
    
    def __init__(self, db_file: str = "trading_bot.db"):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                pnl REAL,
                strategy TEXT,
                mode TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                signal_type TEXT,
                price REAL,
                timestamp TIMESTAMP,
                confidence REAL,
                indicators TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade_data: Dict):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, entry_time, exit_time, entry_price, exit_price, quantity, pnl, strategy, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['symbol'],
            trade_data.get('entry_time'),
            trade_data.get('exit_time'),
            trade_data['entry_price'],
            trade_data.get('exit_price'),
            trade_data['quantity'],
            trade_data.get('pnl', 0),
            trade_data.get('strategy', 'Multi-Indicator'),
            trade_data.get('mode', 'paper')
        ))
        conn.commit()
        conn.close()
    
    def save_signal(self, signal: TradingSignal):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO signals (symbol, signal_type, price, timestamp, confidence, indicators)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            signal.symbol,
            signal.signal.value,
            signal.price,
            signal.timestamp,
            signal.confidence,
            json.dumps(signal.indicators)
        ))
        conn.commit()
        conn.close()
    
    def get_trade_history(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_file)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY exit_time DESC LIMIT 100", conn)
        conn.close()
        return df
    
    def get_signals(self, limit=50) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_file)
        df = pd.read_sql_query(f"SELECT * FROM signals ORDER BY timestamp DESC LIMIT {limit}", conn)
        conn.close()
        return df


# ==================== MAIN TRADING BOT ====================

class TradingBot:
    """Main trading bot"""
    
    def __init__(self):
        self.config = Config()
        self.broker = BrokerAPI(self.config)
        self.strategy = TradingStrategy(self.config)
        self.risk_manager = RiskManager(self.config)
        self.notifier = NotificationManager(self.config)
        self.db = DatabaseManager()
        
        self.running = False
        self.positions = {}
    
    def start(self):
        logger.info("=" * 50)
        logger.info("STARTING TRADING BOT")
        logger.info("=" * 50)
        
        self.running = True
        
        monitor_thread = threading.Thread(target=self.monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        square_off_thread = threading.Thread(target=self.check_square_off)
        square_off_thread.daemon = True
        square_off_thread.start()
        
        logger.info("Trading bot is running...")
    
    def stop(self):
        self.running = False
        self.square_off_all_positions()
        logger.info("Trading bot stopped")
    
    def monitor_loop(self):
        while self.running:
            try:
                for symbol in self.config.WATCHLIST:
                    self.process_symbol(symbol)
                time.sleep(60)
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(60)
    
    def process_symbol(self, symbol: str):
        try:
            to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M")
            
            df = self.broker.get_historical_data(symbol, "NSE", "FIVE_MINUTE", from_date, to_date)
            
            if df is None or len(df) < 50:
                return
            
            signal = self.strategy.analyze(df, symbol)
            self.db.save_signal(signal)
            
            if signal.signal == SignalType.BUY and self.risk_manager.can_take_position():
                self.execute_buy(signal)
            elif signal.signal == SignalType.SELL and symbol in self.positions:
                self.execute_sell(signal)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def execute_buy(self, signal: TradingSignal):
        quantity = self.risk_manager.calculate_position_size(signal.price, signal.stop_loss)
        if quantity <= 0:
            return
        
        order_id = self.broker.place_order(
            symbol=signal.symbol,
            exchange="NSE",
            transaction_type=TransactionType.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
        
        if order_id:
            self.positions[signal.symbol] = {
                'quantity': quantity,
                'entry_price': signal.price,
                'stop_loss': signal.stop_loss,
                'target': signal.target,
                'entry_time': datetime.now()
            }
            self.risk_manager.positions_count += 1
            self.notifier.notify_signal(signal)
            logger.info(f"BUY executed: {signal.symbol} @ ₹{signal.price}")
    
    def execute_sell(self, signal: TradingSignal):
        position = self.positions.get(signal.symbol)
        if not position:
            return
        
        order_id = self.broker.place_order(
            symbol=signal.symbol,
            exchange="NSE",
            transaction_type=TransactionType.SELL,
            quantity=position['quantity'],
            order_type=OrderType.MARKET
        )
        
        if order_id:
            pnl = (signal.price - position['entry_price']) * position['quantity']
            self.risk_manager.update_daily_pnl(pnl)
            
            self.db.save_trade({
                'symbol': signal.symbol,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'entry_price': position['entry_price'],
                'exit_price': signal.price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'mode': 'live'
            })
            
            del self.positions[signal.symbol]
            self.risk_manager.positions_count -= 1
            logger.info(f"SELL executed: {signal.symbol} @ ₹{signal.price}, P&L: ₹{pnl:.2f}")
    
    def check_square_off(self):
        while self.running:
            try:
                current_time = datetime.now().strftime("%H:%M")
                if current_time >= self.config.SQUARE_OFF_TIME:
                    self.square_off_all_positions()
                    time.sleep(3600)
                time.sleep(60)
            except Exception as e:
                logger.error(f"Square-off check error: {e}")
                time.sleep(60)
    
    def square_off_all_positions(self):
        logger.info("Squaring off all positions...")
        for symbol in list(self.positions.keys()):
            try:
                position = self.positions[symbol]
                ltp = self.broker.get_ltp(symbol) or position['entry_price']
                
                self.broker.place_order(
                    symbol=symbol,
                    exchange="NSE",
                    transaction_type=TransactionType.SELL,
                    quantity=position['quantity'],
                    order_type=OrderType.MARKET
                )
                
                pnl = (ltp - position['entry_price']) * position['quantity']
                self.risk_manager.update_daily_pnl(pnl)
                
                self.db.save_trade({
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'entry_price': position['entry_price'],
                    'exit_price': ltp,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'mode': 'live'
                })
                
                del self.positions[symbol]
                self.risk_manager.positions_count -= 1
                logger.info(f"Squared off {symbol}, P&L: ₹{pnl:.2f}")
            except Exception as e:
                logger.error(f"Error squaring off {symbol}: {e}")


# ==================== BACKTESTER ====================

class Backtester:
    """Backtest trading strategies"""
    
    def __init__(self, config: Config):
        self.config = config
        self.strategy = TradingStrategy(config)
    
    def run_backtest(self, df: pd.DataFrame, symbol: str, initial_capital: float = 100000) -> Dict:
        capital = initial_capital
        position = None
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(50, len(df)):
            current_df = df.iloc[:i+1].copy()
            signal = self.strategy.analyze(current_df, symbol)
            current_price = df['close'].iloc[i]
            
            if signal.signal == SignalType.BUY and position is None:
                quantity = int(capital * 0.95 / current_price)
                if quantity > 0:
                    position = {
                        'entry_price': current_price,
                        'entry_time': df['timestamp'].iloc[i],
                        'quantity': quantity,
                        'stop_loss': signal.stop_loss,
                        'target': signal.target
                    }
            
            elif position is not None:
                exit_triggered = False
                exit_reason = ""
                
                if current_price <= position['stop_loss']:
                    exit_triggered = True
                    exit_reason = "Stop Loss"
                elif current_price >= position['target']:
                    exit_triggered = True
                    exit_reason = "Target"
                elif signal.signal == SignalType.SELL:
                    exit_triggered = True
                    exit_reason = "Sell Signal"
                
                if exit_triggered:
                    pnl = (current_price - position['entry_price']) * position['quantity']
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': df['timestamp'].iloc[i],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    })
                    position = None
            
            equity_curve.append(capital)
        
        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            winning = trades_df[trades_df['pnl'] > 0]
            losing = trades_df[trades_df['pnl'] < 0]
            
            total_profit = winning['pnl'].sum() if len(winning) > 0 else 0
            total_loss = abs(losing['pnl'].sum()) if len(losing) > 0 else 0
            
            metrics = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning),
                'losing_trades': len(losing),
                'win_rate': len(winning) / len(trades_df) * 100,
                'total_pnl': capital - initial_capital,
                'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
                'final_capital': capital,
                'return_pct': (capital - initial_capital) / initial_capital * 100
            }
        else:
            metrics = {'total_trades': 0, 'error': 'No trades generated'}
        
        return {'metrics': metrics, 'trades': trades, 'equity_curve': equity_curve}


if __name__ == "__main__":
    print("Trading Bot Module - Import this in app.py")
    config = Config()
    print(f"API Key configured: {'Yes' if config.API_KEY else 'No'}")
    print(f"Watchlist: {config.WATCHLIST}")
