"""
Enhanced Real-Time Data Manager for Indian Markets
Supports: WebSocket feeds, Smart caching, Market regime detection
"""

import asyncio
import websockets
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketTick:
    """Real-time market tick"""
    symbol: str
    ltp: float
    volume: int
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    oi: int = 0  # Open Interest for F&O


@dataclass
class Candle:
    """OHLCV Candle"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketRegime:
    """Detect market regime for strategy weighting"""
    VOLATILE = "volatile"
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    OPENING = "opening"  # First 15 min
    CLOSING = "closing"  # Last 15 min
    
    @staticmethod
    def detect(candles: List[Candle], current_time: dtime) -> str:
        """Detect current market regime"""
        if not candles or len(candles) < 20:
            return MarketRegime.SIDEWAYS
        
        # Time-based regimes
        if current_time < dtime(9, 30):
            return MarketRegime.OPENING
        if current_time >= dtime(15, 15):
            return MarketRegime.CLOSING
        
        # Calculate volatility and trend
        closes = np.array([c.close for c in candles[-20:]])
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate trend strength (ADX-like)
        highs = np.array([c.high for c in candles[-20:]])
        lows = np.array([c.low for c in candles[-20:]])
        
        tr = np.maximum(highs - lows, 
            np.maximum(np.abs(highs - np.roll(closes, 1)),
                      np.abs(lows - np.roll(closes, 1))))
        atr = np.mean(tr)
        
        # Price range
        price_range = (max(closes) - min(closes)) / min(closes)
        
        # Regime classification
        if volatility > 0.25:  # High volatility
            return MarketRegime.VOLATILE
        elif price_range > 0.02:  # Trending
            return MarketRegime.TRENDING
        else:
            return MarketRegime.SIDEWAYS


class EnhancedDataFetcher:
    """
    Enhanced data fetcher with:
    - WebSocket support for real-time ticks
    - Smart caching with TTL
    - Candle building from ticks
    - Market regime detection
    """
    
    def __init__(self, websocket_url: Optional[str] = None, api_key: Optional[str] = None):
        self.websocket_url = websocket_url
        self.api_key = api_key
        
        # Caching
        self.tick_cache: Dict[str, MarketTick] = {}
        self.candle_cache: Dict[str, deque] = {}  # Symbol -> deque of candles
        self.cache_ttl = 5  # seconds
        self.last_update: Dict[str, float] = {}
        
        # WebSocket connection
        self.ws_connection = None
        self.ws_subscribed: set = set()
        self.ws_running = False
        
        # Candle building (1-minute candles from ticks)
        self.current_candles: Dict[str, Dict] = {}  # Symbol -> building candle
        self.candle_interval = 60  # seconds
        
        logger.info("✅ Enhanced Data Fetcher initialized")
    
    async def connect_websocket(self, symbols: List[str]):
        """Connect to WebSocket feed for real-time ticks"""
        if not self.websocket_url:
            logger.warning("WebSocket URL not configured, using REST fallback")
            return
        
        try:
            self.ws_connection = await websockets.connect(
                self.websocket_url,
                extra_headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            
            # Subscribe to symbols
            subscribe_msg = {
                "action": "subscribe",
                "mode": "full",
                "symbols": symbols
            }
            await self.ws_connection.send(json.dumps(subscribe_msg))
            
            self.ws_subscribed = set(symbols)
            self.ws_running = True
            logger.info(f"✅ WebSocket connected, subscribed to {len(symbols)} symbols")
            
            # Start listening
            asyncio.create_task(self._listen_websocket())
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}, falling back to REST")
            self.ws_running = False
    
    async def _listen_websocket(self):
        """Listen to WebSocket ticks"""
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                
                # Parse tick data
                tick = MarketTick(
                    symbol=data.get('symbol'),
                    ltp=float(data.get('ltp', 0)),
                    volume=int(data.get('volume', 0)),
                    open=float(data.get('open', 0)),
                    high=float(data.get('high', 0)),
                    low=float(data.get('low', 0)),
                    close=float(data.get('close', 0)),
                    bid=float(data.get('bid', 0)),
                    ask=float(data.get('ask', 0)),
                    oi=int(data.get('oi', 0)),
                    timestamp=datetime.now(ZoneInfo("Asia/Kolkata"))
                )
                
                # Update cache
                self.tick_cache[tick.symbol] = tick
                self.last_update[tick.symbol] = time.time()
                
                # Build candles
                self._build_candle_from_tick(tick)
                
        except Exception as e:
            logger.error(f"WebSocket listening error: {e}")
            self.ws_running = False
    
    def _build_candle_from_tick(self, tick: MarketTick):
        """Build 1-minute candles from real-time ticks"""
        symbol = tick.symbol
        current_minute = tick.timestamp.replace(second=0, microsecond=0)
        
        # Initialize candle if new minute
        if symbol not in self.current_candles:
            self.current_candles[symbol] = {
                'timestamp': current_minute,
                'open': tick.ltp,
                'high': tick.ltp,
                'low': tick.ltp,
                'close': tick.ltp,
                'volume': tick.volume,
                'last_update': current_minute
            }
        else:
            candle_data = self.current_candles[symbol]
            
            # Check if new minute started
            if current_minute > candle_data['last_update']:
                # Save completed candle
                completed_candle = Candle(
                    timestamp=candle_data['timestamp'],
                    open=candle_data['open'],
                    high=candle_data['high'],
                    low=candle_data['low'],
                    close=candle_data['close'],
                    volume=candle_data['volume']
                )
                
                if symbol not in self.candle_cache:
                    self.candle_cache[symbol] = deque(maxlen=500)  # Keep 500 candles
                
                self.candle_cache[symbol].append(completed_candle)
                
                # Start new candle
                self.current_candles[symbol] = {
                    'timestamp': current_minute,
                    'open': tick.ltp,
                    'high': tick.ltp,
                    'low': tick.ltp,
                    'close': tick.ltp,
                    'volume': tick.volume,
                    'last_update': current_minute
                }
            else:
                # Update current candle
                candle_data['high'] = max(candle_data['high'], tick.ltp)
                candle_data['low'] = min(candle_data['low'], tick.ltp)
                candle_data['close'] = tick.ltp
                candle_data['volume'] = tick.volume
    
    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get last traded price with smart caching"""
        now = time.time()
        
        # Check WebSocket cache first
        if symbol in self.tick_cache:
            cache_age = now - self.last_update.get(symbol, 0)
            if cache_age < self.cache_ttl:
                return self.tick_cache[symbol].ltp
        
        # Fallback to REST API
        return self._fetch_ltp_rest(symbol)
    
    def _fetch_ltp_rest(self, symbol: str) -> Optional[float]:
        """Fallback REST API for LTP"""
        try:
            yf_symbol = symbol.replace('-EQ', '.NS')
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?interval=1m&range=1d"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('chart', {}).get('result', [])
                
                if result and len(result) > 0:
                    meta = result[0].get('meta', {})
                    price = meta.get('regularMarketPrice')
                    
                    if price:
                        # Update cache
                        tick = MarketTick(
                            symbol=symbol,
                            ltp=float(price),
                            volume=0,
                            open=float(price),
                            high=float(price),
                            low=float(price),
                            close=float(price),
                            timestamp=datetime.now(ZoneInfo("Asia/Kolkata"))
                        )
                        self.tick_cache[symbol] = tick
                        self.last_update[symbol] = time.time()
                        
                        return float(price)
        
        except Exception as e:
            logger.debug(f"REST LTP fetch error for {symbol}: {e}")
        
        return None
    
    def get_candles(self, symbol: str, count: int = 100) -> List[Candle]:
        """Get historical candles with smart caching"""
        # Try cache first
        if symbol in self.candle_cache and len(self.candle_cache[symbol]) >= count:
            return list(self.candle_cache[symbol])[-count:]
        
        # Fetch from API
        candles = self._fetch_candles_rest(symbol, count)
        
        # Update cache
        if candles:
            if symbol not in self.candle_cache:
                self.candle_cache[symbol] = deque(maxlen=500)
            
            for candle in candles:
                self.candle_cache[symbol].append(candle)
        
        return candles
    
    def _fetch_candles_rest(self, symbol: str, count: int) -> List[Candle]:
        """Fetch candles from REST API"""
        try:
            yf_symbol = symbol.replace('-EQ', '.NS')
            # Use 5-minute interval for better data quality
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?interval=5m&range=5d"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('chart', {}).get('result', [])
                
                if result and len(result) > 0:
                    quotes = result[0].get('indicators', {}).get('quote', [{}])[0]
                    timestamps = result[0].get('timestamp', [])
                    
                    candles = []
                    for i in range(len(timestamps)):
                        if quotes.get('close', [None])[i] is not None:
                            candle = Candle(
                                timestamp=datetime.fromtimestamp(timestamps[i], tz=ZoneInfo("Asia/Kolkata")),
                                open=float(quotes.get('open', [0])[i] or quotes['close'][i]),
                                high=float(quotes.get('high', [0])[i] or quotes['close'][i]),
                                low=float(quotes.get('low', [0])[i] or quotes['close'][i]),
                                close=float(quotes['close'][i]),
                                volume=int(quotes.get('volume', [0])[i] or 0)
                            )
                            candles.append(candle)
                    
                    return candles[-count:] if len(candles) > count else candles
        
        except Exception as e:
            logger.error(f"Candle fetch error for {symbol}: {e}")
        
        return []
    
    def get_market_regime(self, symbol: str) -> str:
        """Get current market regime for strategy weighting"""
        candles = self.get_candles(symbol, 20)
        current_time = datetime.now(ZoneInfo("Asia/Kolkata")).time()
        
        return MarketRegime.detect(candles, current_time)
    
    def get_tick_data(self, symbol: str) -> Optional[MarketTick]:
        """Get full tick data including bid/ask/OI"""
        return self.tick_cache.get(symbol)
    
    async def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_running = False
            logger.info("WebSocket disconnected")


class AccurateMarketTimeManager:
    """
    Accurate Indian market hours checker
    Fixes: Pre-market detection issues
    """
    
    @staticmethod
    def get_current_ist_time() -> datetime:
        """Get accurate IST time"""
        return datetime.now(ZoneInfo("Asia/Kolkata"))
    
    @staticmethod
    def is_market_open() -> Tuple[bool, str]:
        """
        Check if market is open with accurate time detection
        Returns: (is_open, status)
        """
        now = AccurateMarketTimeManager.get_current_ist_time()
        current_time = now.time()
        current_day = now.weekday()
        
        # Weekend check
        if current_day >= 5:  # Saturday = 5, Sunday = 6
            return False, "closed_weekend"
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = dtime(9, 15, 0)
        market_close = dtime(15, 30, 0)
        pre_open = dtime(9, 0, 0)
        
        # Pre-market (9:00 - 9:15)
        if pre_open <= current_time < market_open:
            return False, "pre_market"
        
        # Market hours
        if market_open <= current_time < market_close:
            return True, "open"
        
        # After hours
        if current_time >= market_close:
            return False, "closed"
        
        # Before pre-market
        return False, "closed"
    
    @staticmethod
    def minutes_to_close() -> int:
        """Minutes until market closes"""
        now = AccurateMarketTimeManager.get_current_ist_time()
        is_open, status = AccurateMarketTimeManager.is_market_open()
        
        if not is_open:
            return 0
        
        close_time = datetime.combine(now.date(), dtime(15, 30, 0))
        close_time = close_time.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        
        delta = (close_time - now).total_seconds() / 60
        return max(0, int(delta))
    
    @staticmethod
    def minutes_to_open() -> int:
        """Minutes until market opens"""
        now = AccurateMarketTimeManager.get_current_ist_time()
        is_open, status = AccurateMarketTimeManager.is_market_open()
        
        if is_open:
            return 0
        
        # Calculate next market open
        open_time = datetime.combine(now.date(), dtime(9, 15, 0))
        open_time = open_time.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        
        if now.time() >= dtime(15, 30, 0):
            # After close, next day
            open_time += timedelta(days=1)
        
        # Skip weekends
        while open_time.weekday() >= 5:
            open_time += timedelta(days=1)
        
        delta = (open_time - now).total_seconds() / 60
        return max(0, int(delta))


# Usage Example
"""
# Initialize enhanced data fetcher
fetcher = EnhancedDataFetcher(
    websocket_url="wss://your-broker-ws.com/stream",
    api_key="your_api_key"
)

# Connect WebSocket for real-time ticks
symbols = ['RELIANCE-EQ', 'TCS-EQ', 'INFY-EQ']
await fetcher.connect_websocket(symbols)

# Get real-time data
ltp = fetcher.get_ltp('RELIANCE-EQ')
candles = fetcher.get_candles('RELIANCE-EQ', 100)
regime = fetcher.get_market_regime('RELIANCE-EQ')
tick = fetcher.get_tick_data('RELIANCE-EQ')

# Check market status (accurate)
is_open, status = AccurateMarketTimeManager.is_market_open()
minutes_left = AccurateMarketTimeManager.minutes_to_close()
"""
