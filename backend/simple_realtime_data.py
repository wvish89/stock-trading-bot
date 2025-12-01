"""
Simple Real-Time Data Fetcher
Uses Yahoo Finance for free real-time Indian stock data
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import logging
import time

logger = logging.getLogger(__name__)


class SimpleDataFetcher:
    """Fetch real-time data from Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 5  # seconds
        self.last_update = {}
        logger.info("✅ Data Fetcher initialized (Yahoo Finance)")
    
    def get_ltp(self, symbol):
        """Get last traded price"""
        try:
            # Check cache
            now = time.time()
            if symbol in self.cache:
                age = now - self.last_update.get(symbol, 0)
                if age < self.cache_ttl:
                    return self.cache[symbol]
            
            # Fetch from Yahoo
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try multiple price fields
            price = (info.get('currentPrice') or 
                    info.get('regularMarketPrice') or
                    info.get('previousClose'))
            
            if price:
                self.cache[symbol] = float(price)
                self.last_update[symbol] = now
                return float(price)
            
            return None
            
        except Exception as e:
            logger.debug(f"LTP fetch error for {symbol}: {e}")
            return None
    
    def get_candles(self, symbol, period='5d', interval='5m'):
        """Get historical candles"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Candles fetch error for {symbol}: {e}")
            return None


class SimpleMarketTime:
    """Simple market time checker"""
    
    @staticmethod
    def is_market_open():
        """Check if market is open"""
        try:
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            current_time = now.time()
            current_day = now.weekday()
            
            # Weekend
            if current_day >= 5:
                return False, "closed_weekend"
            
            # Market hours: 9:15 AM to 3:30 PM
            if dtime(9, 15) <= current_time < dtime(15, 30):
                return True, "open"
            
            return False, "closed"
            
        except:
            return False, "unknown"
    
    @staticmethod
    def should_square_off():
        """Check if should square off (3:15 PM)"""
        try:
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            return now.time() >= dtime(15, 15)
        except:
            return False
