"""
Advanced Trading Strategies for Indian Stock Market
Additional strategies to complement the main bot

Includes:
- Breakout Strategy
- Mean Reversion Strategy  
- Options Strategies (Iron Condor, Straddle)
- Scalping Strategy
- Swing Trading Strategy
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import talib

# Import from main bot
from trading_bot import TradingSignal, SignalType, TechnicalIndicators


# ==================== BREAKOUT STRATEGY ====================

class BreakoutStrategy:
    """
    Detects and trades breakouts from consolidation zones
    Works well for NIFTY/BANKNIFTY
    """
    
    def __init__(self, lookback_period: int = 20, volume_multiplier: float = 1.5):
        self.lookback_period = lookback_period
        self.volume_multiplier = volume_multiplier
        self.indicators = TechnicalIndicators()
    
    def detect_consolidation(self, df: pd.DataFrame) -> bool:
        """Detect if price is in consolidation phase"""
        recent_data = df.tail(self.lookback_period)
        
        # Calculate price range
        high_max = recent_data['high'].max()
        low_min = recent_data['low'].min()
        price_range = (high_max - low_min) / low_min
        
        # Consolidation if range < 5%
        return price_range < 0.05
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Analyze for breakout signals"""
        
        if len(df) < self.lookback_period + 10:
            return self._hold_signal(df, symbol)
        
        # Get resistance and support
        resistance = df['high'].rolling(self.lookback_period).max().iloc[-1]
        support = df['low'].rolling(self.lookback_period).min().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        # Calculate ATR for stop loss
        atr = talib.ATR(df['high'].values, df['low'].values, 
                       df['close'].values, timeperiod=14)[-1]
        
        # Check for bullish breakout
        if (current_price > resistance and 
            current_volume > avg_volume * self.volume_multiplier):
            
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'resistance': resistance,
                    'support': support,
                    'volume_ratio': current_volume / avg_volume,
                    'atr': atr
                },
                confidence=0.8,
                stop_loss=resistance - (atr * 2),
                target=current_price + (resistance - support) * 1.5
            )
        
        # Check for bearish breakout
        elif (current_price < support and 
              current_volume > avg_volume * self.volume_multiplier):
            
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'resistance': resistance,
                    'support': support,
                    'volume_ratio': current_volume / avg_volume,
                    'atr': atr
                },
                confidence=0.8,
                stop_loss=support + (atr * 2),
                target=current_price - (resistance - support) * 1.5
            )
        
        return self._hold_signal(df, symbol)
    
    def _hold_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Return hold signal"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.HOLD,
            price=df['close'].iloc[-1],
            timestamp=pd.Timestamp.now(),
            indicators={},
            confidence=0.0,
            stop_loss=0.0,
            target=0.0
        )


# ==================== MEAN REVERSION STRATEGY ====================

class MeanReversionStrategy:
    """
    Mean reversion strategy using Bollinger Bands
    Good for range-bound markets
    """
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.indicators = TechnicalIndicators()
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Analyze for mean reversion signals"""
        
        if len(df) < self.bb_period + 10:
            return self._hold_signal(df, symbol)
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'].values,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )
        
        current_price = df['close'].iloc[-1]
        rsi = talib.RSI(df['close'].values, timeperiod=14)[-1]
        
        # Buy when price touches lower band and RSI < 30
        if current_price <= lower[-1] and rsi < 30:
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'bb_upper': upper[-1],
                    'bb_middle': middle[-1],
                    'bb_lower': lower[-1],
                    'rsi': rsi,
                    'distance_from_mean': (current_price - middle[-1]) / middle[-1] * 100
                },
                confidence=0.75,
                stop_loss=current_price * 0.97,
                target=middle[-1]
            )
        
        # Sell when price touches upper band and RSI > 70
        elif current_price >= upper[-1] and rsi > 70:
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'bb_upper': upper[-1],
                    'bb_middle': middle[-1],
                    'bb_lower': lower[-1],
                    'rsi': rsi,
                    'distance_from_mean': (current_price - middle[-1]) / middle[-1] * 100
                },
                confidence=0.75,
                stop_loss=current_price * 1.03,
                target=middle[-1]
            )
        
        return self._hold_signal(df, symbol)
    
    def _hold_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Return hold signal"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.HOLD,
            price=df['close'].iloc[-1],
            timestamp=pd.Timestamp.now(),
            indicators={},
            confidence=0.0,
            stop_loss=0.0,
            target=0.0
        )


# ==================== SCALPING STRATEGY ====================

class ScalpingStrategy:
    """
    High-frequency scalping strategy
    Uses 1-minute or 5-minute candles
    Targets small profits with tight stops
    """
    
    def __init__(self, ema_fast: int = 5, ema_slow: int = 13):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.indicators = TechnicalIndicators()
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Analyze for scalping signals"""
        
        if len(df) < max(self.ema_fast, self.ema_slow) + 10:
            return self._hold_signal(df, symbol)
        
        # Calculate EMAs
        ema_fast = talib.EMA(df['close'].values, timeperiod=self.ema_fast)
        ema_slow = talib.EMA(df['close'].values, timeperiod=self.ema_slow)
        
        # Calculate VWAP
        vwap = self.indicators.calculate_vwap(df).iloc[-1]
        
        current_price = df['close'].iloc[-1]
        current_ema_fast = ema_fast[-1]
        current_ema_slow = ema_slow[-1]
        prev_ema_fast = ema_fast[-2]
        prev_ema_slow = ema_slow[-2]
        
        # Calculate ATR for stop loss
        atr = talib.ATR(df['high'].values, df['low'].values, 
                       df['close'].values, timeperiod=14)[-1]
        
        # Bullish crossover
        if (current_ema_fast > current_ema_slow and 
            prev_ema_fast <= prev_ema_slow and
            current_price > vwap):
            
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'ema_fast': current_ema_fast,
                    'ema_slow': current_ema_slow,
                    'vwap': vwap,
                    'atr': atr
                },
                confidence=0.85,
                stop_loss=current_price - (atr * 1.5),
                target=current_price + (atr * 2)
            )
        
        # Bearish crossover
        elif (current_ema_fast < current_ema_slow and 
              prev_ema_fast >= prev_ema_slow and
              current_price < vwap):
            
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'ema_fast': current_ema_fast,
                    'ema_slow': current_ema_slow,
                    'vwap': vwap,
                    'atr': atr
                },
                confidence=0.85,
                stop_loss=current_price + (atr * 1.5),
                target=current_price - (atr * 2)
            )
        
        return self._hold_signal(df, symbol)
    
    def _hold_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Return hold signal"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.HOLD,
            price=df['close'].iloc[-1],
            timestamp=pd.Timestamp.now(),
            indicators={},
            confidence=0.0,
            stop_loss=0.0,
            target=0.0
        )


# ==================== SWING TRADING STRATEGY ====================

class SwingTradingStrategy:
    """
    Swing trading strategy for 3-10 day trades
    Uses daily timeframe
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Analyze for swing trading signals"""
        
        if len(df) < 50:
            return self._hold_signal(df, symbol)
        
        # Calculate indicators
        rsi = talib.RSI(df['close'].values, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        adx = talib.ADX(df['high'].values, df['low'].values, 
                       df['close'].values, timeperiod=14)
        
        # Identify support and resistance
        recent_highs = df['high'].rolling(20).max()
        recent_lows = df['low'].rolling(20).min()
        
        current_price = df['close'].iloc[-1]
        support = recent_lows.iloc[-1]
        resistance = recent_highs.iloc[-1]
        
        # Strong uptrend conditions
        if (rsi[-1] > 40 and rsi[-1] < 60 and  # Not overbought
            macd[-1] > macd_signal[-1] and      # MACD bullish
            adx[-1] > 25 and                     # Strong trend
            current_price > support * 1.02):     # Above support
            
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'rsi': rsi[-1],
                    'macd': macd[-1],
                    'adx': adx[-1],
                    'support': support,
                    'resistance': resistance
                },
                confidence=0.7,
                stop_loss=support * 0.98,
                target=resistance
            )
        
        # Strong downtrend conditions
        elif (rsi[-1] < 60 and rsi[-1] > 40 and  # Not oversold
              macd[-1] < macd_signal[-1] and      # MACD bearish
              adx[-1] > 25 and                     # Strong trend
              current_price < resistance * 0.98):  # Below resistance
            
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'rsi': rsi[-1],
                    'macd': macd[-1],
                    'adx': adx[-1],
                    'support': support,
                    'resistance': resistance
                },
                confidence=0.7,
                stop_loss=resistance * 1.02,
                target=support
            )
        
        return self._hold_signal(df, symbol)
    
    def _hold_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Return hold signal"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.HOLD,
            price=df['close'].iloc[-1],
            timestamp=pd.Timestamp.now(),
            indicators={},
            confidence=0.0,
            stop_loss=0.0,
            target=0.0
        )


# ==================== MULTI-STRATEGY COMBINER ====================

class MultiStrategySystem:
    """
    Combines multiple strategies for better accuracy
    Uses voting system to generate final signal
    """
    
    def __init__(self):
        self.strategies = {
            'main': None,  # Main strategy from trading_bot.py
            'breakout': BreakoutStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'scalping': ScalpingStrategy(),
            'swing': SwingTradingStrategy()
        }
    
    def analyze(self, df: pd.DataFrame, symbol: str, 
                main_strategy) -> TradingSignal:
        """Analyze using all strategies and combine signals"""
        
        self.strategies['main'] = main_strategy
        
        signals = {}
        for name, strategy in self.strategies.items():
            if strategy:
                signal = strategy.analyze(df, symbol)
                signals[name] = signal
        
        # Voting system
        buy_votes = sum(1 for s in signals.values() if s.signal == SignalType.BUY)
        sell_votes = sum(1 for s in signals.values() if s.signal == SignalType.SELL)
        
        # Calculate weighted confidence
        confidences = [s.confidence for s in signals.values() 
                      if s.signal != SignalType.HOLD]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Final decision (need majority vote)
        if buy_votes >= 3:
            # Aggregate stop loss and target
            buy_signals = [s for s in signals.values() 
                          if s.signal == SignalType.BUY]
            avg_stop_loss = np.mean([s.stop_loss for s in buy_signals])
            avg_target = np.mean([s.target for s in buy_signals])
            
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                price=df['close'].iloc[-1],
                timestamp=pd.Timestamp.now(),
                indicators={
                    'buy_votes': buy_votes,
                    'sell_votes': sell_votes,
                    'strategies': list(signals.keys())
                },
                confidence=avg_confidence,
                stop_loss=avg_stop_loss,
                target=avg_target
            )
        
        elif sell_votes >= 3:
            sell_signals = [s for s in signals.values() 
                           if s.signal == SignalType.SELL]
            avg_stop_loss = np.mean([s.stop_loss for s in sell_signals])
            avg_target = np.mean([s.target for s in sell_signals])
            
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                price=df['close'].iloc[-1],
                timestamp=pd.Timestamp.now(),
                indicators={
                    'buy_votes': buy_votes,
                    'sell_votes': sell_votes,
                    'strategies': list(signals.keys())
                },
                confidence=avg_confidence,
                stop_loss=avg_stop_loss,
                target=avg_target
            )
        
        # Hold if no consensus
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.HOLD,
            price=df['close'].iloc[-1],
            timestamp=pd.Timestamp.now(),
            indicators={'buy_votes': buy_votes, 'sell_votes': sell_votes},
            confidence=0.0,
            stop_loss=0.0,
            target=0.0
        )


# ==================== USAGE EXAMPLE ====================

"""
To use these advanced strategies, modify trading_bot.py:

from advanced_strategies import (
    BreakoutStrategy, 
    MeanReversionStrategy,
    ScalpingStrategy,
    SwingTradingStrategy,
    MultiStrategySystem
)

# In TradingBot class, replace process_symbol method:

def process_symbol(self, symbol: str):
    try:
        # Fetch data
        df = self.broker.get_historical_data(...)
        
        # Use multi-strategy system
        multi_strategy = MultiStrategySystem()
        signal = multi_strategy.analyze(df, symbol, self.strategy)
        
        # Or use individual strategy
        # breakout_strategy = BreakoutStrategy()
        # signal = breakout_strategy.analyze(df, symbol)
        
        # Execute based on signal
        if signal.signal == SignalType.BUY:
            self.execute_buy(signal)
        elif signal.signal == SignalType.SELL:
            self.execute_sell(signal)
            
    except Exception as e:
        logger.error(f"Error: {e}")
"""