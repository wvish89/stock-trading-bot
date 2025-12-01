"""
Simple Strategy Engine with 7 Trading Strategies
All strategies vote, majority decides
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimpleStrategyEngine:
    """7-Strategy Ensemble System"""
    
    def __init__(self):
        self.strategies = [
            self.opening_range_breakout,
            self.momentum_strategy,
            self.breakout_strategy,
            self.scalping_strategy,
            self.moving_average_strategy,
            self.rsi_strategy,
            self.bollinger_bands_strategy
        ]
        logger.info(f"✅ Strategy Engine initialized with {len(self.strategies)} strategies")
    
    def generate_signal(self, symbol, candles_df, current_price):
        """
        Generate signal from all strategies
        Returns: dict with action, confidence, stop_loss, take_profit
        """
        try:
            votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            buy_signals = []
            sell_signals = []
            
            # Get votes from all strategies
            for strategy in self.strategies:
                signal = strategy(candles_df, current_price)
                
                votes[signal['action']] += 1
                
                if signal['action'] == 'BUY':
                    buy_signals.append(signal)
                elif signal['action'] == 'SELL':
                    sell_signals.append(signal)
            
            # Determine final action
            if votes['BUY'] >= 4:  # Need 4+ votes
                # Average the stop loss and targets
                avg_sl = np.mean([s['stop_loss'] for s in buy_signals])
                avg_tp = np.mean([s['take_profit'] for s in buy_signals])
                
                return {
                    'action': 'BUY',
                    'confidence': votes['BUY'] / len(self.strategies),
                    'strategies_agreeing': votes['BUY'],
                    'total_strategies': len(self.strategies),
                    'stop_loss': avg_sl,
                    'take_profit': avg_tp
                }
            
            elif votes['SELL'] >= 4:
                avg_sl = np.mean([s['stop_loss'] for s in sell_signals])
                avg_tp = np.mean([s['take_profit'] for s in sell_signals])
                
                return {
                    'action': 'SELL',
                    'confidence': votes['SELL'] / len(self.strategies),
                    'strategies_agreeing': votes['SELL'],
                    'total_strategies': len(self.strategies),
                    'stop_loss': avg_sl,
                    'take_profit': avg_tp
                }
            
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'strategies_agreeing': 0,
                'total_strategies': len(self.strategies),
                'stop_loss': current_price,
                'take_profit': current_price
            }
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return self._hold_signal(current_price)
    
    def _hold_signal(self, price):
        """Return HOLD signal"""
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'strategies_agreeing': 0,
            'total_strategies': len(self.strategies),
            'stop_loss': price,
            'take_profit': price
        }
    
    # ==================== STRATEGY 1: Opening Range Breakout ====================
    
    def opening_range_breakout(self, df, current_price):
        """Opening range breakout strategy"""
        try:
            if len(df) < 15:
                return self._hold_signal(current_price)
            
            # Get first 15 candles as opening range
            or_high = df['High'].iloc[:15].max()
            or_low = df['Low'].iloc[:15].min()
            
            # Volume confirmation
            avg_volume = df['Volume'].iloc[-20:].mean()
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Breakout above
            if current_price > or_high * 1.002 and volume_ratio > 1.2:
                return {
                    'action': 'BUY',
                    'stop_loss': or_high * 0.995,
                    'take_profit': current_price + (or_high - or_low) * 1.5
                }
            
            # Breakdown below
            elif current_price < or_low * 0.998 and volume_ratio > 1.2:
                return {
                    'action': 'SELL',
                    'stop_loss': or_low * 1.005,
                    'take_profit': current_price - (or_high - or_low) * 1.5
                }
            
        except:
            pass
        
        return self._hold_signal(current_price)
    
    # ==================== STRATEGY 2: Momentum ====================
    
    def momentum_strategy(self, df, current_price):
        """Momentum strategy using ROC"""
        try:
            if len(df) < 20:
                return self._hold_signal(current_price)
            
            closes = df['Close'].values
            
            # Rate of change
            roc_14 = (closes[-1] - closes[-14]) / closes[-14]
            roc_5 = (closes[-1] - closes[-5]) / closes[-5]
            
            # Volume trend
            recent_vol = df['Volume'].iloc[-5:].mean()
            older_vol = df['Volume'].iloc[-20:-5].mean()
            vol_trend = recent_vol / older_vol if older_vol > 0 else 1
            
            # Strong upward momentum
            if roc_14 > 0.03 and roc_5 > 0.015 and vol_trend > 1.3:
                return {
                    'action': 'BUY',
                    'stop_loss': current_price * 0.98,
                    'take_profit': current_price * (1 + abs(roc_14) * 1.5)
                }
            
            # Strong downward momentum
            elif roc_14 < -0.03 and roc_5 < -0.015 and vol_trend > 1.3:
                return {
                    'action': 'SELL',
                    'stop_loss': current_price * 1.02,
                    'take_profit': current_price * (1 - abs(roc_14) * 1.5)
                }
            
        except:
            pass
        
        return self._hold_signal(current_price)
    
    # ==================== STRATEGY 3: Breakout ====================
    
    def breakout_strategy(self, df, current_price):
        """Breakout from consolidation"""
        try:
            if len(df) < 20:
                return self._hold_signal(current_price)
            
            # Resistance and support
            resistance = df['High'].iloc[-20:].max()
            support = df['Low'].iloc[-20:].min()
            
            # Volume
            avg_volume = df['Volume'].iloc[-20:].mean()
            current_volume = df['Volume'].iloc[-1]
            vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Breakout above resistance
            if current_price > resistance and vol_ratio > 1.5:
                return {
                    'action': 'BUY',
                    'stop_loss': resistance * 0.98,
                    'take_profit': current_price + (resistance - support) * 1.5
                }
            
            # Breakdown below support
            elif current_price < support and vol_ratio > 1.5:
                return {
                    'action': 'SELL',
                    'stop_loss': support * 1.02,
                    'take_profit': current_price - (resistance - support) * 1.5
                }
            
        except:
            pass
        
        return self._hold_signal(current_price)
    
    # ==================== STRATEGY 4: Scalping ====================
    
    def scalping_strategy(self, df, current_price):
        """Scalping with EMA crossover"""
        try:
            if len(df) < 20:
                return self._hold_signal(current_price)
            
            # Calculate EMAs
            ema_5 = df['Close'].ewm(span=5).mean()
            ema_13 = df['Close'].ewm(span=13).mean()
            
            # Current and previous values
            ema5_curr = ema_5.iloc[-1]
            ema13_curr = ema_13.iloc[-1]
            ema5_prev = ema_5.iloc[-2]
            ema13_prev = ema_13.iloc[-2]
            
            # ATR for stop loss
            high_low = df['High'] - df['Low']
            atr = high_low.iloc[-14:].mean()
            
            # Bullish crossover
            if ema5_curr > ema13_curr and ema5_prev <= ema13_prev:
                return {
                    'action': 'BUY',
                    'stop_loss': current_price - (atr * 1.5),
                    'take_profit': current_price + (atr * 2)
                }
            
            # Bearish crossover
            elif ema5_curr < ema13_curr and ema5_prev >= ema13_prev:
                return {
                    'action': 'SELL',
                    'stop_loss': current_price + (atr * 1.5),
                    'take_profit': current_price - (atr * 2)
                }
            
        except:
            pass
        
        return self._hold_signal(current_price)
    
    # ==================== STRATEGY 5: Moving Average ====================
    
    def moving_average_strategy(self, df, current_price):
        """Moving average trend following"""
        try:
            if len(df) < 50:
                return self._hold_signal(current_price)
            
            # Calculate MAs
            ma_20 = df['Close'].rolling(20).mean().iloc[-1]
            ma_50 = df['Close'].rolling(50).mean().iloc[-1]
            
            # Price position
            above_20 = current_price > ma_20
            above_50 = current_price > ma_50
            ma_20_above_50 = ma_20 > ma_50
            
            # ATR
            high_low = df['High'] - df['Low']
            atr = high_low.iloc[-14:].mean()
            
            # Strong uptrend
            if above_20 and above_50 and ma_20_above_50:
                return {
                    'action': 'BUY',
                    'stop_loss': ma_20 * 0.98,
                    'take_profit': current_price + (atr * 3)
                }
            
            # Strong downtrend
            elif not above_20 and not above_50 and not ma_20_above_50:
                return {
                    'action': 'SELL',
                    'stop_loss': ma_20 * 1.02,
                    'take_profit': current_price - (atr * 3)
                }
            
        except:
            pass
        
        return self._hold_signal(current_price)
    
    # ==================== STRATEGY 6: RSI ====================
    
    def rsi_strategy(self, df, current_price):
        """RSI mean reversion"""
        try:
            if len(df) < 20:
                return self._hold_signal(current_price)
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # ATR
            high_low = df['High'] - df['Low']
            atr = high_low.iloc[-14:].mean()
            
            # Oversold
            if current_rsi < 30:
                return {
                    'action': 'BUY',
                    'stop_loss': current_price * 0.97,
                    'take_profit': current_price + (atr * 2)
                }
            
            # Overbought
            elif current_rsi > 70:
                return {
                    'action': 'SELL',
                    'stop_loss': current_price * 1.03,
                    'take_profit': current_price - (atr * 2)
                }
            
        except:
            pass
        
        return self._hold_signal(current_price)
    
    # ==================== STRATEGY 7: Bollinger Bands ====================
    
    def bollinger_bands_strategy(self, df, current_price):
        """Bollinger Bands mean reversion"""
        try:
            if len(df) < 20:
                return self._hold_signal(current_price)
            
            # Calculate Bollinger Bands
            ma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            
            upper_band = ma_20 + (std_20 * 2)
            lower_band = ma_20 - (std_20 * 2)
            
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_mid = ma_20.iloc[-1]
            
            # Touch lower band - buy
            if current_price <= current_lower:
                return {
                    'action': 'BUY',
                    'stop_loss': current_price * 0.97,
                    'take_profit': current_mid
                }
            
            # Touch upper band - sell
            elif current_price >= current_upper:
                return {
                    'action': 'SELL',
                    'stop_loss': current_price * 1.03,
                    'take_profit': current_mid
                }
            
        except:
            pass
        
        return self._hold_signal(current_price)
