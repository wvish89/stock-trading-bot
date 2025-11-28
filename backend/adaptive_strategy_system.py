"""
Adaptive Multi-Strategy System for Indian Intraday Trading
Features:
- Time-of-day strategy weighting
- Market regime adaptation
- Dynamic confidence scoring
- News/sentiment integration
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time as dtime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategySignal:
    """Individual strategy signal"""
    strategy_name: str
    signal: SignalType
    confidence: float
    price: float
    stop_loss: float
    target: float
    indicators: Dict
    timestamp: datetime


@dataclass
class EnsembleSignal:
    """Combined ensemble signal"""
    symbol: str
    signal: SignalType
    confidence: float
    strategies_agreeing: int
    total_strategies: int
    price: float
    stop_loss: float
    target: float
    timestamp: datetime
    contributing_strategies: List[str]
    market_regime: str
    indicators: Dict


class TimeBasedWeights:
    """
    Strategy weights based on time of day
    Indian market characteristics:
    - Opening (9:15-9:45): High volatility, momentum-driven
    - Mid-morning (9:45-11:30): Trend establishment
    - Lunch (11:30-13:30): Low volume, choppy
    - Afternoon (13:30-15:00): Trend continuation or reversal
    - Closing (15:00-15:30): Volatility spike, squaring off
    """
    
    @staticmethod
    def get_weights(current_time: dtime, regime: str) -> Dict[str, float]:
        """Get strategy weights based on time and regime"""
        
        # Base weights (neutral)
        weights = {
            'opening_range_breakout': 1.0,
            'momentum': 1.0,
            'breakout': 1.0,
            'scalping': 1.0,
            'moving_average': 1.0,
            'rsi': 1.0,
            'bollinger_bands': 1.0
        }
        
        # Opening session (9:15-9:45) - Momentum and ORB dominate
        if dtime(9, 15) <= current_time < dtime(9, 45):
            weights['opening_range_breakout'] = 2.0
            weights['momentum'] = 1.8
            weights['scalping'] = 1.5
            weights['moving_average'] = 0.7
            weights['rsi'] = 0.8
        
        # Mid-morning (9:45-11:30) - Trend strategies
        elif dtime(9, 45) <= current_time < dtime(11, 30):
            weights['momentum'] = 1.5
            weights['moving_average'] = 1.6
            weights['breakout'] = 1.4
            weights['opening_range_breakout'] = 0.8
        
        # Lunch (11:30-13:30) - Mean reversion
        elif dtime(11, 30) <= current_time < dtime(13, 30):
            weights['rsi'] = 1.6
            weights['bollinger_bands'] = 1.5
            weights['momentum'] = 0.8
            weights['breakout'] = 0.7
            weights['scalping'] = 1.2
        
        # Afternoon (13:30-15:00) - Trend continuation
        elif dtime(13, 30) <= current_time < dtime(15, 0):
            weights['moving_average'] = 1.5
            weights['momentum'] = 1.4
            weights['breakout'] = 1.3
        
        # Closing (15:00-15:30) - Reduce all positions
        elif dtime(15, 0) <= current_time < dtime(15, 30):
            weights['scalping'] = 0.5
            weights['momentum'] = 0.6
            weights['opening_range_breakout'] = 0.3
            weights['breakout'] = 0.5
            weights['moving_average'] = 0.7
            weights['rsi'] = 0.8
            weights['bollinger_bands'] = 0.8
        
        # Regime-based adjustments
        if regime == 'volatile':
            weights['scalping'] *= 0.7
            weights['momentum'] *= 1.2
            weights['breakout'] *= 0.8
        
        elif regime == 'trending':
            weights['momentum'] *= 1.3
            weights['moving_average'] *= 1.3
            weights['breakout'] *= 1.2
            weights['rsi'] *= 0.8
            weights['bollinger_bands'] *= 0.8
        
        elif regime == 'sideways':
            weights['rsi'] *= 1.3
            weights['bollinger_bands'] *= 1.3
            weights['momentum'] *= 0.7
            weights['breakout'] *= 0.7
        
        return weights


class EnhancedStrategy:
    """Base strategy with time-aware confidence"""
    
    def __init__(self, name: str):
        self.name = name
    
    def analyze(self, candles: List, current_time: dtime, regime: str) -> Optional[StrategySignal]:
        """Analyze and return signal - to be overridden"""
        pass
    
    def calculate_weighted_confidence(self, base_confidence: float, 
                                     current_time: dtime, regime: str) -> float:
        """Apply time and regime weights to confidence"""
        weights = TimeBasedWeights.get_weights(current_time, regime)
        weight = weights.get(self.name.lower().replace(' ', '_'), 1.0)
        
        return min(base_confidence * weight, 1.0)


class OpeningRangeBreakoutStrategy(EnhancedStrategy):
    """Enhanced ORB with time weighting"""
    
    def __init__(self):
        super().__init__("Opening Range Breakout")
    
    def analyze(self, candles: List, current_time: dtime, regime: str) -> Optional[StrategySignal]:
        if len(candles) < 15:
            return None
        
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        
        # Opening range
        or_high = max(highs[-15:-5])
        or_low = min(lows[-15:-5])
        current = closes[-1]
        avg_volume = np.mean(volumes[-15:])
        current_volume = volumes[-1]
        
        # Calculate base confidence
        volume_confirmation = current_volume / avg_volume if avg_volume > 0 else 1.0
        range_size = (or_high - or_low) / or_low
        
        base_confidence = 0.7
        if volume_confirmation > 1.5:
            base_confidence += 0.1
        if 0.01 < range_size < 0.03:  # Good range size
            base_confidence += 0.1
        
        # Apply time/regime weighting
        confidence = self.calculate_weighted_confidence(base_confidence, current_time, regime)
        
        # Breakout detection
        if current > or_high * 1.002 and volume_confirmation > 1.2:
            return StrategySignal(
                strategy_name=self.name,
                signal=SignalType.BUY,
                confidence=confidence,
                price=current,
                stop_loss=or_high * 0.995,
                target=current + (or_high - or_low) * 1.5,
                indicators={
                    'or_high': or_high,
                    'or_low': or_low,
                    'volume_ratio': volume_confirmation
                },
                timestamp=datetime.now()
            )
        
        elif current < or_low * 0.998 and volume_confirmation > 1.2:
            return StrategySignal(
                strategy_name=self.name,
                signal=SignalType.SELL,
                confidence=confidence,
                price=current,
                stop_loss=or_low * 1.005,
                target=current - (or_high - or_low) * 1.5,
                indicators={
                    'or_high': or_high,
                    'or_low': or_low,
                    'volume_ratio': volume_confirmation
                },
                timestamp=datetime.now()
            )
        
        return None


class MomentumStrategy(EnhancedStrategy):
    """Enhanced momentum with regime awareness"""
    
    def __init__(self):
        super().__init__("Momentum")
    
    def analyze(self, candles: List, current_time: dtime, regime: str) -> Optional[StrategySignal]:
        if len(candles) < 20:
            return None
        
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        
        # Calculate momentum indicators
        roc_14 = (closes[-1] - closes[-14]) / closes[-14] if closes[-14] != 0 else 0
        roc_5 = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] != 0 else 0
        
        # Volume trend
        recent_vol = np.mean(volumes[-5:])
        older_vol = np.mean(volumes[-20:-5])
        vol_trend = recent_vol / older_vol if older_vol > 0 else 1.0
        
        # Price velocity
        velocity = np.mean([closes[i] - closes[i-1] for i in range(-5, 0)])
        
        base_confidence = 0.75
        
        # Strong momentum signals
        if roc_14 > 0.03 and roc_5 > 0.015 and vol_trend > 1.3 and velocity > 0:
            confidence = self.calculate_weighted_confidence(base_confidence, current_time, regime)
            
            return StrategySignal(
                strategy_name=self.name,
                signal=SignalType.BUY,
                confidence=confidence,
                price=closes[-1],
                stop_loss=closes[-1] * 0.98,
                target=closes[-1] * (1 + abs(roc_14) * 1.5),
                indicators={
                    'roc_14': roc_14,
                    'roc_5': roc_5,
                    'volume_trend': vol_trend
                },
                timestamp=datetime.now()
            )
        
        elif roc_14 < -0.03 and roc_5 < -0.015 and vol_trend > 1.3 and velocity < 0:
            confidence = self.calculate_weighted_confidence(base_confidence, current_time, regime)
            
            return StrategySignal(
                strategy_name=self.name,
                signal=SignalType.SELL,
                confidence=confidence,
                price=closes[-1],
                stop_loss=closes[-1] * 1.02,
                target=closes[-1] * (1 - abs(roc_14) * 1.5),
                indicators={
                    'roc_14': roc_14,
                    'roc_5': roc_5,
                    'volume_trend': vol_trend
                },
                timestamp=datetime.now()
            )
        
        return None


class AdaptiveEnsemble:
    """
    Adaptive ensemble with:
    - Time-based strategy weighting
    - Market regime detection
    - Dynamic confidence scoring
    - News/sentiment integration
    """
    
    def __init__(self):
        self.strategies = [
            OpeningRangeBreakoutStrategy(),
            MomentumStrategy(),
            # Add other enhanced strategies here
        ]
        
        self.news_sentiment_score = 0.5  # Neutral by default
        logger.info(f"✅ Adaptive Ensemble initialized with {len(self.strategies)} strategies")
    
    def analyze(self, symbol: str, candles: List, current_time: dtime, 
                regime: str, news_sentiment: float = 0.5) -> EnsembleSignal:
        """
        Generate ensemble signal with adaptive weighting
        
        Args:
            symbol: Stock symbol
            candles: Historical candle data
            current_time: Current IST time
            regime: Market regime (volatile/trending/sideways/opening/closing)
            news_sentiment: News sentiment score (0-1, 0.5 = neutral)
        """
        
        if len(candles) < 20:
            return self._hold_signal(symbol, candles)
        
        # Get strategy weights
        weights = TimeBasedWeights.get_weights(current_time, regime)
        
        # Collect signals from all strategies
        signals: List[StrategySignal] = []
        for strategy in self.strategies:
            signal = strategy.analyze(candles, current_time, regime)
            if signal:
                signals.append(signal)
        
        if not signals:
            return self._hold_signal(symbol, candles)
        
        # Weighted voting
        buy_score = 0.0
        sell_score = 0.0
        buy_signals = []
        sell_signals = []
        
        for signal in signals:
            weight = weights.get(signal.strategy_name.lower().replace(' ', '_'), 1.0)
            weighted_confidence = signal.confidence * weight
            
            if signal.signal == SignalType.BUY:
                buy_score += weighted_confidence
                buy_signals.append(signal)
            elif signal.signal == SignalType.SELL:
                sell_score += weighted_confidence
                sell_signals.append(signal)
        
        # Adjust scores based on news sentiment
        if news_sentiment > 0.6:  # Bullish news
            buy_score *= 1.2
            sell_score *= 0.8
        elif news_sentiment < 0.4:  # Bearish news
            buy_score *= 0.8
            sell_score *= 1.2
        
        # Determine final signal (need significant score difference)
        threshold = 2.5  # Minimum weighted score
        
        if buy_score > threshold and buy_score > sell_score * 1.3:
            # Aggregate stop loss and targets
            avg_sl = np.mean([s.stop_loss for s in buy_signals])
            avg_target = np.mean([s.target for s in buy_signals])
            
            return EnsembleSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                confidence=min(buy_score / len(self.strategies), 0.95),
                strategies_agreeing=len(buy_signals),
                total_strategies=len(self.strategies),
                price=candles[-1].close,
                stop_loss=avg_sl,
                target=avg_target,
                timestamp=datetime.now(),
                contributing_strategies=[s.strategy_name for s in buy_signals],
                market_regime=regime,
                indicators={
                    'buy_score': round(buy_score, 2),
                    'sell_score': round(sell_score, 2),
                    'news_sentiment': news_sentiment,
                    'time_of_day': current_time.strftime("%H:%M")
                }
            )
        
        elif sell_score > threshold and sell_score > buy_score * 1.3:
            avg_sl = np.mean([s.stop_loss for s in sell_signals])
            avg_target = np.mean([s.target for s in sell_signals])
            
            return EnsembleSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                confidence=min(sell_score / len(self.strategies), 0.95),
                strategies_agreeing=len(sell_signals),
                total_strategies=len(self.strategies),
                price=candles[-1].close,
                stop_loss=avg_sl,
                target=avg_target,
                timestamp=datetime.now(),
                contributing_strategies=[s.strategy_name for s in sell_signals],
                market_regime=regime,
                indicators={
                    'buy_score': round(buy_score, 2),
                    'sell_score': round(sell_score, 2),
                    'news_sentiment': news_sentiment,
                    'time_of_day': current_time.strftime("%H:%M")
                }
            )
        
        return self._hold_signal(symbol, candles, regime)
    
    def _hold_signal(self, symbol: str, candles: List, regime: str = "unknown") -> EnsembleSignal:
        """Return HOLD signal"""
        return EnsembleSignal(
            symbol=symbol,
            signal=SignalType.HOLD,
            confidence=0.0,
            strategies_agreeing=0,
            total_strategies=len(self.strategies),
            price=candles[-1].close if candles else 0.0,
            stop_loss=0.0,
            target=0.0,
            timestamp=datetime.now(),
            contributing_strategies=[],
            market_regime=regime,
            indicators={}
        )


# Usage Example
"""
from datetime import time as dtime

# Initialize ensemble
ensemble = AdaptiveEnsemble()

# Get current market data
candles = data_fetcher.get_candles('RELIANCE-EQ', 100)
current_time = datetime.now(ZoneInfo("Asia/Kolkata")).time()
regime = data_fetcher.get_market_regime('RELIANCE-EQ')

# Get news sentiment (from news analyzer)
news_sentiment = 0.7  # Bullish

# Generate adaptive signal
signal = ensemble.analyze(
    symbol='RELIANCE-EQ',
    candles=candles,
    current_time=current_time,
    regime=regime,
    news_sentiment=news_sentiment
)

# Execute based on signal
if signal.signal == SignalType.BUY and signal.confidence > 0.65:
    print(f"BUY {signal.symbol}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Strategies agreeing: {signal.strategies_agreeing}/{signal.total_strategies}")
    print(f"Contributing: {signal.contributing_strategies}")
    print(f"Regime: {signal.market_regime}")
    print(f"Stop Loss: {signal.stop_loss}")
    print(f"Target: {signal.target}")
"""
