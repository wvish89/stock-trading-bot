"""
Enhanced Trading Bot - Complete Integration
Combines all improvements:
- Real-time WebSocket data
- Adaptive multi-strategy with time weighting
- News/sentiment analysis
- Portfolio-level risk management
- Auto square-off
- Capital tracking with realized/unrealized PnL
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import threading

# Import our enhanced modules
from realtime_data_manager import (
    EnhancedDataFetcher, 
    AccurateMarketTimeManager,
    MarketRegime,
    Candle
)
from adaptive_strategy_system import (
    AdaptiveEnsemble,
    EnsembleSignal,
    SignalType
)
from news_sentiment_analyzer import (
    NewsSentimentAnalyzer,
    NewsArticle
)
from portfolio_risk_manager import (
    PortfolioRiskManager,
    RiskLimits,
    Position,
    Trade
)

logger = logging.getLogger(__name__)


class EnhancedTradingBot:
    """
    Production-ready trading bot with all enhancements
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 watchlist: List[str] = None,
                 websocket_url: Optional[str] = None,
                 api_keys: Optional[Dict[str, str]] = None,
                 mode: str = "paper"):
        """
        Initialize enhanced trading bot
        
        Args:
            initial_capital: Starting capital
            watchlist: List of symbols to trade
            websocket_url: WebSocket URL for real-time data
            api_keys: API keys for news, broker, etc.
            mode: 'paper' or 'live'
        """
        
        self.mode = mode
        self.running = False
        
        # Default watchlist
        self.watchlist = watchlist or [
            'RELIANCE-EQ', 'TCS-EQ', 'INFY-EQ', 'HDFCBANK-EQ',
            'ICICIBANK-EQ', 'SBIN-EQ', 'BHARTIARTL-EQ', 'ITC-EQ',
            'KOTAKBANK-EQ', 'LT-EQ'
        ]
        
        # Initialize components
        self.data_fetcher = EnhancedDataFetcher(
            websocket_url=websocket_url,
            api_key=api_keys.get('broker_api') if api_keys else None
        )
        
        self.strategy = AdaptiveEnsemble()
        
        self.news_analyzer = NewsSentimentAnalyzer(
            api_keys={'newsapi': api_keys.get('newsapi')} if api_keys else {}
        )
        
        self.portfolio = PortfolioRiskManager(
            initial_capital=initial_capital,
            limits=RiskLimits(
                max_risk_per_trade_pct=0.02,  # 2% per trade
                max_positions=5,
                max_daily_loss_pct=0.05,  # 5% daily loss limit
                square_off_time=dtime(15, 15),
                no_new_trades_after=dtime(15, 0)
            )
        )
        
        # News cache
        self.news_articles: List[NewsArticle] = []
        self.last_news_fetch = 0
        self.news_fetch_interval = 600  # Fetch news every 10 minutes
        
        # Timing
        self.last_strategy_check = 0
        self.strategy_check_interval = 30  # Check strategies every 30 seconds
        
        self.last_position_update = 0
        self.position_update_interval = 10  # Update positions every 10 seconds
        
        logger.info("=" * 80)
        logger.info("✅ ENHANCED TRADING BOT INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Capital: ₹{initial_capital:,.2f}")
        logger.info(f"Watchlist: {len(self.watchlist)} symbols")
        logger.info(f"Features: Real-time data, Adaptive strategies, News sentiment, Portfolio risk mgmt")
        logger.info("=" * 80)
    
    async def start(self):
        """Start the trading bot"""
        logger.info("🚀 Starting Enhanced Trading Bot...")
        
        self.running = True
        
        # Connect WebSocket for real-time data
        try:
            await self.data_fetcher.connect_websocket(self.watchlist)
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}, using REST fallback")
        
        # Start main trading loop
        await self._trading_loop()
    
    async def _trading_loop(self):
        """Main trading loop"""
        
        while self.running:
            try:
                now = time.time()
                current_time = datetime.now(ZoneInfo("Asia/Kolkata")).time()
                
                # Check market status
                is_open, market_status = AccurateMarketTimeManager.is_market_open()
                
                if not is_open:
                    if market_status == "closed_weekend":
                        logger.info("🏖️  Weekend - Market closed")
                        await asyncio.sleep(3600)  # Sleep 1 hour
                    elif market_status == "pre_market":
                        logger.info("⏰ Pre-market - Waiting for market open...")
                        await asyncio.sleep(60)
                    else:
                        logger.info(f"🔒 Market {market_status}")
                        await asyncio.sleep(300)  # Sleep 5 minutes
                    continue
                
                # Market is open - execute trading logic
                
                # 1. Update news sentiment (every 10 minutes)
                if now - self.last_news_fetch > self.news_fetch_interval:
                    await self._update_news_sentiment()
                    self.last_news_fetch = now
                
                # 2. Update all positions (every 10 seconds)
                if now - self.last_position_update > self.position_update_interval:
                    await self._update_positions()
                    self.last_position_update = now
                
                # 3. Check for square-off time
                if self.portfolio.check_square_off_time():
                    await self._square_off_all()
                    logger.info("✅ Auto square-off completed - stopping for the day")
                    break
                
                # 4. Check strategies and execute trades (every 30 seconds)
                if now - self.last_strategy_check > self.strategy_check_interval:
                    await self._check_strategies_and_trade()
                    self.last_strategy_check = now
                
                # Log portfolio status periodically
                if int(now) % 300 == 0:  # Every 5 minutes
                    self._log_portfolio_status()
                
                # Sleep briefly
                await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Trading loop error: {e}", exc_info=True)
                await asyncio.sleep(30)
    
    async def _update_news_sentiment(self):
        """Fetch and update news sentiment"""
        try:
            logger.info("📰 Fetching news...")
            self.news_articles = self.news_analyzer.fetch_indian_market_news(hours=4)
            
            # Get market sentiment
            market_sentiment = self.news_analyzer.get_index_sentiment(self.news_articles)
            logger.info(f"📊 Market Sentiment: NIFTY={market_sentiment['nifty']:.2f}, "
                       f"BANKNIFTY={market_sentiment['banknifty']:.2f}")
            
            # Get top movers
            movers = self.news_analyzer.get_top_movers_news(self.news_articles)
            if movers['bullish']:
                logger.info(f"📈 Top Bullish: {[m['symbol'] for m in movers['bullish'][:3]]}")
            if movers['bearish']:
                logger.info(f"📉 Top Bearish: {[m['symbol'] for m in movers['bearish'][:3]]}")
        
        except Exception as e:
            logger.error(f"News fetch error: {e}")
    
    async def _update_positions(self):
        """Update all open positions with current prices"""
        try:
            # Get current prices for all positions
            price_data = {}
            for symbol in self.portfolio.positions.keys():
                ltp = self.data_fetcher.get_ltp(symbol)
                if ltp:
                    price_data[symbol] = ltp
            
            # Update positions (will auto-close if SL/TP hit)
            self.portfolio.update_positions(price_data)
        
        except Exception as e:
            logger.error(f"Position update error: {e}")
    
    async def _square_off_all(self):
        """Square off all positions at EOD"""
        try:
            # Get current prices
            price_data = {}
            for symbol in list(self.portfolio.positions.keys()):
                ltp = self.data_fetcher.get_ltp(symbol)
                if ltp:
                    price_data[symbol] = ltp
            
            self.portfolio.square_off_all_positions(price_data)
        
        except Exception as e:
            logger.error(f"Square-off error: {e}")
    
    async def _check_strategies_and_trade(self):
        """Check all strategies and execute trades"""
        try:
            current_time = datetime.now(ZoneInfo("Asia/Kolkata")).time()
            
            # Check each symbol in watchlist
            for symbol in self.watchlist:
                # Skip if already in position
                if symbol in self.portfolio.positions:
                    continue
                
                # Get candles
                candles = self.data_fetcher.get_candles(symbol, count=100)
                if len(candles) < 20:
                    continue
                
                # Get market regime
                regime = self.data_fetcher.get_market_regime(symbol)
                
                # Get news sentiment for this stock
                news_sentiment = self.news_analyzer.get_stock_sentiment(symbol, self.news_articles)
                sentiment_signal = self.news_analyzer.convert_to_trading_signal(news_sentiment)
                
                # Generate adaptive signal
                signal = self.strategy.analyze(
                    symbol=symbol,
                    candles=candles,
                    current_time=current_time,
                    regime=regime,
                    news_sentiment=sentiment_signal
                )
                
                # Execute if strong signal
                if signal.signal == SignalType.BUY and signal.confidence >= 0.65:
                    await self._execute_buy(signal)
                
                elif signal.signal == SignalType.SELL and symbol in self.portfolio.positions:
                    await self._execute_sell(signal)
        
        except Exception as e:
            logger.error(f"Strategy check error: {e}")
    
    async def _execute_buy(self, signal: EnsembleSignal):
        """Execute buy trade"""
        try:
            # Check if can take position
            can_take, reason = self.portfolio.can_take_new_position()
            if not can_take:
                logger.warning(f"❌ Cannot open {signal.symbol}: {reason}")
                return
            
            # Open position
            position = self.portfolio.open_position(
                symbol=signal.symbol,
                entry_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.target,
                strategy=f"Adaptive ({signal.strategies_agreeing}/{signal.total_strategies})"
            )
            
            if position:
                logger.info(f"✅ BUY EXECUTED: {signal.symbol}")
                logger.info(f"   Price: ₹{signal.price:.2f}")
                logger.info(f"   Quantity: {position.quantity}")
                logger.info(f"   Stop Loss: ₹{signal.stop_loss:.2f}")
                logger.info(f"   Target: ₹{signal.target:.2f}")
                logger.info(f"   Confidence: {signal.confidence*100:.1f}%")
                logger.info(f"   Strategies: {signal.contributing_strategies}")
                logger.info(f"   Regime: {signal.market_regime}")
        
        except Exception as e:
            logger.error(f"Buy execution error: {e}")
    
    async def _execute_sell(self, signal: EnsembleSignal):
        """Execute sell trade (close position)"""
        try:
            if signal.symbol in self.portfolio.positions:
                trade = self.portfolio.close_position(
                    symbol=signal.symbol,
                    exit_price=signal.price,
                    exit_reason="SIGNAL"
                )
                
                if trade:
                    logger.info(f"✅ SELL EXECUTED: {signal.symbol}")
                    logger.info(f"   Exit Price: ₹{signal.price:.2f}")
                    logger.info(f"   P&L: ₹{trade.realized_pnl:,.2f} ({trade.realized_pnl_pct:.2f}%)")
        
        except Exception as e:
            logger.error(f"Sell execution error: {e}")
    
    def _log_portfolio_status(self):
        """Log current portfolio status"""
        metrics = self.portfolio.get_portfolio_metrics()
        
        logger.info("=" * 80)
        logger.info("📊 PORTFOLIO STATUS")
        logger.info(f"   Total Equity: ₹{metrics['total_equity']:,.2f}")
        logger.info(f"   Available Capital: ₹{metrics['available_capital']:,.2f}")
        logger.info(f"   Deployed: ₹{metrics['deployed_capital']:,.2f} ({metrics['exposure_pct']:.1f}%)")
        logger.info(f"   Unrealized P&L: ₹{metrics['unrealized_pnl']:,.2f}")
        logger.info(f"   Realized P&L: ₹{metrics['realized_pnl']:,.2f}")
        logger.info(f"   Total P&L: ₹{metrics['total_pnl']:,.2f} ({metrics['total_pnl_pct']:.2f}%)")
        logger.info(f"   Open Positions: {metrics['open_positions']}")
        logger.info(f"   Total Trades: {metrics['total_trades']}")
        logger.info(f"   Win Rate: {metrics['win_rate']:.1f}%")
        logger.info("=" * 80)
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("🛑 Stopping trading bot...")
        self.running = False
        
        # Log final status
        self._log_portfolio_status()
        
        logger.info("✅ Trading bot stopped")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        is_open, market_status = AccurateMarketTimeManager.is_market_open()
        metrics = self.portfolio.get_portfolio_metrics()
        
        return {
            'running': self.running,
            'mode': self.mode,
            'market_status': market_status,
            'market_open': is_open,
            'minutes_to_close': AccurateMarketTimeManager.minutes_to_close(),
            'portfolio': metrics,
            'positions': len(self.portfolio.positions),
            'watchlist_size': len(self.watchlist)
        }
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions_list = []
        
        for symbol, position in self.portfolio.positions.items():
            positions_list.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'entry_time': position.entry_time.isoformat(),
                'strategy': position.strategy
            })
        
        return positions_list
    
    def get_trades(self) -> List[Dict]:
        """Get closed trades"""
        trades_list = []
        
        for trade in self.portfolio.closed_trades[-50:]:  # Last 50 trades
            trades_list.append({
                'symbol': trade.symbol,
                'entry_time': trade.entry_time.isoformat(),
                'exit_time': trade.exit_time.isoformat(),
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'realized_pnl': trade.realized_pnl,
                'realized_pnl_pct': trade.realized_pnl_pct,
                'exit_reason': trade.exit_reason,
                'strategy': trade.strategy,
                'holding_time_minutes': trade.holding_time_minutes
            })
        
        return trades_list


# Synchronous wrapper for Flask integration
class TradingBotManager:
    """
    Synchronous wrapper for managing the bot from Flask
    """
    
    def __init__(self):
        self.bot: Optional[EnhancedTradingBot] = None
        self.bot_thread: Optional[threading.Thread] = None
        self.event_loop = None
    
    def start(self, initial_capital: float = 100000, 
              watchlist: List[str] = None,
              api_keys: Dict[str, str] = None,
              mode: str = "paper"):
        """Start the bot in a background thread"""
        
        if self.bot and self.bot.running:
            logger.warning("Bot already running")
            return False
        
        def run_bot():
            # Create new event loop for this thread
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            
            # Create and start bot
            self.bot = EnhancedTradingBot(
                initial_capital=initial_capital,
                watchlist=watchlist,
                api_keys=api_keys or {},
                mode=mode
            )
            
            # Run async start method
            self.event_loop.run_until_complete(self.bot.start())
        
        # Start bot in background thread
        self.bot_thread = threading.Thread(target=run_bot, daemon=True)
        self.bot_thread.start()
        
        logger.info("✅ Bot started in background thread")
        return True
    
    def stop(self):
        """Stop the bot"""
        if self.bot:
            self.bot.stop()
            
            if self.event_loop:
                self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            
            return True
        return False
    
    def get_status(self) -> Dict:
        """Get bot status"""
        if self.bot:
            return self.bot.get_status()
        return {'running': False}
    
    def get_positions(self) -> List[Dict]:
        """Get positions"""
        if self.bot:
            return self.bot.get_positions()
        return []
    
    def get_trades(self) -> List[Dict]:
        """Get trades"""
        if self.bot:
            return self.bot.get_trades()
        return []


# Usage Example
"""
# For direct async usage:
async def main():
    bot = EnhancedTradingBot(
        initial_capital=100000,
        watchlist=['RELIANCE-EQ', 'TCS-EQ', 'INFY-EQ'],
        api_keys={
            'newsapi': 'your_newsapi_key',
            'broker_api': 'your_broker_api_key'
        },
        mode='paper'
    )
    
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())


# For Flask integration:
from flask import Flask, jsonify

app = Flask(__name__)
bot_manager = TradingBotManager()

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    success = bot_manager.start(
        initial_capital=100000,
        watchlist=['RELIANCE-EQ', 'TCS-EQ'],
        api_keys={'newsapi': 'your_key'},
        mode='paper'
    )
    return jsonify({'success': success})

@app.route('/api/bot/status', methods=['GET'])
def get_status():
    return jsonify(bot_manager.get_status())

@app.route('/api/positions', methods=['GET'])
def get_positions():
    return jsonify({'positions': bot_manager.get_positions()})
"""
