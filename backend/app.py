"""
Enhanced Flask Backend - Production Ready
Integrates all improvements:
- Real-time WebSocket data
- Adaptive strategies with time weighting
- News/sentiment analysis
- Portfolio risk management
- Auto square-off
- Proper capital tracking
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import asyncio
import threading

# Import enhanced trading bot
from enhanced_trading_bot import TradingBotManager
from realtime_data_manager import AccurateMarketTimeManager

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

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global bot manager
bot_manager = TradingBotManager()
current_mode = "paper"

# Configuration
INITIAL_CAPITAL = 100000
WATCHLIST = [
    'RELIANCE-EQ', 'TCS-EQ', 'INFY-EQ', 'HDFCBANK-EQ',
    'ICICIBANK-EQ', 'SBIN-EQ', 'BHARTIARTL-EQ', 'ITC-EQ',
    'KOTAKBANK-EQ', 'LT-EQ'
]

# API Keys (load from environment or config)
API_KEYS = {
    'newsapi': os.getenv('NEWSAPI_KEY', ''),
    'broker_api': os.getenv('BROKER_API_KEY', ''),
    'websocket_url': os.getenv('WEBSOCKET_URL', '')
}


# ==================== HEALTH & STATUS ====================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        is_open, market_status = AccurateMarketTimeManager.is_market_open()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            'market_status': market_status,
            'market_open': is_open,
            'bot_running': bot_manager.bot is not None and bot_manager.bot.running,
            'mode': current_mode,
            'features': [
                'real-time-websocket',
                'adaptive-strategies',
                'news-sentiment',
                'portfolio-risk-mgmt',
                'auto-square-off',
                'capital-tracking'
            ]
        }), 200
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/bot/status', methods=['GET'])
def bot_status():
    """Get detailed bot status"""
    try:
        status = bot_manager.get_status()
        
        return jsonify({
            'success': True,
            'data': {
                'running': status.get('running', False),
                'mode': status.get('mode', 'paper'),
                'market_status': status.get('market_status', 'unknown'),
                'market_open': status.get('market_open', False),
                'minutes_to_close': status.get('minutes_to_close', 0),
                'portfolio': status.get('portfolio', {}),
                'positions_count': status.get('positions', 0),
                'watchlist_size': status.get('watchlist_size', 0)
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== BOT CONTROL ====================

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    try:
        global current_mode
        
        # Get parameters from request
        data = request.get_json() or {}
        mode = data.get('mode', current_mode)
        capital = data.get('capital', INITIAL_CAPITAL)
        watchlist = data.get('watchlist', WATCHLIST)
        
        # Validate mode
        if mode not in ['paper', 'live']:
            return jsonify({
                'success': False,
                'error': 'Invalid mode. Use "paper" or "live"'
            }), 400
        
        # Check if already running
        if bot_manager.bot and bot_manager.bot.running:
            return jsonify({
                'success': False,
                'error': 'Bot already running'
            }), 400
        
        # Check market status
        is_open, market_status = AccurateMarketTimeManager.is_market_open()
        
        if not is_open and market_status != 'pre_market':
            logger.warning(f"Starting bot while market is {market_status}")
        
        # Start bot
        success = bot_manager.start(
            initial_capital=capital,
            watchlist=watchlist,
            api_keys=API_KEYS,
            mode=mode
        )
        
        if success:
            current_mode = mode
            logger.info(f"✅ Bot started in {mode} mode with ₹{capital:,.2f}")
            
            return jsonify({
                'success': True,
                'message': f'Bot started in {mode} mode',
                'data': {
                    'mode': mode,
                    'capital': capital,
                    'watchlist_size': len(watchlist),
                    'market_status': market_status
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start bot'
            }), 500
    
    except Exception as e:
        logger.error(f"Start bot error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    try:
        if not bot_manager.bot or not bot_manager.bot.running:
            return jsonify({
                'success': False,
                'error': 'Bot not running'
            }), 400
        
        # Get final metrics before stopping
        final_status = bot_manager.get_status()
        
        # Stop bot
        success = bot_manager.stop()
        
        if success:
            logger.info("✅ Bot stopped")
            
            return jsonify({
                'success': True,
                'message': 'Bot stopped',
                'data': {
                    'final_portfolio': final_status.get('portfolio', {})
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to stop bot'
            }), 500
    
    except Exception as e:
        logger.error(f"Stop bot error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== PORTFOLIO ====================

@app.route('/api/portfolio', methods=['GET'])
def portfolio():
    """Get portfolio metrics"""
    try:
        if not bot_manager.bot:
            # Return default portfolio
            return jsonify({
                'success': True,
                'data': {
                    'total_equity': INITIAL_CAPITAL,
                    'available_capital': INITIAL_CAPITAL,
                    'deployed_capital': 0.0,
                    'exposure_pct': 0.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'total_pnl': 0.0,
                    'total_pnl_pct': 0.0,
                    'open_positions': 0,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'max_drawdown': 0.0,
                    'circuit_breaker': False
                }
            }), 200
        
        status = bot_manager.get_status()
        portfolio_data = status.get('portfolio', {})
        
        return jsonify({
            'success': True,
            'data': portfolio_data
        }), 200
    
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/positions', methods=['GET'])
def positions():
    """Get open positions"""
    try:
        positions_data = bot_manager.get_positions()
        
        return jsonify({
            'success': True,
            'data': positions_data
        }), 200
    
    except Exception as e:
        logger.error(f"Positions error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trades', methods=['GET'])
def trades():
    """Get closed trades with statistics"""
    try:
        trades_data = bot_manager.get_trades()
        
        # Calculate statistics
        if trades_data:
            winning_trades = [t for t in trades_data if t['realized_pnl'] > 0]
            losing_trades = [t for t in trades_data if t['realized_pnl'] < 0]
            
            total_profit = sum(t['realized_pnl'] for t in winning_trades)
            total_loss = abs(sum(t['realized_pnl'] for t in losing_trades))
            
            stats = {
                'total_trades': len(trades_data),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(trades_data) * 100) if trades_data else 0,
                'total_pnl': sum(t['realized_pnl'] for t in trades_data),
                'avg_win': total_profit / len(winning_trades) if winning_trades else 0,
                'avg_loss': total_loss / len(losing_trades) if losing_trades else 0,
                'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
                'avg_holding_time': sum(t['holding_time_minutes'] for t in trades_data) / len(trades_data) if trades_data else 0
            }
        else:
            stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_holding_time': 0
            }
        
        return jsonify({
            'success': True,
            'data': trades_data,
            'statistics': stats
        }), 200
    
    except Exception as e:
        logger.error(f"Trades error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== SIGNALS & ANALYTICS ====================

@app.route('/api/signals', methods=['GET'])
def signals():
    """Get current trading signals"""
    try:
        # This would come from the bot's strategy analyzer
        # For now, return empty if bot not running
        if not bot_manager.bot or not bot_manager.bot.running:
            return jsonify({
                'success': True,
                'data': []
            }), 200
        
        # Get recent signals from bot's strategy
        # Implementation depends on how signals are stored
        signals_data = []
        
        return jsonify({
            'success': True,
            'data': signals_data
        }), 200
    
    except Exception as e:
        logger.error(f"Signals error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/market-regime', methods=['GET'])
def market_regime():
    """Get current market regime for each symbol"""
    try:
        if not bot_manager.bot:
            return jsonify({
                'success': True,
                'data': {}
            }), 200
        
        regimes = {}
        for symbol in WATCHLIST[:5]:  # Get regime for top 5 symbols
            try:
                regime = bot_manager.bot.data_fetcher.get_market_regime(symbol)
                regimes[symbol] = regime
            except:
                regimes[symbol] = 'unknown'
        
        return jsonify({
            'success': True,
            'data': regimes
        }), 200
    
    except Exception as e:
        logger.error(f"Market regime error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/news-sentiment', methods=['GET'])
def news_sentiment():
    """Get news sentiment analysis"""
    try:
        if not bot_manager.bot or not bot_manager.bot.news_articles:
            return jsonify({
                'success': True,
                'data': {
                    'market_sentiment': {'nifty': 0, 'banknifty': 0, 'market': 0},
                    'articles': [],
                    'top_movers': {'bullish': [], 'bearish': []}
                }
            }), 200
        
        # Get market sentiment
        market_sentiment = bot_manager.bot.news_analyzer.get_index_sentiment(
            bot_manager.bot.news_articles
        )
        
        # Get top movers
        movers = bot_manager.bot.news_analyzer.get_top_movers_news(
            bot_manager.bot.news_articles
        )
        
        # Get recent articles (top 10)
        recent_articles = [
            {
                'title': a.title,
                'summary': a.summary[:200],
                'source': a.source,
                'sentiment': a.sentiment_score,
                'published_at': a.published_at.isoformat(),
                'symbols': a.symbols[:3]  # Limit symbols
            }
            for a in bot_manager.bot.news_articles[:10]
        ]
        
        return jsonify({
            'success': True,
            'data': {
                'market_sentiment': market_sentiment,
                'articles': recent_articles,
                'top_movers': {
                    'bullish': [
                        {'symbol': m['symbol'], 'sentiment': m['sentiment']}
                        for m in movers['bullish'][:5]
                    ],
                    'bearish': [
                        {'symbol': m['symbol'], 'sentiment': m['sentiment']}
                        for m in movers['bearish'][:5]
                    ]
                }
            }
        }), 200
    
    except Exception as e:
        logger.error(f"News sentiment error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== CONFIGURATION ====================

@app.route('/api/config', methods=['GET'])
def config():
    """Get bot configuration"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'watchlist': WATCHLIST,
                'initial_capital': INITIAL_CAPITAL,
                'mode': current_mode,
                'features': {
                    'real_time_data': True,
                    'adaptive_strategies': True,
                    'news_sentiment': bool(API_KEYS.get('newsapi')),
                    'websocket': bool(API_KEYS.get('websocket_url')),
                    'auto_square_off': True,
                    'portfolio_risk_mgmt': True
                },
                'risk_limits': {
                    'max_risk_per_trade_pct': 2.0,
                    'max_positions': 5,
                    'max_daily_loss_pct': 5.0,
                    'square_off_time': '15:15',
                    'no_new_trades_after': '15:00'
                }
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Config error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mode', methods=['POST'])
def switch_mode():
    """Switch trading mode"""
    try:
        global current_mode
        
        if bot_manager.bot and bot_manager.bot.running:
            return jsonify({
                'success': False,
                'error': 'Stop bot before switching mode'
            }), 400
        
        data = request.get_json()
        new_mode = data.get('mode', 'paper')
        
        if new_mode not in ['paper', 'live']:
            return jsonify({
                'success': False,
                'error': 'Invalid mode'
            }), 400
        
        current_mode = new_mode
        logger.info(f"✅ Mode switched to {current_mode}")
        
        return jsonify({
            'success': True,
            'mode': current_mode
        }), 200
    
    except Exception as e:
        logger.error(f"Mode switch error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/paper/reset', methods=['POST'])
def reset_paper():
    """Reset paper trading"""
    try:
        if current_mode != 'paper':
            return jsonify({
                'success': False,
                'error': 'Can only reset in paper mode'
            }), 400
        
        if bot_manager.bot and bot_manager.bot.running:
            return jsonify({
                'success': False,
                'error': 'Stop bot before resetting'
            }), 400
        
        # Reset bot manager
        bot_manager.bot = None
        
        logger.info("✅ Paper trading reset")
        
        return jsonify({
            'success': True,
            'message': 'Paper trading reset'
        }), 200
    
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== RISK METRICS ====================

@app.route('/api/risk-metrics', methods=['GET'])
def risk_metrics():
    """Get risk and performance metrics"""
    try:
        if not bot_manager.bot:
            return jsonify({
                'success': True,
                'data': {
                    'exposure_pct': 0,
                    'risk_used_pct': 0,
                    'max_drawdown': 0,
                    'circuit_breaker': False,
                    'positions_vs_max': '0/5'
                }
            }), 200
        
        status = bot_manager.get_status()
        portfolio_data = status.get('portfolio', {})
        
        return jsonify({
            'success': True,
            'data': {
                'exposure_pct': portfolio_data.get('exposure_pct', 0),
                'risk_used_pct': (portfolio_data.get('deployed_capital', 0) / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0,
                'max_drawdown': portfolio_data.get('max_drawdown', 0),
                'circuit_breaker': portfolio_data.get('circuit_breaker', False),
                'positions_vs_max': f"{portfolio_data.get('open_positions', 0)}/5",
                'realized_pnl': portfolio_data.get('realized_pnl', 0),
                'unrealized_pnl': portfolio_data.get('unrealized_pnl', 0)
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Risk metrics error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("\n" + "="*100)
    logger.info("🚀 ENHANCED TRADING BOT V5.0 - PRODUCTION READY")
    logger.info("="*100)
    logger.info(f"📊 Initial Capital: ₹{INITIAL_CAPITAL:,.2f}")
    logger.info(f"📈 Watchlist: {len(WATCHLIST)} symbols")
    logger.info(f"⏰ Market Status: {AccurateMarketTimeManager.get_market_status()}")
    logger.info("")
    logger.info("✅ FEATURES ENABLED:")
    logger.info("   • Real-time WebSocket data feed")
    logger.info("   • Adaptive strategies with time-of-day weighting")
    logger.info("   • Market regime detection (volatile/trending/sideways)")
    logger.info("   • News & sentiment analysis integration")
    logger.info("   • Portfolio-level risk management")
    logger.info("   • Per-trade risk limits (1-2% equity)")
    logger.info("   • Total exposure caps (60% max)")
    logger.info("   • Daily loss limits with circuit breakers")
    logger.info("   • Auto square-off at 3:15 PM IST")
    logger.info("   • Proper capital tracking (realized + unrealized PnL)")
    logger.info("   • Dynamic position sizing with ATR")
    logger.info("")
    logger.info("="*100 + "\n")
    
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting Flask server on http://0.0.0.0:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )
