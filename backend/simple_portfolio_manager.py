"""
Simple Portfolio & Risk Manager
Tracks positions, calculates P&L, manages risk
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class SimplePortfolioManager:
    """Portfolio and risk management"""
    
    def __init__(self, initial_capital=100000, max_positions=5, 
                 risk_per_trade=0.02, max_daily_loss=0.05):
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss
        
        # Tracking
        self.positions = {}  # symbol -> position dict
        self.trades = []  # completed trades
        self.daily_realized_pnl = 0.0
        
        logger.info(f"✅ Portfolio Manager initialized: ₹{initial_capital:,.2f}")
    
    def can_take_position(self):
        """Check if can take new position"""
        # Max positions check
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"
        
        # Daily loss check
        equity = self.get_total_equity()
        daily_loss_pct = (self.daily_realized_pnl / self.initial_capital) * 100
        
        if daily_loss_pct <= -self.max_daily_loss * 100:
            return False, "Daily loss limit hit"
        
        # Capital check
        if self.available_capital < 5000:
            return False, "Insufficient capital"
        
        return True, "OK"
    
    def calculate_position_size(self, price, stop_loss):
        """Calculate position size based on risk"""
        equity = self.get_total_equity()
        max_risk = equity * self.risk_per_trade
        
        risk_per_share = abs(price - stop_loss)
        if risk_per_share <= 0:
            return 0
        
        quantity = int(max_risk / risk_per_share)
        
        # Apply limits
        max_position_value = equity * 0.20  # 20% max per position
        max_qty_by_value = int(max_position_value / price)
        
        quantity = min(quantity, max_qty_by_value)
        
        # Min/max trade value
        trade_value = quantity * price
        if trade_value < 5000:
            quantity = int(5000 / price)
        if trade_value > 50000:
            quantity = int(50000 / price)
        
        # Final check against available capital
        if quantity * price > self.available_capital:
            quantity = int(self.available_capital / price)
        
        return max(1, quantity)
    
    def open_position(self, symbol, quantity, entry_price, stop_loss, 
                     take_profit, strategy=""):
        """Open a new position"""
        try:
            cost = quantity * entry_price
            
            if cost > self.available_capital:
                logger.warning(f"Insufficient capital for {symbol}")
                return False
            
            # Deduct capital
            self.available_capital -= cost
            
            # Create position
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': strategy,
                'entry_time': datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                'unrealized_pnl': 0.0
            }
            
            logger.info(f"Position opened: {symbol} x{quantity} @ ₹{entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Open position error: {e}")
            return False
    
    def close_position(self, symbol, exit_price, reason="MANUAL"):
        """Close an existing position"""
        try:
            if symbol not in self.positions:
                return None
            
            pos = self.positions[symbol]
            quantity = pos['quantity']
            entry_price = pos['entry_price']
            
            # Calculate P&L
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            
            # Return capital
            proceeds = exit_price * quantity
            self.available_capital += proceeds
            self.daily_realized_pnl += pnl
            
            # Record trade
            trade = {
                'symbol': symbol,
                'qty': quantity,
                'price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_time': pos['entry_time'],
                'exit_time': datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                'reason': reason,
                'strategy': pos['strategy']
            }
            self.trades.append(trade)
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"Position closed: {symbol}, P&L: ₹{pnl:,.2f} ({pnl_pct:.2f}%)")
            return trade
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return None
    
    def update_position_price(self, symbol, current_price):
        """Update position with current price and check SL/TP"""
        try:
            if symbol not in self.positions:
                return
            
            pos = self.positions[symbol]
            pos['current_price'] = current_price
            
            # Calculate unrealized P&L
            pnl = (current_price - pos['entry_price']) * pos['quantity']
            pos['unrealized_pnl'] = pnl
            
            # Check stop loss
            if current_price <= pos['stop_loss']:
                logger.warning(f"Stop loss hit: {symbol}")
                self.close_position(symbol, current_price, "STOP_LOSS")
            
            # Check take profit
            elif current_price >= pos['take_profit']:
                logger.info(f"Take profit hit: {symbol}")
                self.close_position(symbol, current_price, "TAKE_PROFIT")
                
        except Exception as e:
            logger.error(f"Update position error: {e}")
    
    def has_position(self, symbol):
        """Check if has position in symbol"""
        return symbol in self.positions
    
    def get_total_equity(self):
        """Get total portfolio value"""
        unrealized = sum(p['unrealized_pnl'] for p in self.positions.values())
        return self.available_capital + unrealized
    
    def get_deployed_capital(self):
        """Get capital deployed in positions"""
        return sum(p['entry_price'] * p['quantity'] for p in self.positions.values())
    
    def get_metrics(self):
        """Get portfolio metrics"""
        equity = self.get_total_equity()
        deployed = self.get_deployed_capital()
        unrealized = sum(p['unrealized_pnl'] for p in self.positions.values())
        
        total_pnl = equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Win rate
        if len(self.trades) > 0:
            winning = sum(1 for t in self.trades if t['pnl'] > 0)
            win_rate = (winning / len(self.trades)) * 100
        else:
            win_rate = 0.0
        
        return {
            'total_value': round(equity, 2),
            'available_capital': round(self.available_capital, 2),
            'deployed_capital': round(deployed, 2),
            'exposure_pct': round((deployed / equity * 100) if equity > 0 else 0, 2),
            'unrealized_pnl': round(unrealized, 2),
            'realized_pnl': round(self.daily_realized_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'open_positions': len(self.positions),
            'total_trades': len(self.trades),
            'win_rate': round(win_rate, 2)
        }
    
    def get_positions_list(self):
        """Get list of positions for API"""
        positions_list = []
        
        for symbol, pos in self.positions.items():
            positions_list.append({
                'symbol': symbol,
                'quantity': pos['quantity'],
                'price': pos['entry_price'],
                'current_price': pos['current_price'],
                'pnl': pos['unrealized_pnl'],
                'entry_time': pos['entry_time'],
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit']
            })
        
        return positions_list
    
    def get_trades_list(self):
        """Get list of trades for API"""
        return self.trades[-20:]  # Last 20 trades
    
    def get_trade_statistics(self):
        """Get trade statistics"""
        if len(self.trades) > 0:
            winning = sum(1 for t in self.trades if t['pnl'] > 0)
            win_rate = (winning / len(self.trades)) * 100
            total_pnl = sum(t['pnl'] for t in self.trades)
        else:
            win_rate = 0.0
            total_pnl = 0.0
        
        return {
            'total_trades': len(self.trades),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2)
        }
    
    def reset(self):
        """Reset portfolio"""
        self.available_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_realized_pnl = 0.0
        logger.info("Portfolio reset")
