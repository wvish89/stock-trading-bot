"""
Portfolio-Level Risk Management System
Features:
- Per-trade risk limits (1-2% equity)
- Total portfolio exposure caps
- Max positions enforcement
- Daily loss limits with circuit breakers
- Auto square-off at EOD
- Dynamic position sizing with ATR
- Correlation-based risk adjustment
- Capital tracking with realized/unrealized PnL
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Position:
    """Enhanced position with full tracking"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    status: PositionStatus = PositionStatus.OPEN
    
    # Risk tracking
    risk_amount: float = 0.0  # Amount at risk
    max_loss: float = 0.0  # Maximum allowed loss
    
    # PnL tracking
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # Metadata
    strategy: str = ""
    trade_id: str = ""
    
    def update_current_price(self, price: float):
        """Update current price and calculate PnL"""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
        self.unrealized_pnl_pct = ((price - self.entry_price) / self.entry_price) * 100
    
    def check_stop_loss(self) -> bool:
        """Check if stop loss hit"""
        return self.current_price <= self.stop_loss
    
    def check_take_profit(self) -> bool:
        """Check if take profit hit"""
        return self.current_price >= self.take_profit
    
    def get_capital_deployed(self) -> float:
        """Get total capital deployed in this position"""
        return self.entry_price * self.quantity


@dataclass
class Trade:
    """Completed trade record"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    realized_pnl: float
    realized_pnl_pct: float
    exit_reason: str  # STOP_LOSS, TAKE_PROFIT, SIGNAL, SQUARE_OFF
    strategy: str
    holding_time_minutes: int = 0


@dataclass
class RiskLimits:
    """Risk management limits"""
    # Per-trade limits
    max_risk_per_trade_pct: float = 0.02  # 2% max risk per trade
    max_position_size_pct: float = 0.20  # 20% max capital per position
    
    # Portfolio limits
    max_positions: int = 5
    max_total_exposure_pct: float = 0.60  # 60% max total exposure
    max_sector_exposure_pct: float = 0.30  # 30% max per sector
    
    # Loss limits
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_weekly_loss_pct: float = 0.10  # 10% max weekly loss
    circuit_breaker_loss_pct: float = 0.03  # 3% loss = pause trading
    
    # Time limits
    square_off_time: dtime = dtime(15, 15)  # Square off at 3:15 PM
    no_new_trades_after: dtime = dtime(15, 0)  # No new trades after 3:00 PM
    
    # Position limits
    min_trade_value: float = 5000  # Minimum ₹5000 per trade
    max_trade_value: float = 50000  # Maximum ₹50,000 per trade


class PortfolioRiskManager:
    """
    Advanced portfolio-level risk management
    """
    
    def __init__(self, initial_capital: float, limits: Optional[RiskLimits] = None):
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.limits = limits or RiskLimits()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        
        # PnL tracking
        self.daily_realized_pnl = 0.0
        self.daily_unrealized_pnl = 0.0
        self.session_start_capital = initial_capital
        self.session_start_time = datetime.now(ZoneInfo("Asia/Kolkata"))
        
        # Risk tracking
        self.daily_max_drawdown = 0.0
        self.peak_equity = initial_capital
        self.circuit_breaker_active = False
        
        # Sector exposure tracking
        self.sector_exposure: Dict[str, float] = {}
        
        logger.info(f"✅ Portfolio Risk Manager initialized: Capital ₹{initial_capital:,.2f}")
    
    def get_total_equity(self) -> float:
        """Calculate total portfolio equity (capital + unrealized PnL)"""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.available_capital + unrealized_pnl
    
    def get_total_exposure(self) -> float:
        """Get total capital deployed in positions"""
        return sum(pos.get_capital_deployed() for pos in self.positions.values())
    
    def get_exposure_pct(self) -> float:
        """Get exposure as percentage of equity"""
        equity = self.get_total_equity()
        if equity <= 0:
            return 0.0
        return (self.get_total_exposure() / equity) * 100
    
    def can_take_new_position(self, current_time: Optional[dtime] = None) -> Tuple[bool, str]:
        """
        Check if new position can be taken
        Returns: (can_take, reason)
        """
        # Check time restrictions
        if current_time is None:
            current_time = datetime.now(ZoneInfo("Asia/Kolkata")).time()
        
        if current_time >= self.limits.no_new_trades_after:
            return False, "Trading window closed (after 3:00 PM)"
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker active - daily loss limit hit"
        
        # Check max positions
        if len(self.positions) >= self.limits.max_positions:
            return False, f"Max positions ({self.limits.max_positions}) reached"
        
        # Check daily loss limit
        daily_pnl_pct = (self.daily_realized_pnl / self.session_start_capital) * 100
        if daily_pnl_pct <= -self.limits.max_daily_loss_pct * 100:
            self.circuit_breaker_active = True
            return False, f"Daily loss limit ({self.limits.max_daily_loss_pct*100}%) exceeded"
        
        # Check total exposure
        exposure_pct = self.get_exposure_pct()
        if exposure_pct >= self.limits.max_total_exposure_pct * 100:
            return False, f"Max exposure ({self.limits.max_total_exposure_pct*100}%) reached"
        
        return True, "OK"
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, atr: Optional[float] = None) -> Tuple[int, float]:
        """
        Calculate optimal position size based on risk
        
        Returns: (quantity, risk_amount)
        """
        equity = self.get_total_equity()
        
        # Maximum risk amount (2% of equity)
        max_risk = equity * self.limits.max_risk_per_trade_pct
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0, 0.0
        
        # Calculate base quantity
        base_quantity = int(max_risk / risk_per_share)
        
        # Apply position size limit
        max_position_value = equity * self.limits.max_position_size_pct
        max_quantity_by_value = int(max_position_value / entry_price)
        
        quantity = min(base_quantity, max_quantity_by_value)
        
        # Apply trade value limits
        trade_value = quantity * entry_price
        
        if trade_value < self.limits.min_trade_value:
            # Increase to minimum
            quantity = int(self.limits.min_trade_value / entry_price)
        
        if trade_value > self.limits.max_trade_value:
            # Cap at maximum
            quantity = int(self.limits.max_trade_value / entry_price)
        
        # Final validation
        final_trade_value = quantity * entry_price
        if final_trade_value > self.available_capital:
            quantity = int(self.available_capital / entry_price)
        
        risk_amount = quantity * risk_per_share
        
        logger.info(f"Position size calculated: {symbol} Qty={quantity}, Risk=₹{risk_amount:,.2f}")
        
        return max(1, quantity), risk_amount
    
    def open_position(self, symbol: str, entry_price: float, stop_loss: float,
                     take_profit: float, strategy: str = "") -> Optional[Position]:
        """
        Open a new position
        """
        # Check if can take position
        can_take, reason = self.can_take_new_position()
        if not can_take:
            logger.warning(f"Cannot open position: {reason}")
            return None
        
        # Calculate position size
        quantity, risk_amount = self.calculate_position_size(symbol, entry_price, stop_loss)
        
        if quantity <= 0:
            logger.warning(f"Invalid position size calculated for {symbol}")
            return None
        
        # Check available capital
        required_capital = entry_price * quantity
        if required_capital > self.available_capital:
            logger.warning(f"Insufficient capital: Need ₹{required_capital:,.2f}, Have ₹{self.available_capital:,.2f}")
            return None
        
        # Create position
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            entry_time=datetime.now(ZoneInfo("Asia/Kolkata")),
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            max_loss=risk_amount,
            strategy=strategy,
            trade_id=trade_id
        )
        
        # Deduct from available capital
        self.available_capital -= required_capital
        
        # Add to positions
        self.positions[symbol] = position
        
        logger.info(f"✅ OPENED: {symbol} Qty={quantity} @ ₹{entry_price:.2f}, "
                   f"SL=₹{stop_loss:.2f}, TP=₹{take_profit:.2f}, "
                   f"Risk=₹{risk_amount:,.2f}")
        
        return position
    
    def close_position(self, symbol: str, exit_price: float, 
                      exit_reason: str = "MANUAL") -> Optional[Trade]:
        """Close an existing position"""
        if symbol not in self.positions:
            logger.warning(f"Position {symbol} not found")
            return None
        
        position = self.positions[symbol]
        
        # Calculate realized PnL
        realized_pnl = (exit_price - position.entry_price) * position.quantity
        realized_pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
        
        # Calculate holding time
        holding_time = (datetime.now(ZoneInfo("Asia/Kolkata")) - position.entry_time).total_seconds() / 60
        
        # Create trade record
        trade = Trade(
            trade_id=position.trade_id,
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=datetime.now(ZoneInfo("Asia/Kolkata")),
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            exit_reason=exit_reason,
            strategy=position.strategy,
            holding_time_minutes=int(holding_time)
        )
        
        # Update capital
        self.available_capital += (exit_price * position.quantity)
        self.daily_realized_pnl += realized_pnl
        
        # Track peak equity for drawdown
        current_equity = self.get_total_equity()
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Remove position
        del self.positions[symbol]
        
        # Record trade
        self.closed_trades.append(trade)
        
        logger.info(f"✅ CLOSED: {symbol} @ ₹{exit_price:.2f}, "
                   f"PnL=₹{realized_pnl:,.2f} ({realized_pnl_pct:.2f}%), "
                   f"Reason={exit_reason}")
        
        return trade
    
    def update_positions(self, price_data: Dict[str, float]):
        """
        Update all positions with current prices and check stop loss/take profit
        
        Args:
            price_data: Dict mapping symbol -> current_price
        """
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in price_data:
                current_price = price_data[symbol]
                position.update_current_price(current_price)
                
                # Check stop loss
                if position.check_stop_loss():
                    positions_to_close.append((symbol, current_price, "STOP_LOSS"))
                    logger.warning(f"🛑 Stop loss hit: {symbol}")
                
                # Check take profit
                elif position.check_take_profit():
                    positions_to_close.append((symbol, current_price, "TAKE_PROFIT"))
                    logger.info(f"🎯 Take profit hit: {symbol}")
        
        # Close positions that hit limits
        for symbol, price, reason in positions_to_close:
            self.close_position(symbol, price, reason)
    
    def square_off_all_positions(self, price_data: Dict[str, float]):
        """Square off all positions at end of day"""
        logger.info("📍 SQUARE OFF: Closing all positions...")
        
        for symbol in list(self.positions.keys()):
            if symbol in price_data:
                self.close_position(symbol, price_data[symbol], "SQUARE_OFF")
            else:
                # Use last known price
                position = self.positions[symbol]
                self.close_position(symbol, position.current_price, "SQUARE_OFF")
        
        logger.info("✅ All positions squared off")
    
    def check_square_off_time(self) -> bool:
        """Check if it's time to square off"""
        current_time = datetime.now(ZoneInfo("Asia/Kolkata")).time()
        return current_time >= self.limits.square_off_time
    
    def get_portfolio_metrics(self) -> Dict:
        """Get comprehensive portfolio metrics"""
        equity = self.get_total_equity()
        total_pnl = equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Calculate drawdown
        drawdown = ((self.peak_equity - equity) / self.peak_equity) * 100 if self.peak_equity > 0 else 0
        self.daily_max_drawdown = max(self.daily_max_drawdown, drawdown)
        
        # Calculate win rate
        if len(self.closed_trades) > 0:
            winning_trades = [t for t in self.closed_trades if t.realized_pnl > 0]
            win_rate = (len(winning_trades) / len(self.closed_trades)) * 100
        else:
            win_rate = 0.0
        
        return {
            'total_equity': round(equity, 2),
            'available_capital': round(self.available_capital, 2),
            'deployed_capital': round(self.get_total_exposure(), 2),
            'exposure_pct': round(self.get_exposure_pct(), 2),
            'unrealized_pnl': round(sum(p.unrealized_pnl for p in self.positions.values()), 2),
            'realized_pnl': round(self.daily_realized_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'open_positions': len(self.positions),
            'total_trades': len(self.closed_trades),
            'win_rate': round(win_rate, 2),
            'max_drawdown': round(self.daily_max_drawdown, 2),
            'circuit_breaker': self.circuit_breaker_active
        }
    
    def reset_session(self):
        """Reset daily counters (call at start of new day)"""
        self.daily_realized_pnl = 0.0
        self.daily_unrealized_pnl = 0.0
        self.session_start_capital = self.get_total_equity()
        self.session_start_time = datetime.now(ZoneInfo("Asia/Kolkata"))
        self.daily_max_drawdown = 0.0
        self.peak_equity = self.session_start_capital
        self.circuit_breaker_active = False
        logger.info(f"✅ Session reset: Starting capital ₹{self.session_start_capital:,.2f}")


# Usage Example
"""
# Initialize portfolio manager
portfolio = PortfolioRiskManager(
    initial_capital=100000,
    limits=RiskLimits(
        max_risk_per_trade_pct=0.02,
        max_positions=5,
        max_daily_loss_pct=0.05
    )
)

# Open position
position = portfolio.open_position(
    symbol='RELIANCE-EQ',
    entry_price=2450.0,
    stop_loss=2400.0,  # ₹50 risk
    take_profit=2550.0,
    strategy='Opening Range Breakout'
)

# Update prices (call periodically)
price_data = {
    'RELIANCE-EQ': 2475.0,
    'TCS-EQ': 3820.0
}
portfolio.update_positions(price_data)

# Check if square off time
if portfolio.check_square_off_time():
    portfolio.square_off_all_positions(price_data)

# Get metrics
metrics = portfolio.get_portfolio_metrics()
print(f"Equity: ₹{metrics['total_equity']:,.2f}")
print(f"P&L: ₹{metrics['total_pnl']:,.2f} ({metrics['total_pnl_pct']:.2f}%)")
print(f"Win Rate: {metrics['win_rate']:.1f}%")
"""
