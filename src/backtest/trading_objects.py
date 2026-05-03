"""Minimal trading data objects for backtesting.

Ported from vnpy.alpha concepts, stripped of vnpy.trader dependencies.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional


class Direction(Enum):
    """Order/trade direction"""
    LONG = "多"
    SHORT = "空"


class Offset(Enum):
    """Order offset (open/close)"""
    OPEN = "开"
    CLOSE = "平"


class OrderStatus(Enum):
    """Order lifecycle status"""
    SUBMITTING = "提交中"
    NOTTRADED = "未成交"
    ALLTRADED = "全部成交"
    CANCELLED = "已撤销"


@dataclass
class BarData:
    """Single OHLCV bar"""
    vt_symbol: str
    datetime: datetime
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    turnover: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.close == 0.0


@dataclass
class OrderData:
    """Limit order"""
    vt_orderid: str = ""
    vt_symbol: str = ""
    direction: Direction = Direction.LONG
    offset: Offset = Offset.OPEN
    price: float = 0.0
    volume: float = 0.0
    traded: float = 0.0
    status: OrderStatus = OrderStatus.SUBMITTING
    datetime: Optional[datetime] = None

    def is_active(self) -> bool:
        return self.status in (OrderStatus.SUBMITTING, OrderStatus.NOTTRADED)


@dataclass
class TradeData:
    """Filled trade"""
    vt_tradeid: str = ""
    vt_symbol: str = ""
    direction: Direction = Direction.LONG
    offset: Offset = Offset.OPEN
    price: float = 0.0
    volume: float = 0.0
    datetime: Optional[datetime] = None


# --- Daily PnL tracking (adapted from vnpy.alpha) ---

@dataclass
class ContractDailyResult:
    """Single-contract daily PnL"""
    date: date
    close_price: float = 0.0
    pre_close: float = 0.0
    trades: list = field(default_factory=list)
    trade_count: int = 0
    start_pos: float = 0.0
    end_pos: float = 0.0
    turnover: float = 0.0
    commission: float = 0.0
    trading_pnl: float = 0.0
    holding_pnl: float = 0.0
    total_pnl: float = 0.0
    net_pnl: float = 0.0

    def add_trade(self, trade: TradeData) -> None:
        self.trades.append(trade)

    def calculate_pnl(
        self, pre_close: float, start_pos: float,
        size: int, rate: float
    ) -> None:
        self.pre_close = pre_close if pre_close else 0.0
        self.start_pos = start_pos
        self.end_pos = start_pos
        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size
        self.trade_count = len(self.trades)

        for t in self.trades:
            pos_change = t.volume if t.direction == Direction.LONG else -t.volume
            self.end_pos += pos_change
            turnover = t.volume * size * t.price
            self.trading_pnl += pos_change * (self.close_price - t.price) * size
            self.turnover += turnover
            self.commission += turnover * rate

        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission

    def update_close_price(self, close_price: float) -> None:
        self.close_price = close_price


@dataclass
class PortfolioDailyResult:
    """Multi-contract daily PnL aggregation"""
    date: date
    close_prices: dict = field(default_factory=dict)
    pre_closes: dict = field(default_factory=dict)
    start_poses: dict = field(default_factory=dict)
    end_poses: dict = field(default_factory=dict)
    contract_results: dict = field(default_factory=dict)
    trade_count: int = 0
    turnover: float = 0.0
    commission: float = 0.0
    trading_pnl: float = 0.0
    holding_pnl: float = 0.0
    total_pnl: float = 0.0
    net_pnl: float = 0.0

    def __post_init__(self):
        for vt_symbol, cp in self.close_prices.items():
            self.contract_results[vt_symbol] = ContractDailyResult(
                date=self.date, close_price=cp
            )

    def add_trade(self, trade: TradeData) -> None:
        cr = self.contract_results.get(trade.vt_symbol)
        if cr:
            cr.add_trade(trade)

    def calculate_pnl(
        self, pre_closes: dict, start_poses: dict,
        sizes: dict, rate: float
    ) -> None:
        self.pre_closes = pre_closes
        self.start_poses = start_poses
        for vt_symbol, cr in self.contract_results.items():
            cr.calculate_pnl(
                pre_closes.get(vt_symbol, 0),
                start_poses.get(vt_symbol, 0),
                sizes.get(vt_symbol, 100),
                rate
            )
            self.trade_count += cr.trade_count
            self.turnover += cr.turnover
            self.commission += cr.commission
            self.trading_pnl += cr.trading_pnl
            self.holding_pnl += cr.holding_pnl
            self.total_pnl += cr.total_pnl
            self.net_pnl += cr.net_pnl
            self.end_poses[vt_symbol] = cr.end_pos

    def update_close_prices(self, close_prices: dict) -> None:
        self.close_prices.update(close_prices)
        for vt_symbol, cp in close_prices.items():
            cr = self.contract_results.get(vt_symbol)
            if cr:
                cr.update_close_price(cp)
            else:
                self.contract_results[vt_symbol] = ContractDailyResult(
                    date=self.date, close_price=cp
                )
