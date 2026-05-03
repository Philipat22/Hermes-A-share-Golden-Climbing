"""Strategy template for AHF backtesting.

Adapted from vnpy.alpha AlphaStrategy. Intended to be subclassed
with on_init() and on_bars() logic.

Usage:
    class MyStrategy(AlphaStrategy):
        def on_init(self):
            pass

        def on_bars(self, bars):
            signal = self.get_signal()
            for vt_symbol, bar in bars.items():
                if signal and signal > 0.3:
                    self.set_target(vt_symbol, 100)
                else:
                    self.set_target(vt_symbol, 0)
            self.execute_trading(bars, price_add=0.01)
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import polars as pl

from .trading_objects import Direction, Offset

if TYPE_CHECKING:
    from .engine import BacktestingEngine


class AlphaStrategy(ABC):
    """Strategy base class for backtesting"""

    def __init__(
        self,
        strategy_engine: "BacktestingEngine",
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        self.engine = strategy_engine
        self.name = strategy_name
        self.vt_symbols = vt_symbols

        # Position tracking
        self.pos_data: dict[str, float] = defaultdict(float)
        self.target_data: dict[str, float] = defaultdict(float)

        # Order tracking
        self.orders: dict = {}
        self.active_orderids: set[str] = set()

        # Apply custom settings
        for k, v in setting.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @abstractmethod
    def on_init(self) -> None:
        """Called once before backtest starts"""
        ...

    @abstractmethod
    def on_bars(self, bars: dict[str, "BarData"]) -> None:
        """Called on each bar slice (typically daily)"""
        ...

    def on_trade(self, trade) -> None:
        """Override for trade event handling (optional)"""
        pass

    # --- Order methods ---

    def buy(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        return self._send_order(vt_symbol, Direction.LONG, Offset.OPEN, price, volume)

    def sell(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        return self._send_order(vt_symbol, Direction.SHORT, Offset.CLOSE, price, volume)

    def short(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        return self._send_order(vt_symbol, Direction.SHORT, Offset.OPEN, price, volume)

    def cover(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        return self._send_order(vt_symbol, Direction.LONG, Offset.CLOSE, price, volume)

    def _send_order(
        self, vt_symbol: str, direction: Direction, offset: Offset,
        price: float, volume: float
    ) -> list[str]:
        order_ids = self.engine.send_order(self, vt_symbol, direction, offset, price, volume)
        self.active_orderids.update(order_ids)
        return order_ids

    def cancel_order(self, vt_orderid: str) -> None:
        self.engine.cancel_order(self, vt_orderid)

    def cancel_all(self) -> None:
        for oid in list(self.active_orderids):
            self.cancel_order(oid)

    def update_trade(self, trade) -> None:
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        else:
            self.pos_data[trade.vt_symbol] -= trade.volume
        self.on_trade(trade)

    def update_order(self, order) -> None:
        self.orders[order.vt_orderid] = order
        if not order.is_active() and order.vt_orderid in self.active_orderids:
            self.active_orderids.remove(order.vt_orderid)

    # --- Signal & position helpers ---

    def get_signal(self) -> pl.DataFrame:
        """Get current day's ML signal"""
        return self.engine.get_signal()

    def get_pos(self, vt_symbol: str) -> float:
        return self.pos_data.get(vt_symbol, 0.0)

    def get_target(self, vt_symbol: str) -> float:
        return self.target_data.get(vt_symbol, 0.0)

    def set_target(self, vt_symbol: str, target: float) -> None:
        self.target_data[vt_symbol] = target

    def execute_trading(self, bars: dict, price_add: float = 0.01) -> None:
        """Rebalance positions to match targets.
        price_add: slippage tolerance (1% by default).
        """
        self.cancel_all()
        for vt_symbol, bar in bars.items():
            target = self.get_target(vt_symbol)
            pos = self.get_pos(vt_symbol)
            diff = target - pos
            if abs(diff) < 1:
                continue

            if diff > 0:
                order_price = bar.close * (1 + price_add)
                cover_v = min(diff, abs(pos)) if pos < 0 else 0
                buy_v = diff - cover_v
                if cover_v:
                    self.cover(vt_symbol, order_price, cover_v)
                if buy_v:
                    self.buy(vt_symbol, order_price, buy_v)
            else:
                order_price = bar.close * (1 - price_add)
                sell_v = min(abs(diff), pos) if pos > 0 else 0
                short_v = abs(diff) - sell_v
                if sell_v:
                    self.sell(vt_symbol, order_price, sell_v)
                if short_v:
                    self.short(vt_symbol, order_price, short_v)

    def write_log(self, msg: str) -> None:
        self.engine.write_log(msg, self)

    def get_cash(self) -> float:
        return self.engine.cash

    def get_holding_value(self) -> float:
        return self.engine.get_holding_value()

    def get_portfolio_value(self) -> float:
        return self.get_cash() + self.get_holding_value()
