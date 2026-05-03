"""Full portfolio-level backtesting engine for AHF.

Adapted from vnpy.alpha BacktestingEngine. Provides:
  - Order matching against daily OHLC
  - Position tracking
  - Daily mark-to-market PnL
  - Transaction costs and A-share limits (10% daily band)
  - Full statistics (Sharpe, drawdown, win rate, etc.)
  - Equity curve + drawdown charts
  - Benchmark comparison

Core integration with AHF pipeline:
  - Takes prices from data_fetcher (DataFrame with OHLCV per symbol)
  - Takes signals from predictor.py / runner.py signal DataFrame
  - Uses a pluggable AlphaStrategy with on_bars() callback
"""
import traceback
from datetime import datetime, date
from collections import defaultdict
from typing import Optional, Type

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from .trading_objects import (
    Direction, Offset, OrderStatus,
    BarData, OrderData, TradeData,
    PortfolioDailyResult,
)


class BacktestingEngine:
    """Full-featured backtesting engine for A-shares"""

    def __init__(self):
        # --- Parameters ---
        self.vt_symbols: list[str] = []
        self.start: Optional[datetime] = None
        self.end: Optional[datetime] = None
        self.capital: float = 1_000_000
        self.risk_free: float = 0.03        # 3% annual risk-free rate
        self.annual_days: int = 240
        self.rate: float = 0.0003            # 万3 commission
        self.slippage: float = 0.001         # 0.1% slippage
        self.sizes: dict[str, int] = {}      # contract size (100 for A-shares)
        self.limit_up_pct: float = 0.10      # A-share daily limit
        self.limit_down_pct: float = -0.10

        # --- Strategy ---
        self.strategy_class: Optional[Type] = None
        self.strategy = None
        self.signal_df: Optional[pl.DataFrame] = None  # pre-computed signals

        # --- Data ---
        self.price_data: dict[str, pl.DataFrame] = {}  # {vt_symbol: OHLCV df}
        self.all_dates: list[datetime] = []             # sorted unique trading days
        self.bars: dict[str, BarData] = {}              # current day's bars

        # --- Order matching ---
        self.order_counter: int = 0
        self.limit_orders: dict[str, OrderData] = {}
        self.active_limit_orders: dict[str, OrderData] = {}
        self.pre_closes: dict[str, float] = defaultdict(float)

        # --- Trade tracking ---
        self.trade_counter: int = 0
        self.trades: dict[str, TradeData] = {}

        # --- PnL ---
        self.cash: float = 0.0
        self.daily_results: dict[date, PortfolioDailyResult] = {}
        self.daily_df: Optional[pl.DataFrame] = None

        # --- Logs ---
        self.logs: list[str] = []

    # ---- Public setup API ----

    def set_parameters(
        self,
        vt_symbols: list[str],
        start: datetime, end: datetime,
        capital: float = 1_000_000,
        risk_free: float = 0.03,
        rate: float = 0.0003,
        slippage: float = 0.001,
        annual_days: int = 240,
    ) -> None:
        self.vt_symbols = vt_symbols
        self.start = start
        self.end = end
        self.capital = capital
        self.cash = capital
        self.risk_free = risk_free
        self.rate = rate
        self.slippage = slippage
        self.annual_days = annual_days
        for s in vt_symbols:
            self.sizes[s] = 100  # A-shares: 1 lot = 100 shares

    def add_strategy(
        self, strategy_class: Type,
        setting: dict, signal_df: Optional[pl.DataFrame] = None
    ) -> None:
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, list(self.vt_symbols), setting
        )
        self.signal_df = signal_df

    def load_price_data(self, price_data: dict[str, pl.DataFrame]) -> None:
        """Inject pre-loaded OHLCV data.

        Each df must have columns:
            datetime, open, high, low, close, volume, turnover
        Index must contain the trading days in order.
        """
        self.price_data = price_data

        # Build sorted trading day list
        all_dates: set[datetime] = set()
        for s, df in price_data.items():
            for dt in df["datetime"]:
                all_dates.add(dt)
        self.all_dates = sorted(all_dates)

    # ---- Main backtest loop ----

    def run_backtesting(self) -> None:
        """Execute the full backtest."""
        if not self.strategy:
            raise RuntimeError("Strategy not set. Call add_strategy() first.")
        if not self.price_data:
            raise RuntimeError("Price data not loaded. Call load_price_data() first.")

        # Init strategy
        self.strategy.on_init()
        self.write_log("Strategy initialized")

        # Convert price data to dict-of-BarData lookup for speed
        bar_lookup: dict[tuple, BarData] = {}
        for s, df in self.price_data.items():
            for row in df.iter_rows(named=True):
                dt = row["datetime"]
                if isinstance(dt, str):
                    dt = datetime.fromisoformat(dt)
                bar_lookup[(dt, s)] = BarData(
                    vt_symbol=s,
                    datetime=dt,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row.get("volume", 0),
                    turnover=row.get("turnover", 0),
                )

        self.write_log(f"Backtesting {len(self.vt_symbols)} symbols over {len(self.all_dates)} trading days")

        for dt in tqdm(self.all_dates, desc="Backtesting"):
            try:
                self._new_day(dt, bar_lookup)
            except Exception:
                self.write_log(f"Error at {dt}, aborting")
                self.write_log(traceback.format_exc())
                return

        self.write_log("Backtest complete")

    # ---- Internal: per-day processing ----

    def _new_day(self, dt: datetime, bar_lookup: dict) -> None:
        """Process one trading day."""
        self.current_dt = dt

        # Load bars for this day
        bars: dict[str, BarData] = {}
        for s in self.vt_symbols:
            last_bar = self.bars.get(s)
            if last_bar and last_bar.close:
                self.pre_closes[s] = last_bar.close

            bar = bar_lookup.get((dt, s))
            if bar:
                self.bars[s] = bar
                bars[s] = bar
            elif s in self.bars:
                old = self.bars[s]
                fill = BarData(
                    vt_symbol=s, datetime=dt,
                    open=old.close, high=old.close,
                    low=old.close, close=old.close,
                )
                self.bars[s] = fill
                bars[s] = fill

        # Match orders against today's bars
        self._cross_orders()

        # Call strategy
        if self.strategy:
            self.strategy.on_bars(bars)

        # Update daily PnL
        self._update_daily_close(dt)

    def _cross_orders(self) -> None:
        """Match active limit orders against today's OHLC."""
        for order in list(self.active_limit_orders.values()):
            bar = self.bars.get(order.vt_symbol)
            if not bar or bar.is_empty:
                continue

            pre_close = self.pre_closes.get(order.vt_symbol, 0)
            if pre_close == 0:
                continue

            limit_up = pre_close * (1 + self.limit_up_pct)
            limit_down = pre_close * (1 + self.limit_down_pct)

            long_cross = (
                order.direction == Direction.LONG
                and order.price >= bar.low
                and bar.low > 0
                and bar.low < limit_up       # not limit-up all day
            )
            short_cross = (
                order.direction == Direction.SHORT
                and order.price <= bar.high
                and bar.high > 0
                and bar.high > limit_down    # not limit-down all day
            )

            if not long_cross and not short_cross:
                continue

            # Fill order
            order.traded = order.volume
            order.status = OrderStatus.ALLTRADED
            self.strategy.update_order(order)
            self.active_limit_orders.pop(order.vt_orderid, None)

            # Generate trade
            self.trade_counter += 1
            trade_price = (
                min(order.price, bar.open) if long_cross
                else max(order.price, bar.open)
            )

            trade = TradeData(
                vt_tradeid=f"trade_{self.trade_counter}",
                vt_symbol=order.vt_symbol,
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.current_dt,
            )

            size = self.sizes.get(order.vt_symbol, 100)
            trade_turnover = trade.price * trade.volume * size
            commission = trade_turnover * self.rate

            if trade.direction == Direction.LONG:
                self.cash -= trade_turnover
            else:
                self.cash += trade_turnover
            self.cash -= commission

            self.strategy.update_trade(trade)
            self.trades[trade.vt_tradeid] = trade

    def _update_daily_close(self, dt: datetime) -> None:
        """Record end-of-day positions and prices for PnL calculation."""
        d = dt.date() if hasattr(dt, "date") else dt
        close_prices = {}
        for s in self.vt_symbols:
            bar = self.bars.get(s)
            close_prices[s] = bar.close if bar else self.pre_closes.get(s, 0)

        daily = self.daily_results.get(d)
        if daily:
            daily.update_close_prices(close_prices)
        else:
            self.daily_results[d] = PortfolioDailyResult(d, close_prices)



    # ---- PnL calculation ----

    def calculate_result(self) -> Optional[pl.DataFrame]:
        """Compute daily mark-to-market PnL for all trading days."""
        if not self.trades:
            self.write_log("No trades, cannot calculate result")
            return None

        for trade in self.trades.values():
            if not trade.datetime:
                continue
            d = trade.datetime.date()
            daily = self.daily_results.get(d)
            if daily:
                daily.add_trade(trade)

        pre_closes: dict = {}
        start_poses: dict = {}

        for d in sorted(self.daily_results.keys()):
            daily = self.daily_results[d]
            daily.calculate_pnl(pre_closes, start_poses, self.sizes, self.rate)
            pre_closes = daily.close_prices
            start_poses = daily.end_poses

        rows = []
        for d in sorted(self.daily_results.keys()):
            dr = self.daily_results[d]
            rows.append({
                "date": d,
                "trade_count": dr.trade_count,
                "turnover": dr.turnover,
                "commission": dr.commission,
                "trading_pnl": dr.trading_pnl,
                "holding_pnl": dr.holding_pnl,
                "total_pnl": dr.total_pnl,
                "net_pnl": dr.net_pnl,
            })

        self.daily_df = pl.DataFrame(rows)
        return self.daily_df

    def calculate_statistics(self) -> dict:
        """Compute performance statistics."""
        if self.daily_df is None:
            self.calculate_result()

        df = self.daily_df
        stats: dict = {}

        if df is None or len(df) == 0:
            return stats

        df = df.with_columns(
            balance=(pl.col("net_pnl").cum_sum() + self.capital).alias("balance"),
        ).with_columns(
            pct=(pl.col("balance").pct_change().fill_null(0).alias("return")),
        ).with_columns(
            highlevel=pl.col("balance").cum_max(),
        ).with_columns(
            drawdown=pl.col("balance") - pl.col("highlevel"),
            ddpercent=(pl.col("balance") / pl.col("highlevel") - 1) * 100,
        )

        self.daily_df = df

        # Check bankruptcy
        if (df["balance"] <= 0).any():
            self.write_log("Bankruptcy detected, statistics may be incomplete")

        total_days = len(df)
        profit_days = df.filter(pl.col("net_pnl") > 0).height
        loss_days = df.filter(pl.col("net_pnl") < 0).height

        end_balance = df["balance"][-1]
        max_drawdown = float(df["drawdown"].min())
        max_ddpercent = float(df["ddpercent"].min())

        max_dd_end_idx = int(df["drawdown"].arg_min())
        max_dd_end = df["date"][max_dd_end_idx]
        max_dd_start = df.slice(0, max_dd_end_idx + 1)["balance"].arg_max()
        max_dd_start_date = df["date"][max_dd_start]
        max_dd_duration = (max_dd_end - max_dd_start_date).days if isinstance(max_dd_end, date) else 0

        total_net_pnl = float(df["net_pnl"].sum())
        daily_net_pnl = total_net_pnl / total_days
        total_turnover = float(df["turnover"].sum())
        total_commission = float(df["commission"].sum())
        total_trade_count = int(df["trade_count"].sum())

        total_return = (end_balance / self.capital - 1) * 100
        annual_return = total_return / total_days * self.annual_days
        daily_return = float(df["pct"].mean()) * 100
        return_std = float(df["pct"].std()) * 100

        sharpe = 0.0
        if return_std > 0:
            daily_rf = self.risk_free / np.sqrt(self.annual_days)
            sharpe = (daily_return - daily_rf) / return_std * np.sqrt(self.annual_days)

        return_drawdown_ratio = -total_net_pnl / max_drawdown if max_drawdown else 0

        stats = {
            "start_date": str(df["date"][0]),
            "end_date": str(df["date"][-1]),
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "total_return_pct": total_return,
            "annual_return_pct": annual_return,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_dd_duration_days": max_dd_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "total_turnover": total_turnover,
            "total_trade_count": total_trade_count,
            "sharpe_ratio": sharpe,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        for k, v in stats.items():
            if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
                stats[k] = 0.0

        return stats

    # ---- Charts ----

    def show_chart(self) -> None:
        """Display equity curve + drawdown."""
        if self.daily_df is None:
            self.calculate_result()
        df = self.daily_df
        if df is None:
            return

        fig = make_subplots(rows=3, cols=1, subplot_titles=["Balance", "Drawdown", "Daily Pnl"],
                            vertical_spacing=0.08)

        fig.add_trace(go.Scatter(x=df["date"], y=df["balance"], mode="lines", name="Balance"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["drawdown"], fill="tozeroy", mode="lines", name="Drawdown"), row=2, col=1)
        fig.add_trace(go.Bar(x=df["date"], y=df["net_pnl"], name="Daily Pnl"), row=3, col=1)

        fig.update_layout(height=800, showlegend=False)
        fig.show()

    def show_performance(self, benchmark_df: Optional[pl.DataFrame] = None) -> None:
        """Compare strategy vs benchmark."""
        if self.daily_df is None:
            self.calculate_result()
        df = self.daily_df
        if df is None:
            return

        # Calculate cumulative returns
        perf = df.with_columns(
            cum_ret=pl.col("balance").pct_change().cum_sum(),
        ).with_columns(
            cum_cost=(pl.col("commission") / pl.col("balance").shift(1)).cum_sum(),
        )

        if benchmark_df is not None:
            bench = benchmark_df.with_columns(
                bench_ret=pl.col("close").pct_change().cum_sum(),
            )
            perf = perf.join(bench.select(["date", "bench_ret"]), on="date", how="left")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf["date"], y=perf["cum_ret"], mode="lines", name="Strategy"))
        if benchmark_df is not None and "bench_ret" in perf.columns:
            fig.add_trace(go.Scatter(x=perf["date"], y=perf["bench_ret"], mode="lines", name="Benchmark"))
            fig.add_trace(go.Scatter(x=perf["date"],
                          y=perf["cum_ret"] - perf["bench_ret"],
                          mode="lines", name="Alpha"))
        fig.update_layout(height=500)
        fig.show()

    # ---- Utility ----

    def send_order(
        self, strategy, vt_symbol: str, direction: Direction,
        offset: Offset, price: float, volume: float
    ) -> list[str]:
        self.order_counter += 1
        order = OrderData(
            vt_orderid=f"order_{self.order_counter}",
            vt_symbol=vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=OrderStatus.SUBMITTING,
            datetime=self.current_dt,
        )
        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order
        return [order.vt_orderid]

    def cancel_order(self, strategy, vt_orderid: str) -> None:
        order = self.active_limit_orders.pop(vt_orderid, None)
        if order:
            order.status = OrderStatus.CANCELLED
            self.strategy.update_order(order)

    def get_signal(self) -> pl.DataFrame:
        """Get pre-computed signal for current day."""
        if self.signal_df is None or not hasattr(self, "current_dt"):
            return pl.DataFrame()
        dt = self.current_dt
        if isinstance(dt, datetime):
            return self.signal_df.filter(pl.col("datetime") == dt)
        return pl.DataFrame()

    def get_holding_value(self) -> float:
        value = 0.0
        for s, pos in (self.strategy.pos_data.items() if self.strategy else {}):
            bar = self.bars.get(s)
            if bar and bar.close:
                value += bar.close * pos * self.sizes.get(s, 100)
        return value

    def write_log(self, msg: str, strategy=None) -> None:
        ts = self.current_dt if hasattr(self, "current_dt") else datetime.now()
        self.logs.append(f"{ts}  {msg}")

    def get_all_trades(self) -> list:
        return list(self.trades.values())

    def get_all_daily_results(self) -> list:
        return list(self.daily_results.values())

    def print_stats(self, stats: dict) -> None:
        """Pretty-print statistics dict."""
        lines = [
            f"{'Period':20s}: {stats.get('start_date','')} ~ {stats.get('end_date','')}",
            f"{'Total days':20s}: {stats.get('total_days', 0)}  (profit {stats.get('profit_days',0)} / loss {stats.get('loss_days',0)})",
            f"{'Capital':20s}: {stats.get('capital',0):,.0f}",
            f"{'End balance':20s}: {stats.get('end_balance',0):,.0f}",
            f"",
            f"{'Total return':20s}: {stats.get('total_return_pct',0):+.2f}%",
            f"{'Annual return':20s}: {stats.get('annual_return_pct',0):+.2f}%",
            f"{'Max drawdown':20s}: {stats.get('max_ddpercent',0):+.2f}%",
            f"{'Drawdown days':20s}: {stats.get('max_dd_duration_days',0)}",
            f"",
            f"{'Total net PnL':20s}: {stats.get('total_net_pnl',0):+,.0f}",
            f"{'Total trades':20s}: {stats.get('total_trade_count',0)}",
            f"{'Total commission':20s}: {stats.get('total_commission',0):+,.0f}",
            f"",
            f"{'Sharpe ratio':20s}: {stats.get('sharpe_ratio',0):.2f}",
            f"{'Return/drawdown':20s}: {stats.get('return_drawdown_ratio',0):.2f}",
        ]
        print("\n".join(lines))
