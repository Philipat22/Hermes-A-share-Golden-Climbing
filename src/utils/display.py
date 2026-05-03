from colorama import Fore, Style
from tabulate import tabulate
from .analysts import ANALYST_ORDER
import os
import json


def sort_agent_signals(signals):
    """Sort agent signals in a consistent order."""
    # Create order mapping from ANALYST_ORDER
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order["Risk Management"] = len(ANALYST_ORDER)  # Add Risk Management at the end

    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))


def print_trading_output(result: dict) -> None:
    """
    Print formatted trading results with colored tables for multiple tickers.

    Args:
        result (dict): Dictionary containing decisions and analyst signals for multiple tickers
    """
    decisions = result.get("decisions")
    if not decisions:
        print(f"{Fore.RED}No trading decisions available{Style.RESET_ALL}")
        return

    # Print decisions for each ticker
    for ticker, decision in decisions.items():
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Analysis for {Fore.CYAN}{ticker}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Style.BRIGHT}{'=' * 50}{Style.RESET_ALL}")

        # Prepare analyst signals table for this ticker
        table_data = []
        for agent, signals in result.get("analyst_signals", {}).items():
            if ticker not in signals:
                continue
                
            # Skip Risk Management agent in the signals section
            if agent == "risk_management_agent":
                continue

            signal = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            signal_type = signal.get("signal", "").upper()
            confidence = signal.get("confidence", 0)

            signal_color = {
                "BULLISH": Fore.GREEN,
                "BEARISH": Fore.RED,
                "NEUTRAL": Fore.YELLOW,
            }.get(signal_type, Fore.WHITE)
            
            # Get reasoning if available
            reasoning_str = ""
            if "reasoning" in signal and signal["reasoning"]:
                reasoning = signal["reasoning"]
                
                # Handle different types of reasoning (string, dict, etc.)
                if isinstance(reasoning, str):
                    reasoning_str = reasoning
                elif isinstance(reasoning, dict):
                    # Convert dict to string representation
                    reasoning_str = json.dumps(reasoning, indent=2)
                else:
                    # Convert any other type to string
                    reasoning_str = str(reasoning)
                
                # Wrap long reasoning text to make it more readable
                wrapped_reasoning = ""
                current_line = ""
                # Use a fixed width of 60 characters to match the table column width
                max_line_length = 60
                for word in reasoning_str.split():
                    if len(current_line) + len(word) + 1 > max_line_length:
                        wrapped_reasoning += current_line + "\n"
                        current_line = word
                    else:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                if current_line:
                    wrapped_reasoning += current_line
                
                reasoning_str = wrapped_reasoning

            table_data.append(
                [
                    f"{Fore.CYAN}{agent_name}{Style.RESET_ALL}",
                    f"{signal_color}{signal_type}{Style.RESET_ALL}",
                    f"{Fore.WHITE}{confidence}%{Style.RESET_ALL}",
                    f"{Fore.WHITE}{reasoning_str}{Style.RESET_ALL}",
                ]
            )

        # Sort the signals according to the predefined order
        table_data = sort_agent_signals(table_data)

        print(f"\n{Fore.WHITE}{Style.BRIGHT}AGENT ANALYSIS:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(
            tabulate(
                table_data,
                headers=[f"{Fore.WHITE}Agent", "Signal", "Confidence", "Reasoning"],
                tablefmt="grid",
                colalign=("left", "center", "right", "left"),
            )
        )

        # Print Trading Decision Table
        action = decision.get("action", "").upper()
        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED,
        }.get(action, Fore.WHITE)

        # Get reasoning and format it
        reasoning = decision.get("reasoning", "")
        # Wrap long reasoning text to make it more readable
        wrapped_reasoning = ""
        if reasoning:
            current_line = ""
            # Use a fixed width of 60 characters to match the table column width
            max_line_length = 60
            for word in reasoning.split():
                if len(current_line) + len(word) + 1 > max_line_length:
                    wrapped_reasoning += current_line + "\n"
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
            if current_line:
                wrapped_reasoning += current_line

        decision_data = [
            ["Action", f"{action_color}{action}{Style.RESET_ALL}"],
            ["Quantity", f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}"],
            [
                "Confidence",
                f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}",
            ],
            ["Reasoning", f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}"],
        ]
        
        print(f"\n{Fore.WHITE}{Style.BRIGHT}TRADING DECISION:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(tabulate(decision_data, tablefmt="grid", colalign=("left", "left")))

    # Print Portfolio Summary
    print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")
    portfolio_data = []
    
    # Extract portfolio manager reasoning (common for all tickers)
    portfolio_manager_reasoning = None
    for ticker, decision in decisions.items():
        if decision.get("reasoning"):
            portfolio_manager_reasoning = decision.get("reasoning")
            break
            
    analyst_signals = result.get("analyst_signals", {})
    for ticker, decision in decisions.items():
        action = decision.get("action", "").upper()
        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED,
        }.get(action, Fore.WHITE)

        # Calculate analyst signal counts
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        if analyst_signals:
            for agent, signals in analyst_signals.items():
                if ticker in signals:
                    signal = signals[ticker].get("signal", "").upper()
                    if signal == "BULLISH":
                        bullish_count += 1
                    elif signal == "BEARISH":
                        bearish_count += 1
                    elif signal == "NEUTRAL":
                        neutral_count += 1

        portfolio_data.append(
            [
                f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
                f"{action_color}{action}{Style.RESET_ALL}",
                f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}",
                f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}",
                f"{Fore.GREEN}{bullish_count}{Style.RESET_ALL}",
                f"{Fore.RED}{bearish_count}{Style.RESET_ALL}",
                f"{Fore.YELLOW}{neutral_count}{Style.RESET_ALL}",
            ]
        )

    headers = [
        f"{Fore.WHITE}Ticker",
        f"{Fore.WHITE}Action",
        f"{Fore.WHITE}Quantity",
        f"{Fore.WHITE}Confidence",
        f"{Fore.WHITE}Bullish",
        f"{Fore.WHITE}Bearish",
        f"{Fore.WHITE}Neutral",
    ]
    
    # Print the portfolio summary table
    print(
        tabulate(
            portfolio_data,
            headers=headers,
            tablefmt="grid",
            colalign=("left", "center", "right", "right", "center", "center", "center"),
        )
    )
    
    # Print Portfolio Manager's reasoning if available
    if portfolio_manager_reasoning:
        # Handle different types of reasoning (string, dict, etc.)
        reasoning_str = ""
        if isinstance(portfolio_manager_reasoning, str):
            reasoning_str = portfolio_manager_reasoning
        elif isinstance(portfolio_manager_reasoning, dict):
            # Convert dict to string representation
            reasoning_str = json.dumps(portfolio_manager_reasoning, indent=2)
        else:
            # Convert any other type to string
            reasoning_str = str(portfolio_manager_reasoning)
            
        # Wrap long reasoning text to make it more readable
        wrapped_reasoning = ""
        current_line = ""
        # Use a fixed width of 60 characters to match the table column width
        max_line_length = 60
        for word in reasoning_str.split():
            if len(current_line) + len(word) + 1 > max_line_length:
                wrapped_reasoning += current_line + "\n"
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        if current_line:
            wrapped_reasoning += current_line
            
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Portfolio Strategy:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{wrapped_reasoning}{Style.RESET_ALL}")


def print_backtest_results(table_rows: list) -> None:
    """Print the backtest results in a nicely formatted table"""
    # Clear the screen
    os.system("cls" if os.name == "nt" else "clear")

    # Split rows into ticker rows and summary rows
    ticker_rows = []
    summary_rows = []

    for row in table_rows:
        if isinstance(row[1], str) and "PORTFOLIO SUMMARY" in row[1]:
            summary_rows.append(row)
        else:
            ticker_rows.append(row)

    # Display latest portfolio summary
    if summary_rows:
        # Pick the most recent summary by date (YYYY-MM-DD)
        latest_summary = max(summary_rows, key=lambda r: r[0])
        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")

        # Adjusted indexes after adding Long/Short Shares
        position_str = latest_summary[7].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        cash_str     = latest_summary[8].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        total_str    = latest_summary[9].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")

        print(f"Cash Balance: {Fore.CYAN}${float(cash_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Position Value: {Fore.YELLOW}${float(position_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Value: {Fore.WHITE}${float(total_str):,.2f}{Style.RESET_ALL}")
        print(f"Portfolio Return: {latest_summary[10]}")
        if len(latest_summary) > 14 and latest_summary[14]:
            print(f"Benchmark Return: {latest_summary[14]}")

        # Display performance metrics if available
        if latest_summary[11]:  # Sharpe ratio
            print(f"Sharpe Ratio: {latest_summary[11]}")
        if latest_summary[12]:  # Sortino ratio
            print(f"Sortino Ratio: {latest_summary[12]}")
        if latest_summary[13]:  # Max drawdown
            print(f"Max Drawdown: {latest_summary[13]}")

    # Add vertical spacing
    print("\n" * 2)

    # Print the table with just ticker rows
    print(
        tabulate(
            ticker_rows,
            headers=[
                "Date",
                "Ticker",
                "Action",
                "Quantity",
                "Price",
                "Long Shares",
                "Short Shares",
                "Position Value",
            ],
            tablefmt="grid",
            colalign=(
                "left",    # Date
                "left",    # Ticker
                "center",  # Action
                "right",   # Quantity
                "right",   # Price
                "right",   # Long Shares
                "right",   # Short Shares
                "right",   # Position Value
            ),
        )
    )

    # Add vertical spacing
    print("\n" * 4)


def format_backtest_row(
    date: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    long_shares: float = 0,
    short_shares: float = 0,
    position_value: float = 0,
    is_summary: bool = False,
    total_value: float = None,
    return_pct: float = None,
    cash_balance: float = None,
    total_position_value: float = None,
    sharpe_ratio: float = None,
    sortino_ratio: float = None,
    max_drawdown: float = None,
    benchmark_return_pct: float | None = None,
) -> list[any]:
    """Format a row for the backtest results table"""
    # Color the action
    action_color = {
        "BUY": Fore.GREEN,
        "COVER": Fore.GREEN,
        "SELL": Fore.RED,
        "SHORT": Fore.RED,
        "HOLD": Fore.WHITE,
    }.get(action.upper(), Fore.WHITE)

    if is_summary:
        return_color = Fore.GREEN if return_pct >= 0 else Fore.RED
        benchmark_str = ""
        if benchmark_return_pct is not None:
            bench_color = Fore.GREEN if benchmark_return_pct >= 0 else Fore.RED
            benchmark_str = f"{bench_color}{benchmark_return_pct:+.2f}%{Style.RESET_ALL}"
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY{Style.RESET_ALL}",
            "",  # Action
            "",  # Quantity
            "",  # Price
            "",  # Long Shares
            "",  # Short Shares
            f"{Fore.YELLOW}${total_position_value:,.2f}{Style.RESET_ALL}",  # Total Position Value
            f"{Fore.CYAN}${cash_balance:,.2f}{Style.RESET_ALL}",  # Cash Balance
            f"{Fore.WHITE}${total_value:,.2f}{Style.RESET_ALL}",  # Total Value
            f"{return_color}{return_pct:+.2f}%{Style.RESET_ALL}",  # Return
            f"{Fore.YELLOW}{sharpe_ratio:.2f}{Style.RESET_ALL}" if sharpe_ratio is not None else "",  # Sharpe Ratio
            f"{Fore.YELLOW}{sortino_ratio:.2f}{Style.RESET_ALL}" if sortino_ratio is not None else "",  # Sortino Ratio
            f"{Fore.RED}{max_drawdown:.2f}%{Style.RESET_ALL}" if max_drawdown is not None else "",  # Max Drawdown (signed)
            benchmark_str,  # Benchmark (S&P 500)
        ]
    else:
        return [
            date,
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action.upper()}{Style.RESET_ALL}",
            f"{action_color}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{price:,.2f}{Style.RESET_ALL}",
            f"{Fore.GREEN}{long_shares:,.0f}{Style.RESET_ALL}",   # Long Shares
            f"{Fore.RED}{short_shares:,.0f}{Style.RESET_ALL}",    # Short Shares
            f"{Fore.YELLOW}{position_value:,.2f}{Style.RESET_ALL}",
        ]

# ────────────────────────────────────────────────
# P3: Agent对比信号卡
# ────────────────────────────────────────────────

def generate_signal_card(ticker, analyst_signals, stock_info=None, include_reasoning=False):
    """生成一张格式化文本卡，展示所有Agent对该股票的信号对比。"""
    from src.utils.analysts import ANALYST_ORDER, ANALYST_CONFIG

    lines = []
    width = 66

    all_signals = {}
    for agent_key, signals in analyst_signals.items():
        if isinstance(signals, dict) and ticker in signals:
            sig = signals[ticker]
            if isinstance(sig, dict):
                all_signals[agent_key] = sig

    name = ''
    if stock_info:
        name = getattr(stock_info, 'name', '') or ''
    elif ticker:
        try:
            from src.tools.a_stock_api import get_stock_info as gsi
            info = gsi(ticker)
            name = info.name if info else ''
        except Exception:
            pass

    title = f' {name} {ticker} ' if name else f' {ticker} '

    lines.append('\u250c' + '\u2500' * width + '\u2510')
    lines.append('\u2502' + Fore.WHITE + Style.BRIGHT + title.center(width) + Style.RESET_ALL + '\u2502')
    lines.append('\u2502' + '\u2500' * width + '\u2502')
    lines.append(f'\u2502{"Agent":30s} \u2502 {"Signal":10s} \u2502 {"Conf":6s}\u2502')
    lines.append('\u2502' + '\u2500' * width + '\u2502')

    llm_agents = []
    calc_agents = []
    for display_name, agent_key in ANALYST_ORDER:
        found_key = None
        for k in all_signals:
            if k == agent_key or k == agent_key + '_agent' or agent_key == k + '_agent':
                found_key = k
                break
        if found_key:
            sig = all_signals[found_key]
            entry = (display_name, found_key, sig)
            llm_names = [
                'Aswath Damodaran','Ben Graham','Bill Ackman','Cathie Wood',
                'Charlie Munger','Michael Burry','Mohnish Pabrai','Nassim Taleb',
                'Peter Lynch','Phil Fisher','Rakesh Jhunjhunwala','Stanley Druckenmiller',
                'Warren Buffett',
            ]
            if display_name in llm_names:
                llm_agents.append(entry)
            else:
                calc_agents.append(entry)

    leftover = []
    for k, v in all_signals.items():
        used = [ek for _, ek, _ in llm_agents + calc_agents]
        if k not in used and k != 'risk_management_agent':
            leftover.append((k, k, v))
    for item in leftover:
        calc_agents.append(item)

    def _render_row(dn, k, sig, is_llm):
        s = sig.get('signal', 'neutral').upper()
        c = sig.get('confidence', 0) or 0
        emoji = {'BULLISH': '\U0001f7e2', 'BEARISH': '\U0001f534', 'NEUTRAL': '\U0001f7e1'}.get(s, '\u26aa')
        color = {'BULLISH': Fore.GREEN, 'BEARISH': Fore.RED, 'NEUTRAL': Fore.YELLOW}.get(s, Fore.WHITE)
        prefix = '\U0001f9e0 ' if is_llm else '\U0001f4ca '
        label = f'{prefix}{dn}'
        if len(label) > 28:
            label = label[:27] + '\u2026'
        conf_str = f'{c:3.0f}%' if isinstance(c, (int, float)) else str(c)
        lines.append(f'\u2502{label:30s} \u2502 {color}{emoji} {s:5s}{Style.RESET_ALL} \u2502 {conf_str:6s}\u2502')

    for dn, k, sig in llm_agents:
        _render_row(dn, k, sig, is_llm=True)
    for dn, k, sig in calc_agents:
        _render_row(dn, k, sig, is_llm=False)

    lines.append('\u2502' + '\u2500' * width + '\u2502')
    bullish = sum(1 for _, _, s in llm_agents + calc_agents if s.get("signal") == "bullish")
    bearish = sum(1 for _, _, s in llm_agents + calc_agents if s.get("signal") == "bearish")
    neutral = sum(1 for _, _, s in llm_agents + calc_agents if s.get("signal") == "neutral")
    tag = f'\U0001f7e2 {bullish}  \U0001f534 {bearish}  \U0001f7e1 {neutral}'
    lines.append(f'\u2502{"信号分布":30s} \u2502 {tag:^28s}\u2502')
    lines.append('\u2502' + '\u2500' * width + '\u2502')

    total_score = 0.0
    total_weight = 0.0
    for _, k, sig in llm_agents + calc_agents:
        s = sig.get('signal', 'neutral')
        c = sig.get('confidence', 0) or 0
        weight = 2.0
        if s == 'bullish':
            total_score += c * weight
        elif s == 'bearish':
            total_score -= c * weight
        total_weight += weight
    avg_score = total_score / total_weight if total_weight else 0.0
    score_emoji = '\U0001f7e2' if avg_score > 15 else ('\U0001f534' if avg_score < -15 else '\U0001f7e1')
    lines.append(f'\u2502{"综合得分":30s} \u2502 {score_emoji} {avg_score:+5.1f} / 100{"":11s}\u2502')
    lines.append('\u2514' + '\u2500' * width + '\u2518')

    return '\n'.join(lines)


def print_signal_card(ticker, analyst_signals, stock_info=None, include_reasoning=False):
    """打印Agent对比信号卡到终端。"""
    card = generate_signal_card(ticker, analyst_signals, stock_info, include_reasoning)
    print(card)


# ────────────────────────────────────────────────
# P3: 交易指令卡（实战交易参数）
# ────────────────────────────────────────────────

def generate_trading_card(ticker: str, analyst_signals: dict) -> str:
    """
    生成实战交易指令卡，包含：
    - 当前股价 vs 内在价值估值
    - 买入区间建议
    - 目标位
    - 止损位
    - 仓位建议
    """
    from src.utils.analysts import ANALYST_CONFIG

    lines = []
    width = 66
    tickers_list = [ticker]

    # ── 获取当前股价 ──
    price = None
    try:
        from src.tools.a_stock_api import get_prices
        df = get_prices(tickers_list, "2026-04-29", "2026-04-29")
        if df is not None and not df.empty and ticker in df.columns:
            vals = df[ticker].dropna()
            if not vals.empty:
                price = float(vals.iloc[-1])
    except Exception:
        pass

    # ── 收集所有Agent信号 ──
    all_signals = {}
    for agent_key, signals in analyst_signals.items():
        if isinstance(signals, dict) and ticker in signals:
            sig = signals[ticker]
            if isinstance(sig, dict):
                all_signals[agent_key] = sig

    # ── 尝试从推理文本中提取内在价值和目标价 ──
    import re
    intrinsic_values = []
    target_prices = []
    margin_of_safety_vals = []

    # 估值型Agent：尝试从reasoning提取内在价值
    val_agent_keys = [
        "warren_buffett_agent", "aswath_damodaran_agent",
        "ben_graham_agent", "valuation_analyst_agent",
        "mohnish_pabrai_agent", "michael_burry_agent",
    ]
    for k in all_signals:
        raw = all_signals[k].get("reasoning", "") or ""
        # reasoning可能是dict（旧的格式兼容）, 也可能是str
        reason = str(raw) if not isinstance(raw, str) else raw
        sig = all_signals[k].get("signal", "")
        conf = all_signals[k].get("confidence", 0) or 0

        # 提取数值：匹配 X元/股、intrinsic value ¥XXX、PE XXX 等
        nums = re.findall(r'(?:intrinsic|内在|价值|估值|fair|合理)[^。]*?[¥￥]?\s*(\d+(?:\.\d+)?)(?:元|\s|$)', reason, re.IGNORECASE)
        if nums:
            intrinsic_values.extend([float(n) for n in nums])

        # 提取目标价：target price ¥XX
        targets = re.findall(r'(?:target|目标|TP)[^。]*?[¥￥]?\s*(\d+(?:\.\d+)?)(?:元|\s|$)', reason, re.IGNORECASE)
        if targets:
            target_prices.extend([float(n) for n in targets])

    # ── 提取风险/波动率数据 ──
    volatility = None
    max_position = None
    for k, sig in all_signals.items():
        if "risk" in k.lower() or "management" in k.lower():
            vol_raw = sig.get("reasoning", "") or ""
            vol_text = str(vol_raw) if not isinstance(vol_raw, str) else vol_raw
            vol_nums = re.findall(r'(?:波动率|volatility|vol|波动)[^，。]*?(\d+(?:\.\d+)?)%', vol_text, re.IGNORECASE)
            if vol_nums:
                volatility = float(vol_nums[0])
            pos_nums = re.findall(r'(?:仓位|position|limit|上限)[^，。]*?[¥￥]?\s*(\d+(?:\.\d+)?)', vol_text, re.IGNORECASE)
            if pos_nums:
                max_position = float(pos_nums[0])

    # ── 计算统计 ──
    total = len(all_signals)
    bullish = sum(1 for s in all_signals.values() if s.get("signal") == "bullish")
    bearish = sum(1 for s in all_signals.values() if s.get("signal") == "bearish")
    neutral = total - bullish - bearish

    # LLM大师统计
    llm_names = {
        'aswath_damodaran_agent','ben_graham_agent','bill_ackman_agent','cathie_wood_agent',
        'charlie_munger_agent','michael_burry_agent','mohnish_pabrai_agent','nassim_taleb_agent',
        'peter_lynch_agent','phil_fisher_agent','rakesh_jhunjhunwala_agent','stanley_druckenmiller_agent',
        'warren_buffett_agent','news_sentiment_agent',
    }
    llm_signals = {k: v for k, v in all_signals.items() if k in llm_names}
    calc_signals = {k: v for k, v in all_signals.items() if k not in llm_names and 'risk' not in k.lower()}
    llm_bullish = sum(1 for s in llm_signals.values() if s.get("signal") == "bullish")
    llm_bearish = sum(1 for s in llm_signals.values() if s.get("signal") == "bearish")

    # ── 计算综合买入区间 ──
    buy_zone_low = None
    buy_zone_high = None
    if price and intrinsic_values:
        # 取内在价值的中位数+平均值做区间
        median_iv = sorted(intrinsic_values)[len(intrinsic_values)//2]
        mean_iv = sum(intrinsic_values) / len(intrinsic_values)
        # 安全边际：低于内在价值20-30%为买入区
        discount = 0.25
        buy_zone_high = median_iv * (1 - discount * 0.5)  # 合理价格上沿
        buy_zone_low = median_iv * (1 - discount)          # 深度价值下沿
    elif price:
        # 没有内在价值数据，用价格区间法
        buy_zone_low = price * 0.85
        buy_zone_high = price * 1.0 if bullish > bearish else price * 0.95

    # ── 计算目标位 ──
    target_price = None
    if target_prices:
        target_price = sum(target_prices) / len(target_prices)
    elif buy_zone_high:
        target_price = buy_zone_high * 1.2  # 合理价格上浮20%
    elif price and bullish > bearish:
        target_price = price * 1.15

    # ── 计算止损位 ──
    stop_loss = None
    if price and volatility:
        # 2倍波动率止损
        stop_loss = price * (1 - min(volatility / 100 * 2, 0.15))
    elif price:
        stop_loss = price * 0.92  # 默认止损8%

    # ── 计算仓位建议 ──
    position_pct = 50  # 默认半仓
    if total > 0:
        net_score = (bullish - bearish) / total
        if net_score > 0.4:
            position_pct = 90
        elif net_score > 0.2:
            position_pct = 70
        elif net_score > 0:
            position_pct = 50
        elif net_score > -0.2:
            position_pct = 25
        else:
            position_pct = 10

    # ── 生成卡片 ──
    title = f' 📋 交易指令卡 {ticker} '
    lines.append('\u250c' + '\u2500' * width + '\u2510')
    lines.append('\u2502' + Fore.WHITE + Style.BRIGHT + title.center(width) + Style.RESET_ALL + '\u2502')

    # 当前价格
    price_str = f'¥{price:.2f}' if price else 'N/A'
    lines.append(f'  💰 当前股价:  {Fore.CYAN}{price_str}{Style.RESET_ALL}')

    # 综合信号
    signal_color = Fore.GREEN if bullish > bearish else (Fore.RED if bearish > bullish else Fore.YELLOW)
    signal_text = '看多' if bullish > bearish else ('看空' if bearish > bullish else '中性')
    lines.append(f'  🔮 综合信号:  {signal_color}{signal_text}{Style.RESET_ALL}  (LLM大师 {llm_bullish}多/{llm_bearish}空 | 计算型 {len(calc_signals)}位)')

    lines.append('')
    lines.append(f'  {"─" * 60}')
    lines.append(f'  {Style.BRIGHT}交易参数{Style.RESET_ALL}')
    lines.append(f'  {"─" * 60}')

    # 买入区间
    if buy_zone_low and buy_zone_high:
        i_str = f'(目前{(price / buy_zone_high - 1) * 100:+.0f}% vs 上沿)' if price else ''
        lines.append(f'  📈 买入区间:  {Fore.GREEN}¥{buy_zone_low:.2f} ~ ¥{buy_zone_high:.2f}{Style.RESET_ALL}  {i_str}')
    else:
        lines.append(f'  📈 买入区间:  {Fore.YELLOW}数据不足，参考大师信号{Style.RESET_ALL}')

    # 目标位
    if target_price:
        upside = (target_price / price - 1) * 100 if price else 0
        color_t = Fore.GREEN if upside > 5 else (Fore.YELLOW if upside > 0 else Fore.RED)
        ups = f'+{upside:.1f}%' if upside > 0 else f'{upside:.1f}%'
        lines.append(f'  🎯 目标位:    {color_t}¥{target_price:.2f} ({ups}){Style.RESET_ALL}')

    # 止损位
    if stop_loss:
        downside = (stop_loss / price - 1) * 100 if price else 0
        lines.append(f'  ⛔ 止损位:    {Fore.RED}¥{stop_loss:.2f} ({downside:+.1f}%){Style.RESET_ALL}')

    # 盈亏比
    if target_price and stop_loss and price:
        risk = (price - stop_loss) / price
        reward = (target_price - price) / price
        if risk > 0:
            ror = reward / risk
            ror_color = Fore.GREEN if ror > 2 else (Fore.YELLOW if ror > 1 else Fore.RED)
            lines.append(f'  ⚖️ 盈亏比:    {ror_color}1:{ror:.1f}{Style.RESET_ALL} (上{reward*100:.0f}% / 下{risk*100:.0f}%)')

    lines.append('')
    lines.append(f'  {"─" * 60}')
    lines.append(f'  {Style.BRIGHT}仓位与风控{Style.RESET_ALL}')
    lines.append(f'  {"─" * 60}')

    # 仓位建议
    pos_color = Fore.GREEN if position_pct >= 70 else (Fore.YELLOW if position_pct >= 40 else Fore.RED)
    pos_bar = '█' * (position_pct // 10) + '░' * (10 - position_pct // 10)
    lines.append(f'  💡 建议仓位:  {pos_color}{position_pct}%{Style.RESET_ALL}  [{pos_bar}]')

    if volatility:
        vol_color = Fore.GREEN if volatility < 25 else (Fore.YELLOW if volatility < 40 else Fore.RED)
        lines.append(f'  🌊 波动率:    {vol_color}{volatility:.1f}%{Style.RESET_ALL}')

    if max_position:
        lines.append(f'  🔑 仓位上限:  ¥{max_position:,.0f}')

    # 风险等级
    if volatility:
        risk_level = '低' if volatility < 20 else ('中' if volatility < 35 else '高')
        risk_color = Fore.GREEN if risk_level == '低' else (Fore.YELLOW if risk_level == '中' else Fore.RED)
        lines.append(f'  ⚡ 风险等级:  {risk_color}{risk_level}{Style.RESET_ALL}')

    lines.append('')
    lines.append(f'  {"─" * 60}')
    lines.append(f'  {Style.BRIGHT}大师共识{Style.RESET_ALL}')
    lines.append(f'  {"─" * 60}')
    lines.append(f'  LLM大师:  {Fore.GREEN}{llm_bullish}多{Style.RESET_ALL} / {Fore.RED}{llm_bearish}空{Style.RESET_ALL} / 中{len(llm_signals)-llm_bullish-llm_bearish}  |  {len(llm_signals)}位参与')
    lines.append(f'  计算型:   {Fore.GREEN}{bullish - llm_bullish}多{Style.RESET_ALL} / {Fore.RED}{bearish - llm_bearish}空{Style.RESET_ALL} / 中{neutral}  |  {len(calc_signals)}位参与')
    lines.append(f'  合计:     {Fore.GREEN}{bullish}多{Style.RESET_ALL} / {Fore.RED}{bearish}空{Style.RESET_ALL} / 中{neutral}  |  {total}位Agent')

    lines.append('\u2514' + '\u2500' * width + '\u2518')

    return '\n'.join(lines)


def print_trading_card(ticker: str, analyst_signals: dict):
    """打印交易指令卡到终端。"""
    card = generate_trading_card(ticker, analyst_signals)
    print(card)
