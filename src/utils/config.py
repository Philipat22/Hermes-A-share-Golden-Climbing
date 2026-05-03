"""
统一配置加载器 — 从 .env 和 config.yaml 读取所有配置

用法:
    from src.utils.config import config
    token = config.tushare_token
    model = config.deepseek_model

优先级: 环境变量 > .env 文件 > 默认值
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 自动加载项目根目录的 .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class AppConfig:
    """应用全局配置"""

    # ── 项目路径 ──
    project_root: str = str(PROJECT_ROOT)
    data_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "data"))
    cache_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "cache"))
    quant_archive_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "quant_archive"))

    # ── 数据源 ──
    tushare_token: str = field(
        default_factory=lambda: os.getenv("TUSHARE_PRO_TOKEN", "")
    )

    # ── LLM ──
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )
    deepseek_provider: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_PROVIDER", "DeepSeek")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "deepseek-chat")
    )
    default_provider: str = field(
        default_factory=lambda: os.getenv("DEFAULT_PROVIDER", "DeepSeek")
    )

    # ── 回测 ──
    initial_cash: float = 1_000_000.0
    commission: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.001
    stop_loss: float = -0.05

    @property
    def cost_entry(self) -> float:
        return self.commission + self.slippage

    @property
    def cost_exit(self) -> float:
        return self.commission + self.stamp_tax + self.slippage

    @property
    def cost_round_trip(self) -> float:
        return self.cost_entry + self.cost_exit

    def validate(self) -> list[str]:
        """检查必需配置是否齐全，返回缺失项列表"""
        missing = []
        if not self.tushare_token:
            missing.append("TUSHARE_PRO_TOKEN")
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        return missing


# 全局单例
config = AppConfig()
