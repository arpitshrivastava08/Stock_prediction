"""
risk_manager.py — Risk Management Module
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from logger import get_logger
from config import config
from env import HOLD, BUY, SELL, ACTION_NAMES
from datetime import date, datetime

logger = get_logger(__name__)


# DATA STRUCTURES

class RiskSignal(Enum):
    OK = "OK"
    STOP_LOSS = "STOP_LOSS"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    POSITION_CAP = "POSITION_CAP"
    OVERTRADING = "OVERTRADING"


@dataclass
class PortfolioState:
    cash_balance: float
    shares_held: int
    current_price: float
    entry_price: float = 0.0
    peak_portfolio: float = 0.0
    trades_today: int = 0
    session_start_value: Optional[float] = None

    @property
    def portfolio_value(self) -> float:
        return self.cash_balance + self.shares_held * self.current_price

    @property
    def position_value(self) -> float:
        return self.shares_held * self.current_price

    @property
    def position_pct(self) -> float:
        pv = self.portfolio_value
        return self.position_value / pv if pv > 0 else 0.0


# RISK MANAGER

class RiskManager:

    def __init__(self):
        self.cfg = config.risk

        self.peak_portfolio: float = 0.0
        self.entry_price: float = 0.0
        self.trades_today: int = 0
        self.trading_halted: bool = False
        self.session_date = date.today().isoformat()
        self.intervention_log: List[dict] = []

        logger.info(
            f"RiskManager initialized | "
            f"stop_loss={self.cfg.stop_loss_pct:.0%} | "
            f"max_drawdown={self.cfg.max_drawdown_pct:.0%} | "
            f"max_position={self.cfg.max_position_pct:.0%} | "
            f"max_trades/day={self.cfg.max_trades_per_day}"
        )

    def reset(self, initial_portfolio_value: float = None):
        initial = initial_portfolio_value or config.env.initial_balance
        self.peak_portfolio = initial
        self.entry_price = 0.0
        self.trades_today = 0
        self.trading_halted = False
        self.intervention_log = []
        self.session_date = date.today().isoformat()

        logger.info(f"RiskManager reset | Initial portfolio: {initial:.2f}")

    def apply(
        self,
        recommended_action: int,
        state: PortfolioState,
    ) -> tuple:

        # Sync optional externally-tracked state when provided.
        if state.peak_portfolio > 0:
            self.peak_portfolio = max(self.peak_portfolio, state.peak_portfolio)
        if state.entry_price > 0:
            self.entry_price = state.entry_price
        if state.trades_today >= 0:
            self.trades_today = state.trades_today

        # Reset trades if new day
        today = date.today().isoformat()
        if self.session_date != today:
            self.session_date = today
            self.trades_today = 0

        # Update peak portfolio
        if state.portfolio_value > self.peak_portfolio:
            self.peak_portfolio = state.portfolio_value

        # Track entry price
        if recommended_action == BUY:
            if state.shares_held == 0:
                self.entry_price = state.current_price
            else:
                total_cost = self.entry_price * state.shares_held
                total_cost += state.current_price
                self.entry_price = total_cost / (state.shares_held + 1)

        # ── RISK CHECK 1: Trading Halted ──
        if self.trading_halted:
            return self._veto(
                recommended_action,
                HOLD,
                RiskSignal.MAX_DRAWDOWN,
                "Trading halted due to drawdown."
            )

        # ── RISK CHECK 2: Drawdown ──
        drawdown = (
            (self.peak_portfolio - state.portfolio_value) / self.peak_portfolio
            if self.peak_portfolio > 0 else 0.0
        )

        if drawdown >= self.cfg.max_drawdown_pct:
            self.trading_halted = True
            return self._veto(
                recommended_action,
                HOLD,
                RiskSignal.MAX_DRAWDOWN,
                f"Drawdown {drawdown:.1%} exceeded limit."
            )

        # ── RISK CHECK 3: Stop Loss ──
        if (
            recommended_action != SELL
            and state.shares_held > 0
            and self.entry_price > 0
        ):
            loss_pct = (self.entry_price - state.current_price) / self.entry_price

            if loss_pct >= self.cfg.stop_loss_pct:
                return self._veto(
                    recommended_action,
                    SELL,
                    RiskSignal.STOP_LOSS,
                    f"Stop loss triggered: {loss_pct:.1%}"
                )

        # ── RISK CHECK 4: Position Limit ──
        if recommended_action == BUY:
            projected_value = (state.shares_held + 1) * state.current_price
            pct = projected_value / max(state.portfolio_value, 1.0)

            if pct > self.cfg.max_position_pct:
                return self._veto(
                    recommended_action,
                    HOLD,
                    RiskSignal.POSITION_CAP,
                    f"Position limit exceeded: {pct:.1%}"
                )

        # ── RISK CHECK 5: Overtrading ──
        if recommended_action in (BUY, SELL):
            if self.trades_today >= self.cfg.max_trades_per_day:
                return self._veto(
                    recommended_action,
                    HOLD,
                    RiskSignal.OVERTRADING,
                    "Too many trades today"
                )
            else:
                self.trades_today += 1

        return recommended_action, RiskSignal.OK, "OK"

    def get_summary(self) -> dict:
        return {
            "trading_halted": self.trading_halted,
            "trades_today": self.trades_today,
            "peak_portfolio": round(self.peak_portfolio, 2),
            "interventions_count": len(self.intervention_log),
            "intervention_types": sorted({i["signal"] for i in self.intervention_log}),
        }

    def _veto(
        self,
        original_action: int,
        override_action: int,
        signal: RiskSignal,
        reason: str,
    ) -> tuple:

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "original": ACTION_NAMES.get(original_action, str(original_action)),
            "override": ACTION_NAMES.get(override_action, str(override_action)),
            "signal": signal.value,
            "reason": reason,
        }

        self.intervention_log.append(entry)

        logger.warning(
            f"RISK OVERRIDE: {entry['original']} → {entry['override']} | {reason}"
        )

        return override_action, signal, reason