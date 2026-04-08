
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import json
import os

from logger import get_logger
from config import config
from env import StockTradingEnv, HOLD, BUY, SELL

logger = get_logger(__name__)


class Backtester:
  

    def __init__(self, env: StockTradingEnv):
        self.env = env

    def run(
        self,
        agent,
        label: str = "Agent",
        deterministic: bool = True,
    ) -> Dict[str, Any]:
       
        obs, _ = self.env.reset()
        done = False
        portfolio_values = [self.env.initial_balance]
        actions_taken = []

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            done = terminated or truncated
            portfolio_values.append(info["portfolio_value"])
            actions_taken.append(int(action))

        metrics = self._compute_metrics(portfolio_values, actions_taken)
        metrics["label"] = label
        metrics["portfolio_history"] = portfolio_values

        logger.info(
            f"Backtest [{label}] | "
            f"Return: {metrics['total_return']:.2%} | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"MaxDD: {metrics['max_drawdown']:.2%}"
        )
        return metrics

    def run_buy_and_hold(self) -> Dict[str, Any]:
        """
        Buy & Hold baseline: buy on day 1, hold until last day, then sell.
        This is the standard benchmark for any trading strategy.
        """
        df = self.env.df
        initial = self.env.initial_balance
        start_price = float(df["Close"].iloc[0])

        shares = initial / start_price
        portfolio_values = [
            initial + shares * (float(df["Close"].iloc[i]) - start_price)
            for i in range(len(df))
        ]

        metrics = self._compute_metrics(
            portfolio_values,
            [BUY] + [HOLD] * (len(df) - 2) + [SELL]
        )
        metrics["label"] = "Buy & Hold"
        metrics["portfolio_history"] = portfolio_values

        logger.info(
            f"Backtest [Buy&Hold] | "
            f"Return: {metrics['total_return']:.2%} | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}"
        )
        return metrics

    def compare(
        self,
        agents: Dict[str, Any],
        include_buy_hold: bool = True,
    ) -> pd.DataFrame:
        """
        Run and compare multiple agents.

        Args:
            agents: Dict of name → agent
            include_buy_hold: Add Buy & Hold baseline to comparison

        Returns:
            DataFrame with one row per strategy
        """
        results = []

        if include_buy_hold:
            bh = self.run_buy_and_hold()
            results.append({
                "Strategy": bh["label"],
                "Total Return": f"{bh['total_return']:.2%}",
                "Sharpe Ratio": f"{bh['sharpe_ratio']:.3f}",
                "Max Drawdown": f"{bh['max_drawdown']:.2%}",
                "Win Rate": f"{bh['win_rate']:.2%}",
                "Trades": bh["trade_count"],
                "Final Value": f"{bh['final_portfolio']:.2f}",
                "_total_return": bh["total_return"],
                "_sharpe": bh["sharpe_ratio"],
            })

        for name, agent in agents.items():
            r = self.run(agent, label=name)
            results.append({
                "Strategy": name,
                "Total Return": f"{r['total_return']:.2%}",
                "Sharpe Ratio": f"{r['sharpe_ratio']:.3f}",
                "Max Drawdown": f"{r['max_drawdown']:.2%}",
                "Win Rate": f"{r['win_rate']:.2%}",
                "Trades": r["trade_count"],
                "Final Value": f"{r['final_portfolio']:.2f}",
                "_total_return": r["total_return"],
                "_sharpe": r["sharpe_ratio"],
            })

        df = pd.DataFrame(results)
        df = df.sort_values("_sharpe", ascending=False).drop(
            columns=["_total_return", "_sharpe"]
        )
        return df

    def compare_and_export(
        self,
        agents: Dict[str, Any],
        save_path: str = None,
        include_buy_hold: bool = True,
    ) -> Dict[str, Any]:
        """
        Run comparison and export results to JSON.
        Used by the retrainer to persist backtest results after each retrain.

        Args:
            agents: Dict of name → agent
            save_path: JSON file path (defaults to models/backtest_results.json)
            include_buy_hold: Add Buy & Hold baseline

        Returns:
            Dict with raw metrics per strategy
        """
        save_path = save_path or os.path.join(
            config.train.models_dir, "backtest_results.json"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        results = {}

        if include_buy_hold:
            bh = self.run_buy_and_hold()
            results["buy_hold"] = {
                "total_return": bh.get("total_return", 0),
                "sharpe_ratio": bh.get("sharpe_ratio", 0),
                "max_drawdown": bh.get("max_drawdown", 0),
                "win_rate": bh.get("win_rate", 0),
                "calmar_ratio": bh.get("calmar_ratio", 0),
            }

        for name, agent in agents.items():
            r = self.run(agent, label=name)
            results[name] = {
                "total_return": r.get("total_return", 0),
                "sharpe_ratio": r.get("sharpe_ratio", 0),
                "max_drawdown": r.get("max_drawdown", 0),
                "win_rate": r.get("win_rate", 0),
                "calmar_ratio": r.get("calmar_ratio", 0),
                "trade_count": r.get("trade_count", 0),
                "final_portfolio": r.get("final_portfolio", 0),
            }

        # Add metadata
        from datetime import datetime
        export = {
            "updated_at": datetime.now().isoformat(),
            "strategies": results,
        }

        with open(save_path, "w") as f:
            json.dump(export, f, indent=2)

        logger.info(f"Backtest results exported to {save_path}")
        return results

    @staticmethod
    def _compute_metrics(
        portfolio_values: List[float],
        actions: List[int],
    ) -> Dict[str, Any]:
        """
        Compute all performance metrics from portfolio value time series.
        """
        pv = np.array(portfolio_values, dtype=float)
        if len(pv) < 2:
            return {}

        # Returns
        daily_returns = np.diff(pv) / (pv[:-1] + 1e-9)

        # Total Return
        total_return = (pv[-1] - pv[0]) / pv[0]

        # Sharpe Ratio (annualized, 252 trading days)
        rf_daily = (1 + config.backtest.risk_free_rate) ** (1 / 252) - 1
        excess_returns = daily_returns - rf_daily
        sharpe = (
            (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)
        )

        # Sortino Ratio (only downside deviation)
        downside = excess_returns[excess_returns < 0]
        sortino = (
            (excess_returns.mean() / (downside.std() + 1e-9)) * np.sqrt(252)
            if len(downside) > 0 else 0.0
        )

        # Max Drawdown
        peak = np.maximum.accumulate(pv)
        drawdowns = (pv - peak) / (peak + 1e-9)
        max_drawdown = float(drawdowns.min())

        # Calmar Ratio
        calmar = abs(total_return / (max_drawdown + 1e-9))

        # Win Rate (profitable days)
        wins = np.sum(daily_returns > 0)
        win_rate = wins / len(daily_returns)

        # Trade stats
        buy_indices  = [i for i, a in enumerate(actions) if a == BUY]
        sell_indices = [i for i, a in enumerate(actions) if a == SELL]
        trade_count  = len(buy_indices) + len(sell_indices)

        # Profit Factor: gross profits / gross losses
        profitable_days = daily_returns[daily_returns > 0]
        losing_days = abs(daily_returns[daily_returns < 0])
        profit_factor = (
            profitable_days.sum() / (losing_days.sum() + 1e-9)
            if len(losing_days) > 0 else float("inf")
        )

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "trade_count": trade_count,
            "final_portfolio": float(pv[-1]),
            "volatility": float(daily_returns.std() * np.sqrt(252)),
            "best_day": float(daily_returns.max()) if len(daily_returns) > 0 else 0.0,
            "worst_day": float(daily_returns.min()) if len(daily_returns) > 0 else 0.0,
        }