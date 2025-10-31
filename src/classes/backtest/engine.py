import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# Auxiliar classes
class Position:
    def __init__(self, shares: int = 0, cash: float = 0.0) -> None:
        self.shares: int = shares
        self.cash: float = cash

    def value(self, price: float) -> float:
        return self.cash + self.shares * price


class Order:
    def __init__(
        self,
        timestamp: pd.Timestamp,
        day: pd.Timestamp,
        qty: int,           # signed (buy > 0, sell < 0)
        open_price: float,  # minute open price for simulation
        side: int,          # +1 for buy, -1 for sell
    ) -> None:
        self.timestamp: pd.Timestamp = timestamp
        self.day: pd.Timestamp = day
        self.qty: int = qty
        self.open_price: float = open_price
        self.side: int = side


# Sizer
class Sizer:
    """
    Determine target shares for the day.
    """

    def __init__(self, sizing_type: str = "vol_target") -> None:
        self.sizing_type: str = sizing_type

    def size_for_day(
        self,
        aum: float,
        target_vol: float,
        cap_leverage: float,
        price_open: float,
        spx_vol: Optional[float],
    ) -> int:
        """
        Returns an integer number of shares as target gross position for |exposure| == 1.
        """
        if self.sizing_type == "vol_target":
            if (spx_vol is None) or (pd.isna(spx_vol)) or (spx_vol <= 0):
                lev = cap_leverage
            else:
                lev = min(target_vol / spx_vol, cap_leverage)
            return int(round(max(0.0, lev) * (aum / price_open)))
        elif self.sizing_type == "full_notional":
            return int(round(aum / price_open))
        else:
            raise ValueError(f"Unknown sizing_type: {self.sizing_type}")


# Execution model
class ExecutionModel:
    """
    Marketable-on-open-at-minute bids/offers with optional slippage and commissions.
    """

    def __init__(self, commission_rate: float, min_comm_per_order: float, slippage_bps: int = 0) -> None:
        self.commission_rate: float = commission_rate
        self.min_comm_per_order: float = min_comm_per_order
        self.slippage_bps: int = slippage_bps

    def _slip_price(self, open_price: float, side: int) -> float:
        if self.slippage_bps <= 0:
            return open_price
        return open_price + side * (self.slippage_bps / 10_000.0) * open_price

    def _commission(self, qty: int) -> float:
        # Matches the original logic: commission based on shares, with a minimum per order.
        return max(self.min_comm_per_order, self.commission_rate * abs(qty))

    def execute_order(self, order: Order, position: Position) -> Dict[str, Any]:
        """
        Fill an order at minute open adjusted by slippage, charge commission, update position.
        Returns a dict suitable for appending to a trade log.
        """
        exec_price = self._slip_price(order.open_price, order.side)
        commission_paid = self._commission(order.qty)

        # Cash flow: buy reduces cash, sell increases cash
        notional = order.qty * exec_price
        position.cash -= notional
        position.cash -= commission_paid
        position.shares += order.qty

        return {
            "timestamp": order.timestamp,
            "day": order.day,
            "side": "BUY" if order.qty > 0 else "SELL",
            "qty": int(order.qty),
            "price_open": float(order.open_price),
            "price_exec": float(exec_price),
            "commission": float(commission_paid),
            "slippage_cost": float((exec_price - order.open_price) * order.qty),
            "shares_after": int(position.shares),
            "cash_after": float(position.cash),
        }


# Backtest engine
class BacktestEngine:
    """
    Reads cleaned_df.pkl and df_and_metrics.pkl, runs the strategy,
    simulates fills at minute open, applies commission/slippage, and
    outputs: trade_log_df, daily_pnl_df, equity_curve_df
    """

    def __init__(self) -> None:
        self.trade_rows: List[Dict[str, Any]] = []
        self.daily_rows: List[Dict[str, Any]] = []

    def _load_minute_data(self, path: str) -> pd.DataFrame:
        df = pd.read_pickle(path)
        if "timestamp" not in df.columns:
            raise ValueError("cleaned_df.pkl must include a 'timestamp' column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "day" not in df.columns:
            df["day"] = df["timestamp"].dt.date
        req = ["day", "open", "close", "vwap", "sigma_open", "spy_dvol", "min_from_open"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"cleaned_df.pkl missing required columns: {missing}")
        return df.sort_values("timestamp").reset_index(drop=True)

    def _load_daily_data(self, path: str) -> Optional[pd.DataFrame]:
        try:
            dfd = pd.read_pickle(path)
        except Exception:
            return None
        if {"caldt", "close"}.issubset(dfd.columns):
            out = dfd[["caldt", "close"]].copy()
            out["caldt"] = pd.to_datetime(out["caldt"]).dt.date
            out.set_index("caldt", inplace=True)
            out["ret_spy"] = out["close"].diff() / out["close"].shift()
            return out[["ret_spy"]]
        return None

    # Helpers
    @staticmethod
    def _compute_bands(
        open_price: float,
        prev_close_adj: float,
        sigma_open: float,
        band_mult: float,
    ) -> Tuple[float, float]:
        UB = max(open_price, prev_close_adj) * (1 + band_mult * sigma_open)
        LB = min(open_price, prev_close_adj) * (1 - band_mult * sigma_open)
        return UB, LB

    @staticmethod
    def _raw_signals(
        close_arr: np.ndarray,
        vwap_arr: np.ndarray,
        UB_arr: np.ndarray,
        LB_arr: np.ndarray,
    ) -> np.ndarray:
        sig = np.zeros_like(close_arr, dtype=int)
        sig[(close_arr > UB_arr) & (close_arr > vwap_arr)] = 1
        sig[(close_arr < LB_arr) & (close_arr < vwap_arr)] = -1
        return sig

    @staticmethod
    def _ffill_exposure_at_frequency(
        raw_signals: np.ndarray,
        minutes_from_open: np.ndarray,
        trade_freq: int,
        index_like: Iterable[Any],
    ) -> np.ndarray:
        TradeMask = (minutes_from_open % trade_freq) == 0
        exposure = np.full_like(raw_signals, np.nan, dtype=float)
        exposure[TradeMask] = raw_signals[TradeMask].astype(float)

        last_valid = np.nan
        filled: List[float] = []
        for val in exposure:
            if not np.isnan(val):
                last_valid = val
            if last_valid == 0:
                last_valid = np.nan
            filled.append(last_valid)

        exposure = pd.Series(filled, index=index_like).shift(1).fillna(0.0).values
        return exposure

    def run_backtest(self, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Params:
            minute_path
            daily_path
            initial_aum
            commission_rate
            min_comm_per_order
            slippage_bps
            band_mult
            trade_freq
            sizing_type
            target_vol
            max_leverage
        Returns:
            trade_log_df, daily_pnl_df (index=day), equity_curve_df (index=day)
        """

        minute_path = params.get("minute_path")
        daily_path = params.get("daily_path")
        initial_aum = float(params.get("initial_aum"))
        commission_rate = float(params.get("commission_rate"))
        min_comm_per_order = float(params.get("min_comm_per_order"))
        slippage_bps = int(params.get("slippage_bps"))
        band_mult = float(params.get("band_mult"))
        trade_freq = int(params.get("trade_freq"))
        sizing_type = params.get("sizing_type")
        target_vol = float(params.get("target_vol"))
        max_leverage = float(params.get("max_leverage"))

        # load data
        mdf = self._load_minute_data(minute_path)
        spy_daily = self._load_daily_data(daily_path)

        groups = mdf.groupby("day")
        all_days = sorted(mdf["day"].unique())

        # components
        sizer = Sizer(sizing_type=sizing_type)
        exec_model = ExecutionModel(
            commission_rate=commission_rate,
            min_comm_per_order=min_comm_per_order,
            slippage_bps=slippage_bps,
        )

        # account
        pos = Position(shares=0, cash=initial_aum)
        last_day_value: Optional[float] = None

        # loop by day
        for i in range(1, len(all_days)):
            prev_day = all_days[i - 1]
            day = all_days[i]

            if prev_day not in groups.indices or day not in groups.indices:
                continue

            prev_df = groups.get_group(prev_day)
            day_df = groups.get_group(day)

            # skip if sigma_open fully NaN
            if day_df["sigma_open"].isna().all():
                continue

            # Previous close adjusted by dividend if present
            dividend = 0.0
            if "dividend" in mdf.columns and len(day_df) > 0:
                try:
                    # align using current day's block indices (last one ok)
                    dividend = float(mdf.loc[day_df.index, "dividend"].iloc[-1])
                except Exception:
                    dividend = 0.0
            prev_close = float(prev_df["close"].iloc[-1])
            prev_close_adj = prev_close - dividend

            # series
            opens = day_df["open"].to_numpy(dtype=float)
            closes = day_df["close"].to_numpy(dtype=float)
            vwap = day_df["vwap"].to_numpy(dtype=float)
            sigma_open = day_df["sigma_open"].to_numpy(dtype=float)
            mfo = day_df["min_from_open"].to_numpy()
            timestamps = day_df["timestamp"].to_list()

            # Sizing at the day's first minute open using AUM marked on yesterday's close
            day_open = float(opens[0])
            aum_prev_close = pos.value(prev_close)
            spx_vol = float(day_df["spy_dvol"].iloc[0]) if "spy_dvol" in day_df.columns else np.nan
            base_shares = sizer.size_for_day(
                aum=aum_prev_close,
                target_vol=target_vol,
                cap_leverage=max_leverage,
                price_open=day_open,
                spx_vol=spx_vol,
            )

            # compute dynamic UB/LB per minute (sigma can vary intraday)
            UB = np.empty_like(closes)
            LB = np.empty_like(closes)
            for k in range(len(closes)):
                UB[k], LB[k] = self._compute_bands(
                    open_price=opens[0],
                    prev_close_adj=prev_close_adj,
                    sigma_open=sigma_open[k],
                    band_mult=band_mult,
                )

            raw_sig = self._raw_signals(closes, vwap, UB, LB)
            exposure = self._ffill_exposure_at_frequency(raw_sig, mfo, trade_freq, index_like=day_df.index)

            # minute execution loop
            prev_target = 0.0
            day_gross = 0.0
            day_comm = 0.0
            day_slip = 0.0

            for idx, row in enumerate(day_df.itertuples()):
                target = float(exposure[idx])  # -1, 0, +1
                if target != prev_target:
                    desired_shares = int(round(target * base_shares))
                    delta = desired_shares - pos.shares
                    if delta != 0:
                        side = 1 if delta > 0 else -1
                        order = Order(
                            timestamp=row.timestamp,
                            day=row.day,
                            qty=delta,
                            open_price=row.open,
                            side=side,
                        )
                        fill = exec_model.execute_order(order, pos)
                        self.trade_rows.append(fill)
                        day_comm += fill["commission"]
                        day_slip += fill["slippage_cost"]
                    prev_target = target

                # incremental P&L from minute close-to-close changes on current shares
                if idx > 0:
                    dP = float(closes[idx] - closes[idx - 1])
                    day_gross += pos.shares * dP

            # end-of-day
            day_close = float(closes[-1])
            aum_eod = pos.value(day_close)

            if last_day_value is None:
                day_ret = (aum_eod - initial_aum) / initial_aum
            else:
                day_ret = (aum_eod - last_day_value) / last_day_value

            ret_spy = np.nan
            if (spy_daily is not None) and (day in spy_daily.index):
                ret_spy = float(spy_daily.loc[day, "ret_spy"])

            self.daily_rows.append(
                {
                    "day": day,
                    "AUM": aum_eod,
                    "ret": day_ret,
                    "gross_pnl": day_gross,
                    "commission": day_comm,
                    "slippage_cost": day_slip,
                    "net_pnl": day_gross - day_comm - day_slip,
                    "ret_spy": ret_spy,
                }
            )
            last_day_value = aum_eod

        # assemble outputs
        trade_log_df = pd.DataFrame(self.trade_rows)
        daily_pnl_df = pd.DataFrame(self.daily_rows).set_index("day").sort_index()

        # equity curve (daily)
        equity_curve_df = daily_pnl_df[["AUM", "ret"]].copy()
        equity_curve_df["equity"] = equity_curve_df["AUM"]
        equity_curve_df["cumret"] = (1.0 + equity_curve_df["ret"].fillna(0)).cumprod() - 1.0

        # Output saving
        trade_log_path = os.path.join(os.getcwd(), "data", "processed", "trade_log.csv")
        daily_pnl_path = os.path.join(os.getcwd(), "data", "processed", "daily_pnl.pkl")

        os.makedirs(os.path.dirname(trade_log_path), exist_ok=True)

        trade_log_df.to_csv(trade_log_path, index=False)
        daily_pnl_df.to_pickle(daily_pnl_path)

        print(f"[INFO] Trade log saved to: {trade_log_path}")
        print(f"[INFO] Daily PnL saved to: {daily_pnl_path}")

        return trade_log_df, daily_pnl_df, equity_curve_df
