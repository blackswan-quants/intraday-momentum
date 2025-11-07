from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path


import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


# Auxiliary classes
class Position:
    """
    Lightweight account state holding share count and cash.

    Attributes
    ----------
    shares : int
        Current number of shares held (signed).
    cash : float
        Current cash balance.
    """

    def __init__(self, shares: int = 0, cash: float = 0.0) -> None:
        """
        Initialize a Position.

        Parameters
        ----------
        shares : int, default=0
            Starting share balance (signed).
        cash : float, default=0.0
            Starting cash balance.
        """
        self.shares: int = shares
        self.cash: float = cash

    def value(self, price: float) -> float:
        """
        Compute mark-to-market account value.

        Parameters
        ----------
        price : float
            Current instrument price.

        Returns
        -------
        float
            Total account value (cash + shares * price).

        Notes
        -----
        The account value :math:`V_t` is

        .. math::

            V_t = \\text{cash}_t + \\text{shares}_t \\times P_t \\, .
        """
        return self.cash + self.shares * price


class Order:
    """
    Simple order container for open-at-minute executions.

    Attributes
    ----------
    timestamp : pandas.Timestamp
        Intraday timestamp of the bar used for execution.
    day : pandas.Timestamp
        Session/day identifier corresponding to `timestamp`.
    qty : int
        Signed order quantity (buy > 0, sell < 0).
    open_price : float
        Minute open price used for simulated execution.
    side : int
        +1 for buy, -1 for sell.
    """

    def __init__(
        self,
        timestamp: pd.Timestamp,
        day: pd.Timestamp,
        qty: int,           # signed (buy > 0, sell < 0)
        open_price: float,  # minute open price for simulation
        side: int,          # +1 for buy, -1 for sell
    ) -> None:
        """
        Initialize an Order.

        Parameters
        ----------
        timestamp : pandas.Timestamp
            Intraday timestamp for the intended execution.
        day : pandas.Timestamp
            Trading day/session for the order.
        qty : int
            Signed order quantity (buy > 0, sell < 0).
        open_price : float
            Minute open price for the bar used in simulation.
        side : int
            Execution side: +1 for buy, -1 for sell.
        """
        self.timestamp: pd.Timestamp = timestamp
        self.day: pd.Timestamp = day
        self.qty: int = qty
        self.open_price: float = open_price
        self.side: int = side


# Sizer
class Sizer:
    """
    Determine target shares for the day.

    Parameters
    ----------
    sizing_type : str, default="vol_target"
        Sizing rule to use. Supported:
        - ``"vol_target"``: target volatility with leverage cap.
        - ``"full_notional"``: fully invest AUM at the open.

    Attributes
    ----------
    sizing_type : str
        Sizing method used by the sizer.
    """

    def __init__(self, sizing_type: str = "vol_target") -> None:
        """
        Initialize a Sizer.

        Parameters
        ----------
        sizing_type : str, default="vol_target"
            Sizing rule to use: ``"vol_target"`` or ``"full_notional"``.
        """
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
        Compute the target *gross* shares for |exposure| == 1 for the day.

        Parameters
        ----------
        aum : float
            Mark-to-market assets under management at prior close.
        target_vol : float
            Target daily volatility (as a fraction, e.g., 0.02 for 2%).
        cap_leverage : float
            Maximum gross leverage allowed.
        price_open : float
            Today's first minute open price (used for sizing).
        spx_vol : float or None
            Exogenous volatility proxy (e.g., SPX daily vol). If missing
            or non-positive, leverage falls back to `cap_leverage`.

        Returns
        -------
        int
            Integer number of shares corresponding to |exposure| == 1.

        Notes
        -----
        For ``sizing_type == "vol_target"`` the leverage :math:`\\ell_t` is

        .. math::

            \\ell_t = \\min\\Big( \\tfrac{\\sigma^\\star}{\\sigma_{\\text{SPX},t}},\\; \\text{cap} \\Big)

        with fallback :math:`\\ell_t = \\text{cap}` if :math:`\\sigma_{\\text{SPX},t}` is unavailable,
        and shares

        .. math::

            N_t = \\operatorname{round}\\big( \\ell_t \\cdot \\tfrac{\\text{AUM}_{t-1}}{P_{t,\\text{open}}} \\big) \\, .

        For ``"full_notional"``, :math:`N_t = \\operatorname{round}( \\text{AUM}_{t-1}/P_{t,\\text{open}} )`.
        """
        if price_open is None or price_open <= 0.0:
            return 0

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

    Parameters
    ----------
    commission_rate : float
        Per-share commission rate.
    min_comm_per_order : float
        Minimum commission charged per order.
    slippage_bps : int, default=0
        Symmetric slippage in basis points applied to the open price.

    Attributes
    ----------
    commission_rate : float
        Per-share commission rate.
    min_comm_per_order : float
        Minimum commission per order.
    slippage_bps : int
        Slippage in basis points applied to the execution price.
    """

    def __init__(self, commission_rate: float, min_comm_per_order: float, slippage_bps: int = 0) -> None:
        """
        Initialize an ExecutionModel.

        Parameters
        ----------
        commission_rate : float
            Per-share commission rate.
        min_comm_per_order : float
            Minimum commission per order.
        slippage_bps : int, default=0
            Slippage in basis points added (buy) or subtracted (sell) from the open price.
        """
        self.commission_rate: float = commission_rate
        self.min_comm_per_order: float = min_comm_per_order
        self.slippage_bps: int = slippage_bps

    def _slip_price(self, open_price: float, side: int) -> float:
        """
        Apply slippage to the minute open price.

        Parameters
        ----------
        open_price : float
            Minute open price.
        side : int
            +1 for buy, -1 for sell.

        Returns
        -------
        float
            Executed price after slippage.

        Notes
        -----
        Slippage is applied as:

        .. math::

            P_{\\text{exec}} = P_{\\text{open}} + \\text{side} \\times
            \\Big( \\tfrac{\\text{bps}}{10{,}000} \\Big) P_{\\text{open}} \\, .
        """
        if self.slippage_bps <= 0:
            return open_price
        return open_price + side * (self.slippage_bps / 10_000.0) * open_price

    def _commission(self, qty: int) -> float:
        """
        Compute commission for an order.

        Parameters
        ----------
        qty : int
            Signed share quantity (sign ignored).

        Returns
        -------
        float
            Commission charged, respecting the per-order minimum.

        Notes
        -----
        .. math::

            \\text{commission} = \\max(\\text{min},\\; \\text{rate} \\times |\\text{qty}|) \\, .
        """
        # Matches the original logic: commission based on shares, with a minimum per order.
        return max(self.min_comm_per_order, self.commission_rate * abs(qty))

    def execute_order(self, order: Order, position: Position) -> Dict[str, Any]:
        """
        Execute an order at minute open (with slippage), charge commission, and update the position.

        Parameters
        ----------
        order : Order
            Order to be executed.
        position : Position
            Mutable account state to update with cash and shares changes.

        Returns
        -------
        dict
            A dictionary suitable for appending to a trade log with keys:
            ``timestamp``, ``day``, ``side``, ``qty``, ``price_open``,
            ``price_exec``, ``commission``, ``slippage_cost``,
            ``shares_after``, ``cash_after``.

        Notes
        -----
        Cash flow convention:

        .. math::

            \\Delta \\text{cash} = -\\text{qty} \\times P_{\\text{exec}} - \\text{commission} \\, ,\\quad
            \\Delta \\text{shares} = \\text{qty} \\, .
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
    Run the strategy over minute data, simulate fills at minute opens, apply
    commission/slippage, and produce trade logs, daily PnL, and equity curves.

    Attributes
    ----------
    trade_rows : list of dict
        Accumulated trade log rows produced during the run.
    daily_rows : list of dict
        Accumulated daily PnL rows produced during the run.
    """

    def __init__(self) -> None:
        """
        Initialize an empty BacktestEngine.
        """
        self.trade_rows: List[Dict[str, Any]] = []
        self.daily_rows: List[Dict[str, Any]] = []

    def _load_minute_data(self, path: str) -> pd.DataFrame:
        """
        Load and validate minute-level data.

        Parameters
        ----------
        path : str
            Path to a pickled pandas DataFrame with minute bars. Must include
            columns: ``timestamp``, and (or derived) ``day`` plus
            ``open``, ``close``, ``vwap``, ``sigma_open``, ``spy_dvol``, ``min_from_open``.

        Returns
        -------
        pandas.DataFrame
            Minute data sorted by ``timestamp`` with ensured ``day`` column.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
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
        """
        Load a daily benchmark series (e.g., SPY) and compute daily returns.

        Parameters
        ----------
        path : str
            Path to a pickled pandas DataFrame expected to contain columns
            ``caldt`` (calendar date) and ``close`` (close price).

        Returns
        -------
        pandas.DataFrame or None
            DataFrame indexed by ``caldt`` with a single column ``ret_spy`` if
            available; otherwise ``None``.

        Notes
        -----
        Daily return is computed as:

        .. math::

            r_t = \\frac{P_t - P_{t-1}}{P_{t-1}} \\, .
        """
        try:
            dfd = pd.read_pickle(path)
        except Exception as e:
            print(f"[WARNING] Could not load daily data from '{path}': {e}")
            return None

        if {"caldt", "close"}.issubset(dfd.columns):
            out = dfd[["caldt", "close"]].copy()
            out["caldt"] = pd.to_datetime(out["caldt"]).dt.date
            out.set_index("caldt", inplace=True)
            out["ret_spy"] = out["close"].diff() / out["close"].shift()
            return out[["ret_spy"]]
        else:
            print(f"[WARNING] Daily data at '{path}' is missing required columns: 'caldt' and/or 'close'.")
            return None

    # Helpers
    @staticmethod
    def _compute_bands(
        open_price: float,
        prev_close_adj: float,
        sigma_open: float,
        band_mult: float,
    ) -> Tuple[float, float]:
        """
        Compute upper/lower bands for breakout signals.

        Parameters
        ----------
        open_price : float
            First minute open price of the day.
        prev_close_adj : float
            Previous day's close adjusted for any dividend.
        sigma_open : float
            Intraday volatility proxy for the minute (can vary through the day).
        band_mult : float
            Multiplier applied to `sigma_open`.

        Returns
        -------
        (float, float)
            Tuple of (UB, LB) for the minute.

        Notes
        -----
        The bands are computed as:

        .. math::

            \\text{UB} = \\max(P_{\\text{open}}, P_{\\text{prev,adj}}) (1 + m \\sigma),\\quad
            \\text{LB} = \\min(P_{\\text{open}}, P_{\\text{prev,adj}}) (1 - m \\sigma) \\, .
        """
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
        """
        Generate raw +/- 1 breakout signals against VWAP and UB/LB.

        Parameters
        ----------
        close_arr : numpy.ndarray
            Minute close prices.
        vwap_arr : numpy.ndarray
            Minute VWAP series.
        UB_arr : numpy.ndarray
            Minute upper-band series.
        LB_arr : numpy.ndarray
            Minute lower-band series.

        Returns
        -------
        numpy.ndarray
            Array of signals in {-1, 0, +1} where +1 indicates long breakout
            and -1 indicates short breakdown.
        """
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
        ffill_zero: bool = False,
    ) -> np.ndarray:
        """
        Convert sparse trade-time signals into next-bar exposure by forward-filling.

        Parameters
        ----------
        raw_signals : numpy.ndarray
            Array in {-1, 0, +1} at minute resolution.
        minutes_from_open : numpy.ndarray
            Minute offset (0-based) from session open.
        trade_freq : int
            Trading frequency in minutes; signals are *eligible* only where
            ``minutes_from_open % trade_freq == 0``.
        index_like : Iterable[Any]
            Index used to align a Pandas shift; typically the minute index for the day.
        ffill_zero : bool, default False
            If False (default), a zero at an eligible trading minute is treated as an
            explicit “flat **for this bar only**” instruction and is **not** forward-filled
            into subsequent non-trading minutes. If True, zeros are carried forward like
            -1/+1.

        Returns
        -------
        numpy.ndarray
            Exposure array in {-1.0, 0.0, +1.0} shifted by one bar (enter on next bar).

        Notes
        -----
        - On eligible trading minutes, exposure takes the last non-null signal.
        - Between trading minutes, exposure is forward-filled.
        - A one-bar execution lag is applied via `shift(1)`.
        - By default, zeros are ephemeral (do not persist), which prevents an
          explicit “go flat now” from unintentionally suppressing later fills until
          the next trade-eligible bar.
        """
        TradeMask = (minutes_from_open % trade_freq) == 0
        exposure = np.full_like(raw_signals, np.nan, dtype=float)
        exposure[TradeMask] = raw_signals[TradeMask].astype(float)

        last_valid = np.nan
        filled: List[float] = []
        for val in exposure:
            if not np.isnan(val):
                last_valid = val
            # Rationale:
            # Treat zero as an ephemeral "flat" signal by default: do not let it
            # propagate into non-trading minutes unless explicitly requested.
            if (last_valid == 0) and (not ffill_zero):
                last_valid = np.nan
            filled.append(last_valid)

        exposure = pd.Series(filled, index=index_like).shift(1).fillna(0.0).values
        return exposure

    def run_backtest(self, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the backtest over minute data and produce outputs.

        Parameters
        ----------
        params : dict
            Dictionary of parameters:
            - ``minute_path`` : str
            - ``daily_path`` : str
            - ``initial_aum`` : float
            - ``commission_rate`` : float
            - ``min_comm_per_order`` : float
            - ``slippage_bps`` : int
            - ``band_mult`` : float
            - ``trade_freq`` : int
            - ``sizing_type`` : str
            - ``target_vol`` : float
            - ``max_leverage`` : float

        Returns
        -------
        (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
            ``trade_log_df`` (row per fill),
            ``daily_pnl_df`` (index = day),
            ``equity_curve_df`` (index = day).

        Notes
        -----
        Intraday gross P&L is accumulated from minute close-to-close changes:

        .. math::

            \\text{dP&L}_{t,i} = N_{t,i-1} (C_{t,i} - C_{t,i-1}) \\quad \\Rightarrow \\quad
            \\text{GrossP&L}_t = \\sum_i \\text{dP&L}_{t,i} \\, .

        Daily return is computed from end-of-day AUM, net of transaction costs
        (commissions and slippage costs recorded at each fill).

        Side Effects
        ------------
        Saves outputs to:
        - ``data/processed/trade_log.csv``
        - ``data/processed/daily_pnl.pkl``
        """
        minute_path: str = str(params.get("minute_path"))
        daily_path: str = str(params.get("daily_path"))
        initial_aum: float = float(params.get("initial_aum"))
        commission_rate: float = float(params.get("commission_rate"))
        min_comm_per_order: float = float(params.get("min_comm_per_order"))
        slippage_bps: int = int(params.get("slippage_bps"))
        band_mult: float = float(params.get("band_mult"))
        trade_freq: int = int(params.get("trade_freq"))
        sizing_type: str = str(params.get("sizing_type"))
        target_vol: float = float(params.get("target_vol"))
        max_leverage: float = float(params.get("max_leverage"))

        # load data
        mdf: pd.DataFrame = self._load_minute_data(minute_path)
        spy_daily: Optional[pd.DataFrame] = self._load_daily_data(daily_path)

        groups = mdf.groupby("day")
        all_days = sorted(mdf["day"].unique().tolist())

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
            dividend: float = 0.0
            if "dividend" in mdf.columns and len(day_df) > 0:
                try:
                    # align using current day's block indices (last one ok)
                    dividend = float(mdf.loc[day_df.index, "dividend"].iloc[-1])
                except Exception:
                    dividend = 0.0
            prev_close: float = float(prev_df["close"].iloc[-1])
            prev_close_adj: float = prev_close - dividend

            # series
            opens: np.ndarray = day_df["open"].to_numpy(dtype=float)
            closes: np.ndarray = day_df["close"].to_numpy(dtype=float)
            vwap: np.ndarray = day_df["vwap"].to_numpy(dtype=float)
            sigma_open: np.ndarray = day_df["sigma_open"].to_numpy(dtype=float)
            mfo: np.ndarray = day_df["min_from_open"].to_numpy()

            # Sizing at the day's first minute open using AUM marked on yesterday's close
            day_open: float = float(opens[0])
            aum_prev_close: float = pos.value(prev_close)
            spx_vol: float = float(day_df["spy_dvol"].iloc[0]) if "spy_dvol" in day_df.columns else np.nan
            base_shares: int = int(
                sizer.size_for_day(
                    aum=aum_prev_close,
                    target_vol=target_vol,
                    cap_leverage=max_leverage,
                    price_open=day_open,
                    spx_vol=spx_vol,
                )
            )

            # compute dynamic UB/LB per minute (sigma can vary intraday)
            UB: np.ndarray = np.empty_like(closes)
            LB: np.ndarray = np.empty_like(closes)
            for k in range(len(closes)):
                UB[k], LB[k] = self._compute_bands(
                    open_price=opens[0],
                    prev_close_adj=prev_close_adj,
                    sigma_open=float(sigma_open[k]),
                    band_mult=band_mult,
                )

            raw_sig: np.ndarray = self._raw_signals(closes, vwap, UB, LB)
            exposure: np.ndarray = self._ffill_exposure_at_frequency(raw_sig, mfo, trade_freq, index_like=day_df.index)

            # minute execution loop
            prev_target: float = 0.0
            day_gross: float = 0.0
            day_comm: float = 0.0
            day_slip: float = 0.0

            for idx, row in enumerate(day_df.itertuples()):
                target: float = float(exposure[idx])  # -1, 0, +1
                if target != prev_target:
                    desired_shares: int = int(round(target * base_shares))
                    delta: int = desired_shares - pos.shares
                    if delta != 0:
                        side: int = 1 if delta > 0 else -1
                        order = Order(
                            timestamp=row.timestamp,
                            day=row.day,
                            qty=delta,
                            open_price=row.open,
                            side=side,
                        )
                        fill = exec_model.execute_order(order, pos)
                        self.trade_rows.append(fill)
                        day_comm += float(fill["commission"])
                        day_slip += float(fill["slippage_cost"])
                    prev_target = target

                # incremental P&L from minute close-to-close changes on current shares
                if idx > 0:
                    dP: float = float(closes[idx] - closes[idx - 1])
                    day_gross += pos.shares * dP

            # end-of-day
            day_close: float = float(closes[-1])
            aum_eod: float = pos.value(day_close)

            if last_day_value is None:
                day_ret: float = (aum_eod - initial_aum) / initial_aum
            else:
                day_ret = (aum_eod - last_day_value) / last_day_value

            ret_spy: float = np.nan
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

        # Configure module-level logger
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

        # Configure data directory
        DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Assemble outputs
        trade_log_df = pd.DataFrame(self.trade_rows)
        daily_pnl_df = (
            pd.DataFrame(self.daily_rows)
            .set_index("day")
            .sort_index()
        )

        # Equity curve (daily)
        equity_curve_df = daily_pnl_df[["AUM", "ret"]].copy()
        equity_curve_df["equity"] = equity_curve_df["AUM"]
        equity_curve_df["cumret"] = (1.0 + equity_curve_df["ret"].fillna(0)).cumprod() - 1.0

        # Output saving
        trade_log_path = DATA_DIR / "trade_log.csv"
        daily_pnl_path = DATA_DIR / "daily_pnl.pkl"

        try:
            trade_log_df.to_csv(trade_log_path, index=False)
            daily_pnl_df.to_pickle(daily_pnl_path)

            logger.info("Trade log saved to %s", trade_log_path)
            logger.info("Daily PnL saved to %s", daily_pnl_path)
        except Exception:
            logger.exception("Failed to save outputs")
            raise

        return trade_log_df, daily_pnl_df, equity_curve_df
