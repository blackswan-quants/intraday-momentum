import logging
import os
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class MetricsCalculator:
    """
    Compute daily and intraday market microstructure metrics from high-frequency financial data.

    This class calculates:
        - Log returns
        - Realized Volatility (RV)
        - Bipower Variation (BV)
        - Daily VWAP (volume-weighted average price)
        - Intraday aggregated profiles

    Example input columns:
        'Datetime' : datetime64
        'close'    : float
        'high'     : float
        'low'      : float
        'volume'   : float
    """

    REQUIRED_COLUMNS = ["close", "high", "low", "volume"]

    def __init__(self, save_path: str = "data/processed/") -> None:
        """
        Initialize the calculator.

        Parameters
        ----------
        save_path : str
            Directory where computed metrics will be stored.
        """
        self.logger = logger
        self.save_path = save_path

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------
    def from_clean_df(
        self, df: pd.DataFrame, save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute all metrics from a cleaned DataFrame and optionally save results.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with required columns.
        save : bool
            Save results to disk if True.

        Returns
        -------
        df_all_days : pd.DataFrame
        df_daily_profiles : pd.DataFrame
        """
        self._validate_input(df)

        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        # Extract day
        df["day"] = df.index.date

        # Logging
        self.logger.info("Starting computation of market microstructure metrics...")

        try:
            self.compute_log_returns(df)
            self.compute_RV(df)
            self.compute_BV(df)
            self.compute_vwap(df)
            df_daily = self.compute_intraday_profiles(df)

        except Exception as exc:
            self.logger.error("Error computing metrics.", exc_info=True)
            raise RuntimeError("Metric computation failed.") from exc

        self.quality_check(df, df_daily)

        if save:
            self._save_results(df, df_daily)

        return df, df_daily

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Check if the input DataFrame has required columns."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # -------------------------------------------------------------------------
    # Computations
    # -------------------------------------------------------------------------
    def compute_log_returns(self, df: pd.DataFrame) -> None:
        """Compute log returns."""
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    def compute_RV(self, df: pd.DataFrame) -> None:
        """Compute daily Realized Volatility."""
        rv = (
            df.groupby("day")["log_returns"]
            .apply(lambda x: np.sqrt(np.sum(np.square(x.dropna()))))
            .astype(float)
        )
        df["RV"] = df["day"].map(rv)

    def compute_BV(self, df: pd.DataFrame) -> None:
        """Compute Bipower Variation."""
        bv = (
            df.groupby("day")["log_returns"]
            .apply(lambda x: np.sum(np.abs(x.shift(1).dropna()) * np.abs(x)))
            .astype(float)
        )
        df["BV"] = df["day"].map(bv)

    def compute_vwap(self, df: pd.DataFrame) -> None:
        """Compute daily VWAP."""
        df["price"] = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (
            df.groupby("day")[["price", "volume"]]
            .apply(lambda x: (x["price"] * x["volume"]).sum() / x["volume"].sum())
            .astype(float)
        )
        df["vwap"] = df["day"].map(vwap)

    def compute_intraday_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average intraday profiles."""
        if "minute_of_day" not in df.columns:
            df["minute_of_day"] = (df.index.hour * 60 + df.index.minute).astype(int)

        return (
            df.groupby("minute_of_day")[["vwap", "RV", "BV", "price", "log_returns"]]
            .mean()
            .astype(float)
        )

    # -------------------------------------------------------------------------

    # Additional SPY Intraday Metrics
    # -------------------------------------------------------------------------

    def compute_intraday_cum_vwap(self, df: pd.DataFrame) -> None:
        """Compute cumulative intraday VWAP for each day."""
        if "day" not in df.columns:
            raise ValueError("Column 'day' must exist before calling this method.")

        df["hlc"] = (df["high"] + df["low"] + df["close"]) / 3

        for d, group in df.groupby("day"):
            cum_vol_price = (group["hlc"] * group["volume"]).cumsum()
            cum_volume = group["volume"].cumsum()
            df.loc[group.index, "vwap"] = cum_vol_price / cum_volume

    def compute_move_open(self, df: pd.DataFrame) -> None:
        """Compute intraday absolute move from daily open."""
        if "day" not in df.columns:
            raise ValueError("Column 'day' must exist before calling this method.")

        df["move_open"] = np.nan

        for d, group in df.groupby("day"):
            open_price = group["open"].iloc[0]
            df.loc[group.index, "move_open"] = (group["close"] / open_price - 1).abs()

    def compute_daily_returns_and_vol(self, df: pd.DataFrame) -> None:
        """Compute daily returns and 15-day rolling volatility."""
        if "day" not in df.columns:
            raise ValueError("Column 'day' must exist before calling this method.")

        days = df["day"].unique()
        daily_groups = df.groupby("day")
        spy_ret = pd.Series(index=days, dtype=float)
        df["spy_dvol"] = np.nan

        for i in range(1, len(days)):
            cur, prev = days[i], days[i - 1]
            cur_close = daily_groups.get_group(cur)["close"].iloc[-1]
            prev_close = daily_groups.get_group(prev)["close"].iloc[-1]
            spy_ret.loc[cur] = cur_close / prev_close - 1

            if i > 14:
                df.loc[daily_groups.get_group(cur).index, "spy_dvol"] = spy_ret.iloc[
                    i - 15 : i
                ].std()

        df["spy_ret"] = df["day"].map(spy_ret)

    def compute_minute_features(self, df: pd.DataFrame) -> None:
        """Compute minute_of_day, rolling mean and sigma for move_open."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DatetimeIndex required for minute_of_day computation.")

        df["min_from_open"] = (
            ((df.index - df.index.normalize()) / pd.Timedelta(minutes=1))
            - (9 * 60 + 30)
            + 1
        )

        df["minute_of_day"] = df["min_from_open"].round().astype(int)

        groups = df.groupby("minute_of_day")

        df["move_open_rolling_mean"] = groups["move_open"].transform(
            lambda x: x.rolling(window=14, min_periods=13).mean()
        )

        df["sigma_open"] = df.groupby("minute_of_day")[
            "move_open_rolling_mean"
        ].transform(lambda x: x.shift(1))

    def merge_dividends(self, df: pd.DataFrame, dividends: pd.DataFrame) -> None:
        """Merge dividend payments into the main dataframe."""
        if "day" not in df.columns:
            raise ValueError("Column 'day' must exist before calling merge_dividends.")

        dividends = dividends.copy()
        dividends["day"] = pd.to_datetime(dividends["caldt"]).dt.date

        df["dividend"] = df["day"].map(dividends.set_index("day")["dividend"]).fillna(0)

    # -------------------------------------------------------------------------
    # Quality Check
    # -------------------------------------------------------------------------
    def quality_check(
        self,
        df_all: pd.DataFrame,
        df_daily: pd.DataFrame,
        expected: Optional[List[str]] = None,
    ) -> None:
        """Perform data integrity checks and log results."""
        expected = expected or [
            "log_returns",
            "RV",
            "BV",
            "vwap",
            "price",
            "day",
        ]

        self.logger.info("Running quality checks...")

        missing_cols = [c for c in expected if c not in df_all.columns]
        if missing_cols:
            self.logger.warning(f"Missing expected columns: {missing_cols}")

        nan_summary = df_all[expected].isna().sum()
        self.logger.info(f"NaN summary:\n{nan_summary}")

        if (df_all["RV"] < 0).any() or (df_all["BV"] < 0).any():
            self.logger.warning("Negative values detected in RV/BV.")

        self.logger.info(f"df_all_days shape: {df_all.shape}")
        self.logger.info(f"df_daily_profiles shape: {df_daily.shape}")

    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------
    def _save_results(self, df_all: pd.DataFrame, df_daily: pd.DataFrame) -> None:
        """Save results to pickle."""
        try:
            os.makedirs(self.save_path, exist_ok=True)
            out_path = os.path.join(self.save_path, "df_and_metrics.pkl")

            pd.to_pickle(
                {"df_all_days": df_all, "df_daily_groups": df_daily},
                out_path,
            )
            self.logger.info(f"Metrics saved to {out_path}")

        except Exception as exc:
            self.logger.error("Failed to save metrics.", exc_info=True)
            raise IOError(f"Error saving metrics to {self.save_path}") from exc
