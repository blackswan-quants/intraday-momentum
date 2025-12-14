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
        self, df: pd.DataFrame, dividends : pd.DataFrame , save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute all metrics from a cleaned DataFrame and optionally save results.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with required columns : [ "close", "high", "low", "volume"]
            index must be the date
        df : Dividens 
            Input DataFrame with dividends
            index must be date
        save : bool
            Save results to disk if True.

        Returns
        -------
        df_all_days : pd.DataFrame
        df_daily_profiles : pd.DataFrame (aggregated per day df)
        """
        self._validate_input(df)

        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        # Extract day
        df["day"] = df.index.date

        # Logging
        self.logger.info("Starting computation of market microstructure metrics...")

        try:
            self.compute_bounds(df)
            self.compute_intraday_cum_vwap(df)
            self.merge_dividends(df,dividends)
            df_daily = self.compute_intraday_profiles(df)

        except Exception as exc:
            self.logger.error("Error computing metrics.", exc_info=True)
            raise RuntimeError("Metric computation failed.") from exc

        df = df.reset_index().rename(columns={"Datetime": "timestamp"})
        df_daily = df.reset_index().rename(columns={"Datetime": "timestamp"})
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
    
    def compute_bounds(self, df : pd.DataFrame) -> None:
        """
        Function that compute lower and upper bound
        """

        # Get first timestamp per day and expand to all rows in that day
        first_ts_per_day = df.groupby("day").apply(lambda x: x.index[0], include_groups=False)
        first_ts = df["day"].map(first_ts_per_day)

        df["minute_of_day"] = ((df.index - first_ts).dt.total_seconds() // 60).astype(int)

        open_930_per_day = (
            df.groupby("day").first()["open"]
        )

        df["open_at_930"] = df["day"].map(open_930_per_day)

        df["move"] = (df["close"] / df["open_at_930"] - 1).abs()

        agg = (
            df.groupby(["day", "minute_of_day"])
                .agg(move=("move", "last"))
                .reset_index()
        )
        agg = agg.sort_values(["minute_of_day", "day"])

        agg["sigma_roling_14d"] = (
            agg.groupby("minute_of_day")["move"]
            .rolling(14, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df = df.merge(
            agg[["day", "minute_of_day", "sigma"]],
            on=["day", "minute_of_day"],
            how="left"
        )

        # Get last close per day and shift to get previous day's close
        last_close_per_day = df.groupby("day")["close"].last()
        prev_close_per_day = last_close_per_day.shift(1)
        df["prev_close"] = df["day"].map(prev_close_per_day)

        df["open_ref"] = df[["open_at_930","prev_close"]].max(axis=1)
        df["low_ref"]  = df[["open_at_930","prev_close"]].min(axis=1)

        df["upper_bnd"] = df["open_ref"] * (1 + df["sigma"])
        df["lower_bnd"] = df["low_ref"]  * (1 - df["sigma"])
        df.drop(columns= ["open_ref", "low_ref", "prev_close", "open_at_930","move"])
    

    def compute_intraday_cum_vwap(self, df: pd.DataFrame) -> None:
        """Compute cumulative intraday VWAP for each day."""
        if "day" not in df.columns:
            raise ValueError("Column 'day' must exist before calling this method.")

        df["hlc"] = (df["high"] + df["low"] + df["close"]) / 3

        for d, group in df.groupby("day"):
            cum_vol_price = (group["hlc"] * group["volume"]).cumsum()
            cum_volume = group["volume"].cumsum()
            df.loc[group.index, "vwap"] = cum_vol_price / cum_volume

        
    def compute_intraday_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average intraday profiles."""
        if "minute_of_day" not in df.columns:
            df["minute_of_day"] = (df.index.hour * 60 + df.index.minute).astype(int)

        return (
            df.groupby("minute_of_day")[[]] #to change
            .mean()
            .astype(float)
        )

    def merge_dividends(self, df: pd.DataFrame, dividends: pd.DataFrame) -> None:
        """Merge dividend payments into the main dataframe."""
        if "day" not in df.columns:
            raise ValueError("Column 'day' must exist before calling merge_dividends.")

        dividends = dividends.copy()
        dividends["day"] = pd.to_datetime(dividends["caldt"]).dt.date

        df["dividend"] = df["day"].map(dividends.set_index("day")["dividend"]).fillna(0)


    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------
    def _save_results(self, df_all: pd.DataFrame, df_daily: pd.DataFrame) -> None:
        """Save results to separate pickle files."""
        try:
            os.makedirs(self.save_path, exist_ok=True)

            # Save df_all
            df_all_path = os.path.join(self.save_path, "df_all_days.pkl")
            pd.to_pickle(df_all, df_all_path)
            self.logger.info(f"df_all saved to {df_all_path}")

            # Save df_daily
            df_daily_path = os.path.join(self.save_path, "df_daily_groups.pkl")
            pd.to_pickle(df_daily, df_daily_path)
            self.logger.info(f"df_daily saved to {df_daily_path}")

        except Exception as exc:
            self.logger.error("Failed to save metrics.", exc_info=True)
            raise IOError(f"Error saving metrics to {self.save_path}") from exc

