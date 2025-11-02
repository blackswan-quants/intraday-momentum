import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


path = "data/processed/"


class MetricsCalculator:
    """
    A class for computing daily and intraday market microstructure metrics from
    high-frequency financial data.

    The class calculates several volatility and liquidity measures such as
    Realized Volatility (RV), Bipower Variation (BV), and Volume-Weighted Average Price (VWAP),
    and produces both full-resolution and aggregated intraday profiles.

    Expected input columns:
        - Datetime : timestamp of each observation (datetime64)
        - close : closing price at that timestamp (float)
        - high : highest price during the period (float)
        - low : lowest price during the period (float)
        - volume : traded volume during the period (float)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def from_clean_df(
        self, df: pd.DataFrame, save=True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute all metrics from a cleaned DataFrame and optionally save results.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing at least the columns
            ['Datetime', 'close', 'high', 'low', 'volume'].
        save : bool, optional
            If True, saves computed dataframes as pickle files in `data/processed/`.

        Returns
        -------
        df_all_days : pd.DataFrame
            DataFrame augmented with computed columns:
            - 'log_returns' : log(price_t / price_{t-1})
            - 'RV' : daily realized volatility
            - 'BV' : daily bipower variation
            - 'vwap' : volume-weighted average price
        df_daily_groups : pd.DataFrame
            Intraday average profiles indexed by 'minute_of_day' (0–1439),
            with mean values of ['vwap', 'RV', 'BV', 'price', 'log_returns'].

        Notes
        -----
        Realized Volatility (RV) is computed as:

            RV_t = sqrt( Σ_i (r_{t,i})² )

        where r_{t,i} are intraday log-returns within day t.

        Bipower Variation (BV) is computed as:

            BV_t = Σ_i |r_{t,i-1}| * |r_{t,i}|

        VWAP is computed as:

            VWAP_t = Σ_i (p_{t,i} * v_{t,i}) / Σ_i v_{t,i}

        Units
        -----
        - Prices are in the same units as the input price columns.
        - RV and BV are in the same scale as log-returns (dimensionless).
        - VWAP is in price units.
        """

        required_cols = ["Datetime", "close", "high", "low", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.logger.error(f"Missing columns: {missing}")
            raise ValueError(f"Missing columns in the dataframe : {missing}")

        # Initialisation of the two desired data sets
        df["day"] = pd.to_datetime(df["Datetime"]).dt.date

        df_all_days = df.copy()

        # Apply Helpers
        self.logger.info("Started Computations:")
        self.compute_RV(df=df_all_days)
        self.logger.info("Realized Vol Computed:")
        self.compute_BV(df=df_all_days)
        self.logger.info("Bivariate Variations Computed:")
        self.compute_vwap(df=df_all_days)
        self.logger.info("Vwap Computed:")

        # daily_groups
        df_daily_groups = self.compute_intraday_profiles(df=df_all_days)
        self.logger.info("Per minute aggregations computed")
        self.logger.info("Finished Computations")

        self.quality_check(df_all_days, df_daily_groups)

        if save:
            try:
                os.makedirs(path, exist_ok=True)
                self.logger.info(f"Saving metrics to {path}")
                pd.to_pickle(
                    {"df_all_days": df_all_days, "df_daily_groups": df_daily_groups},
                    path + "df_and_metrics.pkl",
                )
            except Exception as e:
                self.logger.error(f"Error saving files: {e}", exc_info=True)
        return df_all_days, df_daily_groups

    # Helpers Functions

    def compute_RV(self, df: pd.DataFrame) -> None:
        """
        Compute daily Realized Volatility (RV) from intraday log-returns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least 'close' and 'Datetime' columns.

        Notes
        -----
        Realized Volatility is defined as:

            RV_t = sqrt( Σ_i (r_{t,i})² )

        where r_{t,i} = log(close_i / close_{i-1}) are intraday log-returns.

        Adds a new column:
            - 'RV' : float, same value for all rows belonging to the same day.
        """

        if not "log_returns" in df.columns:
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        rv = df.groupby("day")["log_returns"].apply(
            lambda x: np.sqrt(np.sum(x.dropna() ** 2))
        )

        df["RV"] = df["day"].map(rv, na_action="ignore")

    def compute_BV(self, df: pd.DataFrame) -> None:
        """
        Compute daily Bipower Variation (BV), a robust estimator of integrated variance.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a 'log_returns' column and a 'day' column.

        Notes
        -----
        Bipower Variation is defined as:

            BV_t = Σ_i |r_{t,i-1}| * |r_{t,i}|

        where r_{t,i} are intraday log-returns.

        Adds a new column:
            - 'BV' : float, same value for all rows belonging to the same day.
        """

        bv = df.groupby("day")["log_returns"].apply(
            lambda x: np.sum(np.abs(x.shift(1).dropna()) * np.abs(x))
        )
        # Map back to each row
        df["BV"] = df["day"].map(bv, na_action="ignore")

    def compute_vwap(self, df: pd.DataFrame) -> None:
        """
        Compute daily Volume-Weighted Average Price (VWAP).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'high', 'low', 'close', 'volume', and 'day' columns.

        Notes
        -----
        VWAP is defined as:

            VWAP_t = Σ_i (p_{t,i} * v_{t,i}) / Σ_i v_{t,i}

        where p_{t,i} = (high_i + low_i + close_i) / 3.

        Adds a new column:
            - 'vwap' : float, same value for all rows belonging to the same day.
        """
        # Compute price
        df["price"] = (df["high"] + df["close"] + df["low"]) / 3
        # Compute vwap
        vwap = df.groupby("day")[["price", "volume"]].apply(
            lambda x: (np.sum(x["volume"] * x["price"])) / np.sum(x["volume"])
        )
        # Map back to each row
        df["vwap"] = df["day"].map(vwap, na_action="ignore")

    def compute_intraday_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute intraday average profiles across all days.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns:
            ['Datetime', 'vwap', 'RV', 'BV', 'price', 'log_returns'].

        Returns
        -------
        out : pd.DataFrame
            DataFrame indexed by 'minute_of_day' (0–1439),
            containing the mean of each metric across all days.

        Notes
        -----
        - The 'minute_of_day' column is computed as: 60 * hour + minute.
        - Each row of `out` represents the average behavior at that time of day.

        Units
        -----
        Same as the corresponding metrics:
        - RV, BV, log_returns → dimensionless
        - price, vwap → price units
        """

        if "minute_of_day" not in df.columns:
            df["minute_of_day"] = (
                pd.to_datetime(df["Datetime"]).dt.hour * 60
                + pd.to_datetime(df["Datetime"]).dt.minute
            )

        out = df.groupby("minute_of_day")[
            ["vwap", "RV", "BV", "price", "log_returns"]
        ].apply(lambda x: np.mean(x))
        return out

    def quality_check(
        self,
        df_all: pd.DataFrame,
        df_daily: pd.DataFrame,
        expected_cols: list[str] = None,
    ) -> None:
        """
        Perform a quality check on computed metrics to ensure data integrity.

        Parameters
        ----------
        df_all : pd.DataFrame
            The full dataset containing computed metrics such as RV, BV, and VWAP.
        df_daily : pd.DataFrame
            The intraday profiles dataframe produced by compute_intraday_profiles.
        expected_cols : list of str, optional
            Expected columns to verify in df_all. Defaults to
            ['log_returns', 'RV', 'BV', 'vwap', 'price', 'day'].

        Notes
        -----
        - Logs missing columns if any.
        - Logs NaN counts for computed metrics.
        - Checks for negative values in RV and BV.
        - Logs final dataframe shapes.
        """

        if expected_cols is None:
            expected_cols = ["log_returns", "RV", "BV", "vwap", "price", "day"]

        self.logger.info("Starting quality check of computed metrics...")

        # Check for missing columns
        missing_cols = [col for col in expected_cols if col not in df_all.columns]
        if missing_cols:
            self.logger.error(
                f"Missing computed columns after processing: {missing_cols}"
            )
        else:
            self.logger.info("All expected metrics columns are present in df_all_days")

        # Check for NaN values
        nan_summary = df_all[expected_cols].isna().sum()
        self.logger.info(f"NaN summary for computed metrics:\n{nan_summary}")

        # Log DataFrame shapes
        self.logger.info(f"Final df_all_days shape: {df_all.shape}")
        self.logger.info(f"Final df_daily_groups shape: {df_daily.shape}")

        # Check for invalid (negative) values in variance measures
        if (df_all["RV"] < 0).any() or (df_all["BV"] < 0).any():
            self.logger.warning(
                "Detected negative values in RV or BV. Check computation steps."
            )
        else:
            self.logger.info("RV and BV values are all non-negative")

        self.logger.info("Quality check completed successfully.")
