import requests
from requests.exceptions import HTTPError, RequestException, Timeout
import dotenv
import os
import time
import logging
from pathlib import Path
from datetime import datetime, time as dt_time
from typing import Tuple, Dict, Optional, List, Union, Any
import pytz
import pandas as pd
import numpy as np
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse


# Configure logging - use module-level logger that can be configured externally
logger = logging.getLogger(__name__)

# Set default handler if none exists (only if no handlers are configured)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Default to INFO, but can be overridden


def set_log_level(level: Union[int, str]) -> None:
    """Set the logging level for the DataLoader module.
    
    Convenience function to configure logging level from outside the module.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG, 'INFO', 'DEBUG')
              Can be a logging constant or string name.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    # Also update handler if it exists
    for handler in logger.handlers:
        handler.setLevel(level)


def get_project_root() -> Path:
    """Get the project root directory by looking for pyproject.toml or .git.

    Returns:
        Path: Path object pointing to the project root directory.

    Raises:
        FileNotFoundError: If neither pyproject.toml nor .git is found.
    """
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise FileNotFoundError(
        "Project root not found. Ensure you're in a project with pyproject.toml or .git"
    )


def _retry_request_with_backoff(
    url: str,
    max_attempts: int = 3,
    base_wait: int = 5,
    timeout: Tuple[int, int] = (10, 30),
) -> requests.Response:
    """Make HTTP request with exponential backoff retry logic.

    Args:
        url: URL to request
        max_attempts: Maximum number of retry attempts
        base_wait: Base wait time in seconds for exponential backoff
        timeout: Tuple of (connect_timeout, read_timeout) in seconds

    Returns:
        Response object from successful request

    Raises:
        HTTPError: If request fails after all retries with HTTP error
        RequestException: If request fails due to network issues
        Timeout: If request times out after all retries
    """
    last_response = None

    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=timeout)
            last_response = response

            if response.status_code == 200:
                return response
            elif response.status_code in [429, 500, 502, 503, 504]:
                if attempt < max_attempts - 1:
                    wait = base_wait * (2**attempt)
                    logger.warning(
                        f"Request failed ({response.status_code}). "
                        f"Retrying in {wait}s... (attempt {attempt + 1}/{max_attempts})"
                    )
                    time.sleep(wait)
                else:
                    response.raise_for_status()
            else:
                response.raise_for_status()

        except (Timeout, ConnectionError) as e:
            if attempt < max_attempts - 1:
                wait = base_wait * (2**attempt)
                logger.warning(
                    f"Network error ({type(e).__name__}). "
                    f"Retrying in {wait}s... (attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(wait)
            else:
                raise
        except HTTPError as e:
            # Re-raise HTTP errors that aren't retryable
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code not in [429, 500, 502, 503, 504]:
                    raise
                elif attempt == max_attempts - 1:
                    raise
            else:
                raise

    # This should never be reached due to raise statements above, but as a safeguard
    if last_response:
        last_response.raise_for_status()
    raise RequestException("Request failed after all retries")


class DataLoader:
    """Data loader for fetching stock data from Polygon.io API.

    Features:
    - Fetches OHLCV data with configurable periods
    - Fetches dividend data
    - Rate limiting for free tier (5 requests per minute)
    - Exponential backoff retry logic
    - Timezone-aware datetime handling (Eastern time)
    - CSV export with compression for minute data
    - Calendar validation
    - DataFrame alignment checking

    Args:
        api_key: Polygon.io API key. If None, loads from environment variable POLYGON_API_KEY
        base_url: Base URL for Polygon.io API (default: 'https://api.polygon.io')
        rate_limit_requests: Number of requests allowed per minute (default: 5 for free tier)
        rate_limit_window: Time window for rate limiting in seconds (default: 60)

    Raises:
        EnvironmentError: If API key is not found
    """

    # Constants
    VALID_PERIODS = {"minute", "hour", "day", "week", "month", "quarter", "year"}
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW = 60
    DEFAULT_TIMEOUT = (10, 30)

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.polygon.io",
        rate_limit_requests: int = RATE_LIMIT_REQUESTS,
        rate_limit_window: int = RATE_LIMIT_WINDOW,
    ) -> None:
        """Initialize DataLoader with API configuration."""
        # Try loading .env from multiple locations
        import sys

        env_locations = [
            get_project_root() / ".env",  # Project root
            Path.cwd() / ".env",  # Current working directory
            Path(__file__).parent / ".env",  # DataLoader's directory
        ]

        loaded = False
        for env_path in env_locations:
            if env_path.exists():
                dotenv.load_dotenv(env_path)
                logger.debug(f"Loaded .env from: {env_path}")
                loaded = True
                break

        if not loaded:
            logger.warning("No .env file found in standard locations")

        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if self.api_key:
            self.api_key = self.api_key.strip().strip('"').strip("'")

        if not self.api_key:
            raise EnvironmentError(
                "Missing POLYGON_API_KEY. Provide via constructor parameter, "
                "environment variable, or .env file."
            )

        self.base_url = base_url.rstrip("/")
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.eastern_tz = pytz.timezone("America/New_York")

        # Rate limiting state
        self.request_timestamps: List[float] = []

        logger.info(
            f"DataLoader initialized with rate limit: {rate_limit_requests} requests/{rate_limit_window}s"
        )

    @staticmethod
    def set_log_level(level: Union[int, str]) -> None:
        """Set the logging level for the DataLoader module.
        
        Args:
            level: Logging level (e.g., logging.INFO, logging.DEBUG, 'INFO', 'DEBUG')
                  Can be a logging constant or string name.
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
        # Also update handler if it exists
        for handler in logger.handlers:
            handler.setLevel(level)

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting by sleeping if necessary.

        Maintains a sliding window of request timestamps and sleeps
        if the number of requests in the last window exceeds the limit.
        """
        if not self.rate_limit_requests:
            return

        current_time = time.time()

        # Remove timestamps outside the sliding window
        window_start = current_time - self.rate_limit_window
        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > window_start
        ]

        # Check if we've exceeded the rate limit
        if len(self.request_timestamps) >= self.rate_limit_requests:
            oldest_timestamp = self.request_timestamps[0]
            wait_time = (oldest_timestamp + self.rate_limit_window) - current_time

            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

                # Update timestamps after waiting
                current_time = time.time()
                window_start = current_time - self.rate_limit_window
                self.request_timestamps = [
                    ts for ts in self.request_timestamps if ts > window_start
                ]

        # Add current request timestamp
        self.request_timestamps.append(current_time)

    def _add_api_key_to_url(self, url: str) -> str:
        """Add API key to URL, handling existing query parameters.

        Args:
            url: URL to add API key to

        Returns:
            URL with API key parameter added
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        # Add or replace API key
        query_params["apiKey"] = [self.api_key]

        # Reconstruct URL
        new_query = urlencode(query_params, doseq=True)
        return urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                new_query,
                parsed_url.fragment,
            )
        )

    def fetch_ohlcv_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        period: str = "day",
        adjusted: bool = True,
        trading_hours: Optional[Tuple[str, str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch OHLCV stock data from Polygon.io API.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Data aggregation period. One of: 'minute', 'hour', 'day',
                   'week', 'month', 'quarter', 'year'
            adjusted: Whether to return adjusted data for splits and dividends
            trading_hours: Optional tuple of (start_time, end_time) in 'HH:MM' format
                          for filtering minute data. Default is ('09:30', '16:00')
                          for US equities. Set to None to disable filtering.

        Returns:
            Tuple of (DataFrame with columns: volume, open, high, low, close, caldt,
                     metadata dictionary)

        Raises:
            ValueError: If period is not valid or start_date > end_date
            HTTPError: If API request fails
        """
        # Validate inputs
        if period not in self.VALID_PERIODS:
            raise ValueError(
                f"Invalid period '{period}'. Must be one of: {', '.join(sorted(self.VALID_PERIODS))}"
            )

        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            )

        # Set default trading hours for minute data
        if trading_hours is None and period == "minute":
            trading_hours = ("09:30", "16:00")

        start_time = time.time()
        logger.info(
            f"Fetching {ticker} OHLCV data: {start_date} → {end_date} ({period}, "
            f"adjusted={adjusted})"
        )

        # Construct initial URL
        url = (
            f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/{period}/"
            f"{start_date}/{end_date}?adjusted={'true' if adjusted else 'false'}"
            f"&sort=asc&limit=50000"
        )
        url = self._add_api_key_to_url(url)

        data_list: List[Dict[str, Any]] = []

        while url:
            # Enforce rate limit
            self._enforce_rate_limit()

            try:
                response = _retry_request_with_backoff(url)
                data = response.json()

                results = data.get("results", [])
                if not results:
                    logger.debug(f"No data returned for {ticker} in this batch")
                else:
                    entries_count = len(results)
                    logger.debug(f"Fetched {entries_count} entries")

                    for entry in results:
                        # Convert timestamp to timezone-aware datetime
                        utc_time = datetime.fromtimestamp(entry["t"] / 1000, pytz.UTC)
                        eastern_time = utc_time.astimezone(self.eastern_tz)

                        data_entry = {
                            "volume": entry["v"],
                            "open": entry["o"],
                            "high": entry["h"],
                            "low": entry["l"],
                            "close": entry["c"],
                            "caldt": eastern_time,
                        }

                        # Filter by trading hours for minute data
                        if period == "minute" and trading_hours:
                            start_hour, start_min = map(
                                int, trading_hours[0].split(":")
                            )
                            end_hour, end_min = map(int, trading_hours[1].split(":"))
                            trade_start = dt_time(start_hour, start_min)
                            trade_end = dt_time(end_hour, end_min)

                            if trade_start <= eastern_time.time() <= trade_end:
                                data_list.append(data_entry)
                        else:
                            data_list.append(data_entry)

            except HTTPError as e:
                # Extract error message from response
                error_msg = str(e)
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_data = e.response.json()
                        error_msg = error_data.get("error", error_msg)
                    except:
                        pass

                logger.error(f"Failed to fetch {ticker}: {error_msg}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error fetching {ticker}: {e}")
                raise

            # Get next URL for pagination
            url = data.get("next_url")
            if url:
                url = self._add_api_key_to_url(url)

        # Create DataFrame
        if data_list:
            df = pd.DataFrame(data_list)
        else:
            df = pd.DataFrame(
                columns=["volume", "open", "high", "low", "close", "caldt"]
            )

        # Generate metadata
        elapsed_time = time.time() - start_time
        metadata = self._generate_ohlcv_metadata(
            df,
            ticker,
            start_date,
            end_date,
            period,
            adjusted,
            trading_hours,
            elapsed_time,
        )

        # Save to CSV
        filepath = self._save_to_csv(df, ticker, start_date, end_date, period)
        metadata["filepath"] = str(filepath)

        logger.info(
            f"Fetched {len(df):,} entries for {ticker} "
            f"(elapsed: {elapsed_time:.1f}s)"
        )

        return df, metadata

    def fetch_dividend_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch dividend data from Polygon.io API.

        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format
            limit: Maximum number of records to return (default: 1000)

        Returns:
            Tuple of (DataFrame with dividend data, metadata dictionary)

        Raises:
            HTTPError: If API request fails
        """
        start_time = time.time()
        logger.info(f"Fetching dividend data for {ticker}")

        # Build query parameters
        params = {"ticker": ticker, "limit": limit}
        if start_date:
            params["declaration_date.gte"] = start_date
        if end_date:
            params["declaration_date.lte"] = end_date

        url = f"{self.base_url}/v3/reference/dividends"
        url = self._add_api_key_to_url(url)

        # Add params to URL
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        query_params.update({k: [str(v)] for k, v in params.items()})
        new_query = urlencode(query_params, doseq=True)
        url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                new_query,
                parsed_url.fragment,
            )
        )

        # Make request
        self._enforce_rate_limit()
        response = _retry_request_with_backoff(url)
        data = response.json()

        results = data.get("results", [])

        if results:
            df = pd.DataFrame(results)

            # Convert date columns to datetime
            date_columns = [
                "declaration_date",
                "ex_dividend_date",
                "pay_date",
                "record_date",
            ]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Sort by declaration date
            if "declaration_date" in df.columns:
                df = df.sort_values("declaration_date")
        else:
            df = pd.DataFrame()

        # Generate metadata
        elapsed_time = time.time() - start_time
        metadata = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "source": "polygon",
            "data_type": "dividends",
            "records": len(df),
            "fetched_at": datetime.now(self.eastern_tz).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_time, 2),
        }

        # Save to CSV
        if not df.empty:
            date_range = f"{start_date or 'earliest'}_{end_date or 'latest'}"
            filepath = self._save_dividends_to_csv(df, ticker, date_range)
            metadata["filepath"] = str(filepath)

        logger.info(
            f"Fetched {len(df):,} dividend records for {ticker} "
            f"(elapsed: {elapsed_time:.1f}s)"
        )

        return df, metadata

    def _generate_ohlcv_metadata(
        self,
        df: pd.DataFrame,
        ticker: str,
        start_date: str,
        end_date: str,
        period: str,
        adjusted: bool,
        trading_hours: Optional[Tuple[str, str]],
        elapsed_time: float,
    ) -> Dict[str, Any]:
        """Generate metadata dictionary for OHLCV data.

        Args:
            df: DataFrame with fetched data
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            period: Data period
            adjusted: Whether data is adjusted
            trading_hours: Trading hours filter
            elapsed_time: Time taken to fetch data

        Returns:
            Metadata dictionary
        """
        # Calculate date statistics
        if not df.empty:
            unique_dates = df["caldt"].dt.date.nunique()
            date_range = (df["caldt"].min(), df["caldt"].max())
        else:
            unique_dates = 0
            date_range = (None, None)

        # Calculate basic statistics
        numeric_stats = {}
        if not df.empty and len(df) > 1:
            numeric_cols = ["volume", "open", "high", "low", "close"]
            for col in numeric_cols:
                if col in df.columns:
                    numeric_stats[f"{col}_mean"] = float(df[col].mean())
                    numeric_stats[f"{col}_std"] = float(df[col].std())

        metadata = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "period": period,
            "adjusted": adjusted,
            "source": "polygon",
            "data_type": "ohlcv",
            "entries": len(df),
            "trading_days": unique_dates,
            "date_range": date_range,
            "fetched_at": datetime.now(self.eastern_tz).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_time, 2),
            "trading_hours": trading_hours if period == "minute" else None,
            **numeric_stats,
        }

        return metadata

    def _save_to_csv(
        self, df: pd.DataFrame, ticker: str, start_date: str, end_date: str, period: str
    ) -> Path:
        """Save DataFrame to CSV with standardized naming format.

        Uses gzip compression for minute-level data to save space.

        Args:
            df: DataFrame to save
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            period: Period string

        Returns:
            Path to saved file
        """
        data_dir = get_project_root() / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Map period to filename label
        period_map = {
            "minute": "1min",
            "hour": "1hour",
            "day": "1day",
            "week": "1week",
            "month": "1month",
            "quarter": "1quarter",
            "year": "1year",
        }
        period_label = period_map.get(period, period)

        # Create filename
        filename = f"{ticker}_{period_label}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"

        # Save with compression for minute data
        if period == "minute":
            filepath = data_dir / f"{filename}.csv.gz"
            df.to_csv(filepath, index=False, compression="gzip")
        else:
            filepath = data_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)

        relative_path = filepath.relative_to(get_project_root())
        logger.info(f"Saved raw data to {relative_path}")

        return filepath

    def _save_dividends_to_csv(
        self, df: pd.DataFrame, ticker: str, date_range: str
    ) -> Path:
        """Save dividend DataFrame to CSV.

        Args:
            df: Dividend DataFrame
            ticker: Stock ticker symbol
            date_range: Date range string for filename

        Returns:
            Path to saved file
        """
        data_dir = get_project_root() / "data" / "dividends"
        data_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{ticker}_dividends_{date_range}.csv"
        filepath = data_dir / filename
        df.to_csv(filepath, index=False)

        relative_path = filepath.relative_to(get_project_root())
        logger.info(f"Saved dividend data to {relative_path}")

        return filepath

    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from a CSV file.

        Supports both regular CSV and gzipped CSV files.

        Args:
            file_path: Relative path from project root or absolute path

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file cannot be parsed
        """
        # Convert to Path object
        if isinstance(file_path, str):
            if not os.path.isabs(file_path):
                file_path = get_project_root() / file_path
            else:
                file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            relative_path = (
                file_path.relative_to(get_project_root())
                if file_path.is_relative_to(get_project_root())
                else file_path
            )
            raise FileNotFoundError(f"File not found: {relative_path}")

        # Load file based on extension
        try:
            if file_path.suffix == ".gz":
                df = pd.read_csv(file_path, compression="gzip")
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file {file_path}: {e}")

        # Convert 'caldt' column to timezone-aware datetime if it exists
        if "caldt" in df.columns:
            df["caldt"] = pd.to_datetime(df["caldt"])
            # If timezone-naive, assume Eastern time
            if df["caldt"].dt.tz is None:
                df["caldt"] = df["caldt"].dt.tz_localize(
                    self.eastern_tz, ambiguous="NaT"
                )

        relative_path = (
            file_path.relative_to(get_project_root())
            if file_path.is_relative_to(get_project_root())
            else file_path
        )
        logger.info(f"Loaded {len(df):,} rows from {relative_path}")

        return df

    def validate_calendar(
        self, df: pd.DataFrame, frequency: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate calendar coverage of the dataframe.

        For daily data: validates against business days
        For intraday data: provides frequency statistics

        Args:
            df: DataFrame with 'caldt' column
            frequency: Optional frequency hint ('daily', 'minute', 'hour').
                      If None, auto-detects based on time differences.

        Returns:
            Dictionary with coverage statistics
        """
        df = df.copy()

        # Check if DataFrame has data
        if df.empty or "caldt" not in df.columns:
            return {
                "total_entries": 0,
                "unique_dates": 0,
                "frequency": frequency or "unknown",
                "status": "empty_dataframe",
            }

        # Ensure 'caldt' is datetime
        df["caldt"] = pd.to_datetime(df["caldt"], errors="coerce")
        df = df.dropna(subset=["caldt"])

        if df.empty:
            return {
                "total_entries": 0,
                "unique_dates": 0,
                "frequency": frequency or "unknown",
                "status": "invalid_dates",
            }

        # Auto-detect frequency if not provided
        if frequency is None:
            frequency = self._detect_frequency(df["caldt"])

        # Extract dates
        df["date"] = df["caldt"].dt.date
        unique_dates = sorted(df["date"].unique())

        if not unique_dates:
            return {
                "total_entries": 0,
                "unique_dates": 0,
                "frequency": frequency,
                "status": "no_valid_dates",
            }

        min_date = min(unique_dates)
        max_date = max(unique_dates)

        # Calculate statistics based on frequency
        if frequency == "daily":
            # For daily data, validate against business days
            all_business_days = pd.bdate_range(min_date, max_date).date
            missing_days = sorted(set(all_business_days) - set(unique_dates))

            coverage_pct = (
                len(unique_dates) / len(all_business_days) * 100
                if all_business_days.any()
                else 100
            )

            weekend_dates = [d for d in unique_dates if d.weekday() >= 5]

            result = {
                "total_entries": len(df),
                "unique_dates": len(unique_dates),
                "date_range": (min_date, max_date),
                "frequency": frequency,
                "expected_business_days": len(all_business_days),
                "actual_business_days": len(unique_dates) - len(weekend_dates),
                "weekend_days": len(weekend_dates),
                "missing_days_count": len(missing_days),
                "missing_days": missing_days[:20],  # Limit output
                "coverage_percentage": round(coverage_pct, 2),
                "status": "complete" if len(missing_days) == 0 else "incomplete",
            }

        else:
            # For intraday data, provide time-based statistics
            time_diffs = df["caldt"].diff().dropna()

            if len(time_diffs) > 0:
                diff_stats = {
                    "min_interval_seconds": time_diffs.min().total_seconds(),
                    "max_interval_seconds": time_diffs.max().total_seconds(),
                    "median_interval_seconds": time_diffs.median().total_seconds(),
                    "mean_interval_seconds": time_diffs.mean().total_seconds(),
                    "std_interval_seconds": time_diffs.std().total_seconds(),
                }
            else:
                diff_stats = {}

            # Check each day for completeness (for intraday)
            daily_counts = df.groupby("date").size()
            daily_stats = {
                "min_entries_per_day": (
                    int(daily_counts.min()) if not daily_counts.empty else 0
                ),
                "max_entries_per_day": (
                    int(daily_counts.max()) if not daily_counts.empty else 0
                ),
                "mean_entries_per_day": (
                    float(daily_counts.mean()) if not daily_counts.empty else 0
                ),
                "days_with_data": len(daily_counts),
            }

            result = {
                "total_entries": len(df),
                "unique_dates": len(unique_dates),
                "date_range": (min_date, max_date),
                "frequency": frequency,
                "interval_statistics": diff_stats,
                "daily_statistics": daily_stats,
                "status": "intraday_data",
            }

        return result

    def _detect_frequency(self, datetime_series: pd.Series) -> str:
        """Detect frequency of datetime series.

        Args:
            datetime_series: Series of datetime objects

        Returns:
            Detected frequency: 'minute', 'hour', or 'daily'
        """
        if len(datetime_series) < 2:
            return "daily"

        # Calculate median time difference
        diffs = datetime_series.diff().dropna()
        if len(diffs) == 0:
            return "daily"

        median_diff = diffs.median()

        if pd.isna(median_diff):
            return "daily"

        median_seconds = median_diff.total_seconds()

        if median_seconds <= 90:  # 1.5 minutes
            return "minute"
        elif median_seconds <= 3600:  # 1 hour
            return "hour"
        else:
            return "daily"

    @staticmethod
    def check_dataframe_alignment(
        dataframes: List[pd.DataFrame], tolerance: str = "1min"
    ) -> Dict[str, Any]:
        """Check if multiple DataFrames are aligned by their 'caldt' column.

        Args:
            dataframes: List of DataFrames to check
            tolerance: Pandas frequency string for alignment tolerance

        Returns:
            Dictionary with alignment results
        """
        if not dataframes:
            return {"aligned": True, "message": "No dataframes provided", "details": {}}

        # Filter out empty dataframes
        valid_dfs = [
            (i, df)
            for i, df in enumerate(dataframes)
            if not df.empty and "caldt" in df.columns
        ]

        if len(valid_dfs) < 2:
            return {
                "aligned": True,
                "message": "Insufficient valid dataframes for comparison",
                "details": {},
            }

        # Get the first dataframe as reference
        ref_idx, ref_df = valid_dfs[0]
        ref_dates = pd.Series(ref_df["caldt"]).sort_values().reset_index(drop=True)

        misaligned = []
        alignment_details = {}

        for df_idx, df in valid_dfs[1:]:
            # Get dates from current dataframe
            current_dates = pd.Series(df["caldt"]).sort_values().reset_index(drop=True)

            # Check if lengths match
            if len(ref_dates) != len(current_dates):
                misaligned.append(df_idx)
                alignment_details[f"df_{df_idx}_vs_df_{ref_idx}"] = {
                    "status": "misaligned",
                    "reason": f"Length mismatch: {len(ref_dates)} vs {len(current_dates)}",
                    "ref_length": len(ref_dates),
                    "current_length": len(current_dates),
                }
                continue

            # Check if dates are approximately equal (within tolerance)
            try:
                # Convert to numpy datetime64 for comparison
                ref_np = ref_dates.values.astype("datetime64[s]")
                current_np = current_dates.values.astype("datetime64[s]")

                # Calculate maximum difference
                max_diff = np.max(np.abs(ref_np - current_np))
                max_diff_seconds = max_diff.astype("timedelta64[s]").astype(int)

                if max_diff_seconds > pd.Timedelta(tolerance).total_seconds():
                    misaligned.append(df_idx)
                    alignment_details[f"df_{df_idx}_vs_df_{ref_idx}"] = {
                        "status": "misaligned",
                        "reason": f"Timestamps differ by up to {max_diff_seconds} seconds",
                        "max_difference_seconds": max_diff_seconds,
                        "tolerance_seconds": pd.Timedelta(tolerance).total_seconds(),
                    }
                else:
                    alignment_details[f"df_{df_idx}_vs_df_{ref_idx}"] = {
                        "status": "aligned",
                        "max_difference_seconds": max_diff_seconds,
                        "tolerance_seconds": pd.Timedelta(tolerance).total_seconds(),
                    }

            except Exception as e:
                misaligned.append(df_idx)
                alignment_details[f"df_{df_idx}_vs_df_{ref_idx}"] = {
                    "status": "error",
                    "reason": f"Comparison failed: {str(e)}",
                }

        # Overall result
        aligned = len(misaligned) == 0

        result = {
            "aligned": aligned,
            "total_dataframes": len(dataframes),
            "valid_dataframes": len(valid_dfs),
            "misaligned_indices": misaligned,
            "reference_dataframe": ref_idx,
            "tolerance": tolerance,
            "details": alignment_details,
        }

        if aligned:
            result["message"] = "All dataframes are aligned within tolerance"
        else:
            result["message"] = f"Dataframes at indices {misaligned} are misaligned"

        return result

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status.

        Returns:
            Dictionary with rate limit information
        """
        current_time = time.time()
        window_start = current_time - self.rate_limit_window

        # Count requests in current window
        requests_in_window = [ts for ts in self.request_timestamps if ts > window_start]

        return {
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_seconds": self.rate_limit_window,
            "requests_in_current_window": len(requests_in_window),
            "requests_remaining": max(
                0, self.rate_limit_requests - len(requests_in_window)
            ),
            "window_reset_seconds": max(
                0, window_start + self.rate_limit_window - current_time
            ),
            "total_requests_today": len(self.request_timestamps),
        }
