import requests
from requests.exceptions import HTTPError
import dotenv
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time
from typing import Tuple, Dict, Optional
import pytz
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory by looking for pyproject.toml or .git."""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
            return parent
    return current_file.parent.parent.parent


def _retry_request_with_backoff(url: str, max_attempts: int = 3, base_wait: int = 5) -> requests.Response:
    """Make HTTP request with exponential backoff retry logic.
    
    Args:
        url: URL to request
        max_attempts: Maximum number of retry attempts
        base_wait: Base wait time in seconds for exponential backoff
        
    Returns:
        Response object from successful request
        
    Raises:
        HTTPError: If request fails after all retries
    """
    response = None
    for attempt in range(max_attempts):
        response = requests.get(url)
        
        if response.status_code == 200:
            return response
        elif response.status_code in [429, 500, 502, 503]:
            if attempt < max_attempts - 1:
                wait = base_wait * (2 ** attempt)
                logger.warning(
                    f"Request failed ({response.status_code}). "
                    f"Retrying in {wait}s... (attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(wait)
            else:
                response.raise_for_status()
        else:
            response.raise_for_status()
    
    if response:
        response.raise_for_status()
    raise HTTPError("Request failed after all retries")


class DataLoader:
    """Data loader for fetching stock data from Polygon.io and Yahoo Finance.
    
    Args:
        api_key: Polygon.io API key. If None, loads from environment variable POLYGON_API_KEY
        base_url: Base URL for Polygon.io API (default: 'https://api.polygon.io')
        enforce_rate_limit: Whether to enforce rate limits (default: True for free tier)
    """
    
    VALID_POLYGON_PERIODS = {'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'}
    VALID_YAHOO_PERIODS = {'minute', 'day', 'hour', 'week', 'month'}
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = 'https://api.polygon.io',
        enforce_rate_limit: bool = True
    ):
        """Initialize DataLoader with API configuration."""
        project_root = get_project_root()
        dotenv.load_dotenv(project_root / '.env')
        
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if self.api_key:
            self.api_key = self.api_key.strip().strip('"').strip("'")
        
        if not self.api_key:
            raise EnvironmentError("Missing POLYGON_API_KEY in environment, .env file, or constructor parameter.")
        
        self.base_url = base_url
        self.enforce_rate_limit = enforce_rate_limit

    def fetch_polygon_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        period: str, 
        enforce_rate_limit: Optional[bool] = None,
        trading_hours: Optional[Tuple[str, str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Fetch stock data from Polygon.io API.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: 'minute', 'hour', 'day', 'week', 'month', 'quarter', or 'year'
            enforce_rate_limit: True for free tier, False for paid tier (overrides instance default)
            trading_hours: Optional tuple of (start_time, end_time) in 'HH:MM' format for filtering minute data.
                         Default is ('09:30', '16:00') for US equities. Set to None to disable filtering.
        
        Returns:
            Tuple of (DataFrame with columns: volume, open, high, low, close, caldt, metadata dict)
            
        Raises:
            ValueError: If period is not valid
        """
        if period not in self.VALID_POLYGON_PERIODS:
            raise ValueError(
                f"Invalid period '{period}'. Must be one of: {', '.join(sorted(self.VALID_POLYGON_PERIODS))}"
            )
        
        if enforce_rate_limit is None:
            enforce_rate_limit = self.enforce_rate_limit
        
        if trading_hours is None and period == 'minute':
            trading_hours = ('09:30', '16:00')
        
        start_time = time.time()
        eastern = pytz.timezone('America/New_York')
        url = f'{self.base_url}/v2/aggs/ticker/{ticker}/range/1/{period}/{start_date}/{end_date}?adjusted=false&sort=asc&limit=50000&apiKey={self.api_key}'
        
        logger.info(f"Fetching {ticker} from Polygon: {start_date} → {end_date} ({period})")
        
        data_list = []
        request_count = 0
        first_request_time = None
        
        while True:
            if enforce_rate_limit and request_count == 5:
                elapsed_time = time.time() - first_request_time
                if elapsed_time < 60:
                    wait_time = 60 - elapsed_time
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                request_count = 0
                first_request_time = time.time()

            if first_request_time is None and enforce_rate_limit:
                first_request_time = time.time()

            try:
                response = _retry_request_with_backoff(url)
            except HTTPError as e:
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        error_data = e.response.json()
                        error_msg = error_data.get('error', str(e))
                    else:
                        error_msg = str(e)
                except:
                    error_msg = str(e)
                logger.error(f"Failed to fetch {ticker}: {error_msg}")
                break
            except Exception as e:
                logger.error(f"Unexpected error fetching {ticker}: {e}")
                break

            data = response.json()
            request_count += 1
            
            results = data.get('results', [])
            if not results:
                logger.warning(f"No data returned for {ticker} in this batch.")
                if not data.get('next_url'):
                    break
            else:
                entries_count = len(results)
                logger.info(f"Fetched {entries_count} entries")
                
                for entry in results:
                    utc_time = datetime.fromtimestamp(entry['t'] / 1000, pytz.utc)
                    eastern_time = utc_time.astimezone(eastern)
                    
                    naive_eastern = eastern_time.replace(tzinfo=None)
                    
                    data_entry = {
                        'volume': entry['v'],
                        'open': entry['o'],
                        'high': entry['h'],
                        'low': entry['l'],
                        'close': entry['c'],
                        'caldt': naive_eastern
                    }
                    
                    if period == 'minute' and trading_hours:
                        start_hour, start_min = map(int, trading_hours[0].split(':'))
                        end_hour, end_min = map(int, trading_hours[1].split(':'))
                        trade_start = dt_time(start_hour, start_min)
                        trade_end = dt_time(end_hour, end_min)
                        if trade_start <= eastern_time.time() <= trade_end:
                            data_list.append(data_entry)
                    else:
                        data_list.append(data_entry)
            
            if not data.get('results'):
                break
            if 'next_url' in data and data['next_url']:
                url = data['next_url'] + '&apiKey=' + self.api_key
            else:
                break
        
        df = pd.DataFrame(data_list)
        elapsed_time = time.time() - start_time
        
        if not df.empty:
            unique_dates = df['caldt'].dt.date.nunique()
            logger.info(f"{len(df):,} entries fetched across {unique_dates} trading days (elapsed: {elapsed_time:.2f}s)")
        else:
            logger.warning(f"No data fetched for {ticker}")
            unique_dates = 0
        
        filepath = self._save_to_csv(df, ticker, start_date, end_date, period)
        
        metadata = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "period": period,
            "source": "polygon",
            "entries": len(df),
            "trading_days": unique_dates,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_time, 2),
            "filepath": str(filepath),
            "trading_hours": trading_hours if period == 'minute' else None
        }
        
        return df, metadata

    def fetch_yahoo_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        period: str = 'day'
    ) -> Tuple[pd.DataFrame, Dict]:
        """Fetch stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: 'minute', 'hour', 'day', 'week', or 'month'
        
        Returns:
            Tuple of (DataFrame with columns: volume, open, high, low, close, caldt, metadata dict)
            
        Raises:
            ValueError: If period is not valid
        """
        import yfinance as yf
        
        if period not in self.VALID_YAHOO_PERIODS:
            raise ValueError(
                f"Invalid period '{period}'. Must be one of: {', '.join(sorted(self.VALID_YAHOO_PERIODS))}"
            )
        
        start_time = time.time()
        period_map = {
            'minute': '1m',
            'hour': '1h',
            'day': '1d',
            'week': '1wk',
            'month': '1mo'
        }
        interval = period_map.get(period)
        
        if not interval:
            raise ValueError(f"Period '{period}' not supported by Yahoo Finance. Use one of: {', '.join(sorted(self.VALID_YAHOO_PERIODS))}")
        
        if period == 'minute':
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end_dt - start_dt).days
            if days_diff > 7:
                logger.warning(
                    f"Yahoo intraday (1m) data requires period ≤ 7 days. "
                    f"Requested period: {days_diff} days. Data may be silently truncated."
                )
        
        logger.info(f"Fetching {ticker} from Yahoo Finance: {start_date} → {end_date} ({period})")
        
        stock = yf.Ticker(ticker)
        try:
            df = stock.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
        except Exception as e:
            logger.warning(f"Error fetching {ticker} with Ticker.history(): {e}. Trying yf.download()...")
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
        
        if df.empty:
            logger.warning(f"No data found for {ticker} from Yahoo Finance")
            metadata = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "period": period,
                "source": "yahoo",
                "entries": 0,
                "trading_days": 0,
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(time.time() - start_time, 2),
                "filepath": None
            }
            return pd.DataFrame(), metadata
        
        df = df.reset_index()
        df['caldt'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['Date'])
        df = df[['Volume', 'Open', 'High', 'Low', 'Close', 'caldt']]
        df.columns = ['volume', 'open', 'high', 'low', 'close', 'caldt']
        
        rows_before = len(df)
        df = df.dropna()
        rows_dropped = rows_before - len(df)
        
        if rows_dropped > 0:
            drop_pct = (rows_dropped / rows_before * 100) if rows_before > 0 else 0
            logger.info(f"Dropped {rows_dropped} rows with missing data ({drop_pct:.1f}% of total)")
            if drop_pct < 5 and rows_before > 0:
                logger.info("Consider using forward fill (ffill) or computing averages for small gaps")
        
        elapsed_time = time.time() - start_time
        unique_dates = df['caldt'].dt.date.nunique() if not df.empty else 0
        
        logger.info(f"{len(df):,} entries fetched across {unique_dates} trading days (elapsed: {elapsed_time:.2f}s)")
        
        filepath = self._save_to_csv(df, ticker, start_date, end_date, period)
        
        metadata = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "period": period,
            "source": "yahoo",
            "entries": len(df),
            "trading_days": unique_dates,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_time, 2),
            "filepath": str(filepath) if filepath else None
        }
        
        return df, metadata

    def _save_to_csv(
        self, 
        df: pd.DataFrame, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        period: str
    ) -> Path:
        """Save DataFrame to CSV with standardized naming format.
        
        Uses gzip compression for minute-level data to save space.
        
        Args:
            df: DataFrame to save
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string
            period: Period string ('minute', 'day', etc.)
            
        Returns:
            Path to saved file
        """
        data_dir = get_project_root() / 'data' / 'raw'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        period_map = {'minute': '1min', 'day': '1day', 'hour': '1hour'}
        period_label = period_map.get(period, period)
        
        filename = f"{ticker}_{period_label}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
        
        if period == 'minute':
            filepath = data_dir / f"{filename}.csv.gz"
            df.to_csv(filepath, index=False, compression='gzip')
        else:
            filepath = data_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        
        relative_path = filepath.relative_to(get_project_root())
        logger.info(f"Saved raw data to {relative_path}")
        
        return filepath

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file.
        
        Args:
            file_path: Relative path from project root or absolute path
        
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.isabs(file_path):
            file_path = get_project_root() / file_path
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            relative_path = file_path.relative_to(get_project_root()) if file_path.is_relative_to(get_project_root()) else file_path
            logger.error(f"File not found: {relative_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == '.gz':
            df = pd.read_csv(file_path, compression='gzip')
        else:
            df = pd.read_csv(file_path)
        
        relative_path = file_path.relative_to(get_project_root()) if file_path.is_relative_to(get_project_root()) else file_path
        logger.info(f"Loaded {len(df):,} rows from {relative_path}")
        
        return df

    def validate_calendar(self, df: pd.DataFrame, frequency: Optional[str] = None) -> Dict:
        """Validate calendar coverage of the dataframe.
        
        This method is designed for daily data. For intraday data (minute/hour), 
        it will detect the frequency and skip business day validation if not daily.
        
        Args:
            df: DataFrame with 'caldt' column
            frequency: Optional frequency hint ('daily', 'minute', 'hour'). If None, auto-detects.
        
        Returns:
            Dictionary with coverage statistics including:
            - total_days: Actual trading days in data (for daily) or entries (for intraday)
            - expected_days: Expected business days (only for daily data)
            - coverage_percentage: Coverage vs business days (only for daily data)
            - missing_dates: List of missing business days (only for daily data)
            - weekday_count, weekend_count: Breakdown by day type
            - frequency: Detected frequency of the data
        """
        df = df.copy()
        
        df['caldt'] = pd.to_datetime(df['caldt'], errors='coerce')
        
        if frequency is None:
            try:
                if len(df) > 1:
                    time_diffs = df['caldt'].diff().dropna()
                    if len(time_diffs) > 0:
                        median_diff = time_diffs.median()
                        if pd.notna(median_diff):
                            if median_diff <= pd.Timedelta(minutes=5):
                                frequency = 'minute'
                            elif median_diff <= pd.Timedelta(hours=2):
                                frequency = 'hour'
                            else:
                                frequency = 'daily'
                        else:
                            frequency = 'daily'
                    else:
                        frequency = 'daily'
                else:
                    frequency = 'daily'
            except Exception as e:
                logger.warning(f"Error detecting frequency, defaulting to 'daily': {e}")
                frequency = 'daily'
        
        if frequency is None:
            frequency = 'daily'
        
        df = df.dropna(subset=['caldt'])
        
        if df.empty:
            return {
                'total_days': 0,
                'expected_days': 0,
                'coverage_percentage': 0,
                'unique_dates': 0,
                'date_range': None,
                'missing_dates_count': 0,
                'missing_dates': [],
                'weekday_count': 0,
                'weekend_count': 0,
                'frequency': frequency
            }
        
        try:
            if df['caldt'].dt.tz is not None:
                eastern = pytz.timezone('America/New_York')
                df['caldt'] = df['caldt'].dt.tz_convert(eastern)
            
            df['date'] = df['caldt'].dt.date
            dates = df['date'].unique()
            
            if len(dates) == 0:
                raise ValueError("No valid dates found after processing")
            
            total_days = len(dates)
            min_date = min(dates)
            max_date = max(dates)
        except Exception as e:
            logger.error(f"Error processing dates in validate_calendar: {e}")
            return {
                'total_days': 0,
                'expected_days': 0,
                'coverage_percentage': 0,
                'unique_dates': 0,
                'date_range': None,
                'missing_dates_count': 0,
                'missing_dates': [],
                'weekday_count': 0,
                'weekend_count': 0,
                'frequency': frequency
            }
        
        if frequency == 'daily':
            all_business_dates = pd.bdate_range(min_date, max_date)
            total_expected_business = len(all_business_dates)
            missing_business = sorted(set(all_business_dates.date) - set(dates))
            coverage_pct = (total_days / total_expected_business * 100) if total_expected_business > 0 else 0
        else:
            all_business_dates = pd.bdate_range(min_date, max_date)
            total_expected_business = len(all_business_dates)
            missing_business = []
            coverage_pct = None
            logger.info(
                f"Intraday data detected (frequency: {frequency}). "
                f"Calendar validation compares dates only, not business days."
            )
        
        weekend_dates = [d for d in dates if d.weekday() >= 5]
        weekend_count = len(weekend_dates)
        
        result = {
            'total_days': total_days,
            'expected_days': total_expected_business if frequency == 'daily' else None,
            'coverage_percentage': round(coverage_pct, 2) if coverage_pct is not None else None,
            'unique_dates': len(dates),
            'date_range': (min_date, max_date),
            'missing_dates_count': len(missing_business),
            'missing_dates': missing_business[:50],
            'weekday_count': total_days - weekend_count,
            'weekend_count': weekend_count,
            'frequency': frequency
        }
        
        return result