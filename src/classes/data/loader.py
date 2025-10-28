import requests
import dotenv
import os
import time
from pathlib import Path
from datetime import datetime
import pytz
import pandas as pd


def get_project_root():
    """Get the project root directory by looking for pyproject.toml or .git."""
    current = Path(__file__).parent.parent.parent
    for parent in [current] + list(current.parents):
        if (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
            return parent
    return current


project_root = get_project_root()
dotenv.load_dotenv(project_root / '.env')

API_KEY = os.getenv('POLYGON_API_KEY')
if API_KEY:
    API_KEY = API_KEY.strip().strip('"').strip("'")

if not API_KEY or API_KEY.strip() == '':
    raise ValueError("POLYGON_API_KEY is not set! Please set it in .env file.")

BASE_URL = 'https://api.polygon.io'
ENFORCE_RATE_LIMIT = True


class DataLoader:

    def fetch_polygon_data(self, ticker, start_date, end_date, period, enforce_rate_limit=ENFORCE_RATE_LIMIT):
        """Fetch stock data from Polygon.io API.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: 'minute' or 'day'
            enforce_rate_limit: True for free tier, False for paid tier
        
        Returns:
            DataFrame with columns: volume, open, high, low, close, caldt
        """
        eastern = pytz.timezone('America/New_York')
        url = f'{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/{period}/{start_date}/{end_date}?adjusted=false&sort=asc&limit=50000&apiKey={API_KEY}'
        
        data_list = []
        request_count = 0
        first_request_time = None
        
        while True:
            if enforce_rate_limit and request_count == 5:
                elapsed_time = time.time() - first_request_time
                if elapsed_time < 60:
                    wait_time = 60 - elapsed_time
                    print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                request_count = 0
                first_request_time = time.time()

            if first_request_time is None and enforce_rate_limit:
                first_request_time = time.time()

            response = requests.get(url)
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    print(f"Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"Error: Status code {response.status_code}")
                break

            data = response.json()
            request_count += 1
            print(f"Fetched {len(data.get('results', []))} entries")
            
            if 'results' in data:
                for entry in data['results']:
                    utc_time = datetime.fromtimestamp(entry['t'] / 1000, pytz.utc)
                    eastern_time = utc_time.astimezone(eastern)
                    
                    data_entry = {
                        'volume': entry['v'],
                        'open': entry['o'],
                        'high': entry['h'],
                        'low': entry['l'],
                        'close': entry['c'],
                        'caldt': eastern_time.replace(tzinfo=None)
                    }
                    
                    if period == 'minute':
                        if eastern_time.time() >= datetime.strptime('09:30', '%H:%M').time() and eastern_time.time() <= datetime.strptime('15:59', '%H:%M').time():
                            data_list.append(data_entry)
                    else:
                        data_list.append(data_entry)
            
            if 'next_url' in data and data['next_url']:
                url = data['next_url'] + '&apiKey=' + API_KEY
            else:
                break
        
        df = pd.DataFrame(data_list)
        print(f"Data fetching complete. Total entries: {len(df)}")
        
        self._save_to_csv(df, ticker, start_date, end_date, period)
        return df

    def fetch_yahoo_data(self, ticker, start_date, end_date, period='day'):
        """Fetch stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: 'minute' for intraday or 'day' for daily
        
        Returns:
            DataFrame with columns: volume, open, high, low, close, caldt
        """
        import yfinance as yf
        
        period_map = {'minute': '1m', 'day': '1d'}
        interval = period_map.get(period, '1d')
        
        stock = yf.Ticker(ticker)
        try:
            df = stock.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
        
        if df.empty:
            print(f"No data found for {ticker}")
            return pd.DataFrame()
        
        df = df.reset_index()
        df['caldt'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['Date'])
        df = df[['Volume', 'Open', 'High', 'Low', 'Close', 'caldt']]
        df.columns = ['volume', 'open', 'high', 'low', 'close', 'caldt']
        df = df.dropna()
        
        print(f"Fetched {len(df)} entries from Yahoo Finance")
        self._save_to_csv(df, ticker, start_date, end_date, period)
        return df

    def _save_to_csv(self, df, ticker, start_date, end_date, period):
        """Save DataFrame to CSV with standardized naming format."""
        data_dir = get_project_root() / 'data' / 'raw'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        period_map = {'minute': '1min', 'day': '1day', 'hour': '1hour'}
        period_label = period_map.get(period, period)
        
        filename = f"{ticker}_{period_label}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        filepath = data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")

    def load_csv(self, file_path) -> pd.DataFrame:
        """Load data from a CSV file.
        
        Args:
            file_path: Relative path from project root or absolute path
        
        Returns:
            DataFrame with loaded data
        """
        if not os.path.isabs(file_path):
            file_path = get_project_root() / file_path
        return pd.read_csv(file_path)

    def validate_calendar(self, df) -> dict:
        """Validate calendar coverage of the dataframe.
        
        Args:
            df: DataFrame with 'caldt' column
        
        Returns:
            Dictionary with coverage statistics including:
            - total_days: Actual trading days in data
            - expected_days: Expected business days
            - coverage_percentage: Coverage vs business days
            - missing_dates: List of missing business days
            - weekday_count, weekend_count: Breakdown by day type
        """
        df = df.copy()
        df['caldt'] = pd.to_datetime(df['caldt'], utc=True, errors='coerce')
        df['caldt'] = df['caldt'].dt.tz_localize(None)
        df = df.dropna(subset=['caldt'])
        
        dates = df['caldt'].dt.date.unique()
        total_days = len(dates)
        min_date = min(dates)
        max_date = max(dates)
        
        all_business_dates = pd.bdate_range(min_date, max_date)
        total_expected_business = len(all_business_dates)
        missing_business = sorted(set(all_business_dates.date) - set(dates))
        coverage_pct = (total_days / total_expected_business * 100) if total_expected_business > 0 else 0
        
        weekdays = [d.weekday() for d in dates]
        weekend_count = sum(1 for wd in weekdays if wd >= 5)
        
        return {
            'total_days': total_days,
            'expected_days': total_expected_business,
            'coverage_percentage': round(coverage_pct, 2),
            'unique_dates': len(dates),
            'date_range': (min_date, max_date),
            'missing_dates_count': len(missing_business),
            'missing_dates': missing_business[:50],
            'weekday_count': total_days - weekend_count,
            'weekend_count': weekend_count
        }