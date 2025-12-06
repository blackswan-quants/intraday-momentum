from ..utils.io import PickleHelper
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np 
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
    
class DataCleaner:
    """
    A utility class designed to ingest raw financial time-series data from a CSV, 
    perform necessary cleaning operations (like setting a datetime index and filling NaNs),
    and provide methods for time-series analysis and visualization.
    
    The class assumes the raw data contains a time column named 'caldt' and a 
    price column named 'close', though it issues warnings if they are missing..
    """


    def __init__(self, csv_raw: str) -> None:
        """
        Initializes the DataCleaner by reading a CSV file into a pandas DataFrame.
        Handles file errors and checks for empty data or critical columns.

        Args:
            csv_raw (str): The file path to the raw CSV data.
        """
        try:
            self.df = pd.read_csv(csv_raw)
            self.name = csv_raw.split('/')[-1].replace('.csv', '')

        try:
            self.df = pd.read_csv(csv_raw)
        except FileNotFoundError:
            logger.error(f"Error: The file path '{csv_raw}' was not found.")
            raise
        except Exception as e:
            logger.error(f"An error occurred while reading the CSV file: {e}")
            raise
        
        if self.df.empty:
            raise ValueError("Error: The DataFrame is empty after reading the CSV file.")
        
        if 'caldt' not in self.df.columns or 'close' not in self.df.columns:
            logger.warning('Time and close columns not found in the dataframe or might have a different name, subsequent methods could fail.')
    

    def compute_missing_ratio(self) -> pd.Series:
        """
        Computes the ratio of missing (NaN) values for every column in the DataFrame.

        Returns:
            pd.Series: A Series where index is the column name and value is the missing ratio.
        """

        missing_ratio_series = self.df.isnull().mean()
    
        return missing_ratio_series



    def clean(self) -> None:
        """
        Cleans the DataFrame by converting the 'caldt' column to a datetime object,
        handling invalid dates, and setting it as the DataFrame's index.
        """

        if 'caldt' not in self.df.columns:
            raise ValueError('Time column "caldt" might be missing or have a different name')
        
        # Convert to string for processing
        self.df['caldt']=self.df['caldt'].astype(str)
        try:
            # Try parsing with utc=True to handle timezone-aware strings
            self.df['Datetime'] = pd.to_datetime(self.df['caldt'], errors='raise', utc=True)
            
            if self.df['Datetime'].dtype == object:
                raise ValueError("Pandas failed to infer datetime type automatically.")

        except (Exception, ValueError): 
            
            logging.warning("Automatic datetime inference failed. Attempting fallback format: '%Y-%m-%d %H:%M:%S%z'")
            
            try:
                specific_format = '%Y-%m-%d %H:%M:%S%z'
                self.df['Datetime'] = pd.to_datetime(
                    self.df['caldt'], 
                    format=specific_format, 
                    errors='coerce',
                    utc=True
                )
                
            except ValueError as e:
                logging.error(f"Failed to parse dates even with explicit format {specific_format}: {e}")
                # Final fallback - try without format but with utc=True
                self.df['Datetime'] = pd.to_datetime(self.df['caldt'], errors='coerce', utc=True) 

        nat_count: int = self.df['Datetime'].isna().sum()
        if nat_count > 0:
            logger.warning(f"{nat_count} invalid date values in 'caldt' were coerced to NaT.")


        self.df = self.df.set_index('Datetime')
        self.df = self.df.drop(columns=['caldt'])

        
        logger.info("Missing Ratios After Indexing:")
        logger.info(self.compute_missing_ratio().to_string())
    
    def fill_nan(self, column: str = 'close') -> None:
        """
        Fills missing values (NaN) in a specified column using a forward-fill (ffill) 
        followed by a backward-fill (bfill) strategy.

        Args:
            column (str): The name of the column to fill NaNs in. Defaults to 'close'.
        """

        if column not in self.df.columns:
            raise KeyError(f'Error: required column {column} not found in the Dataframe')
        
        self.df[column] = self.df[column].ffill().bfill()
        logger.info(f"NaN values in column '{column}' filled using ffill/bfill strategy.")


    def plot(self) -> None:
        """
        Generates a 3-panel plot for time-series analysis:
        1. Distribution of Intraday Log Returns.
        2. Evolution of Open and Close Prices.
        3. Close Prices highlighted by Intraday Profit/Loss, showing the profit ratio.
        """
        # Ensure required data/index exist before plotting
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError('Dataframe must have a datetime index for plotting. Run the clean() method first.')
        if 'close' not in self.df.columns or 'open' not in self.df.columns:
            raise KeyError("Error: Required columns 'close' and 'open' not found for plotting.")
            
        try:
            
            daily_close_prices = self.df.groupby(self.df.index.date)['close'].last()
            daily_open_prices = self.df.groupby(self.df.index.date)['open'].first()

            
            intraday_log_change = np.log(daily_close_prices.values / daily_open_prices.values) 
            dates = daily_close_prices.index
            
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            plt.style.use('seaborn-v0_8-whitegrid') 

            # =========================================================
            # #### Plot 1: Distribution of Logarithmic Returns ####
            # =========================================================
            ax1 = axes[0]
            
            # Calculates quantiles to limit the X-axis and focus on the central 95%
            q_low = np.quantile(intraday_log_change, 0.025)
            q_high = np.quantile(intraday_log_change, 0.975)
            
            ax1.hist(
                intraday_log_change,
                density=True,
                color='blue',
                edgecolor='black',
                bins=100,
                alpha=0.6
            )
            ax1.set_title(f'1. Distribution of {self.name} Logarithmic Returns')
            ax1.set_xlim(q_low, q_high) 
            ax1.axvline(x=0, color='red', linestyle='--', label='Zero Return')
            ax1.legend()


            # =========================================================
            # #### Plot 2: Open vs Close Price Evolution ####
            # =========================================================
            ax2 = axes[1]
            
            ax2.plot(dates, daily_close_prices.values, color='red', label='Chiusura')
            ax2.plot(dates, daily_open_prices.values, color='blue', label='Apertura', alpha=0.7)
            ax2.set_title('2. Intraday Price Evolution (Open and Daily Close)')
            ax2.set_ylabel('Prezzo')
            ax2.legend(loc='upper left')

            
            # =========================================================
            # #### Plot 3: Close Prices colored by Profit #### 
            # =========================================================
            ax3 = axes[2]
            
            profit_signal = (intraday_log_change > 0)
            colors=np.where(profit_signal, 'green', 'red')
            
            
            
            ax3.scatter(
                dates, 
                daily_close_prices, 
                c=colors, 
                s=10
            )
            
            
            # Calculation and visualization of the Ratio (corrected to percentage)
            ratio = np.sum(profit_signal) / len(profit_signal) * 100 
            
            
            ymin, ymax = ax3.get_ylim()
            xmin, xmax = ax3.get_xlim()
            ax3.text(
                xmin + (xmax - xmin) * 0.02, 
                ymax - (ymax - ymin) * 0.1,  
                f"Profitable Days: {ratio:.2f}%", 
                fontsize=12, 
                color='darkblue',
                fontweight='bold'
            )

            ax3.set_title('3. Close Prices highlighted by Intraday Profit')
            ax3.set_ylabel('Price')
            ax3.legend(loc='lower left')
            
            # =========================================================
            # 3. Finalization
            # =========================================================
            plt.tight_layout() 
            plt.show()
            
        except Exception as e:
            # Catch general errors that might occur during plotting (e.g., NaN issues, math errors)
            logger.error(f"An unexpected error occurred during plotting: {e}")        
        return
    

    def deseasonalize(self, period: str = 'm', window_days: float = 30.0, plot: bool = False) -> None:
        """
        Performs seasonal decomposition on the 'close' price series to extract 
        trend, seasonal, and residual components. The deseasonalized price (trend * residual)
        is stored in a new column 'close_deseasonalized'.

        Args:
            period (str): Base period for decomposition ('w', 'd', 'h', 'm'). 'm' is minutes.
            window_days (int/float): The number of trading days to estimate the seasonal window.
            plot (bool): If True, plots the seasonal decomposition results.
        
        Raises:
            TypeError: If the DataFrame index is not datetime.
            KeyError: If the 'close' column is missing.
            ValueError: If 'period' is invalid or 'window_days' is non-positive.
        """

        period_multiplier={'w':1/7, 'd':1, 'h':6.5 , 'm':390}

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError('Dataframe must have a datetime index. Run the clean() method first.')
        if 'close' not in self.df.columns:
            raise KeyError("Error: Required column 'close' not found in the DataFrame.")
        if period not in period_multiplier: 
            raise ValueError(f"Invalid period '{period}'. Must be one of: {list(period_multiplier.keys())}")
        if not isinstance(window_days, (int, float)) or window_days <= 0:
            raise ValueError("window_days must be a positive number.")
        
        price_series=self.df['close'].copy()

        if price_series.isnull().any():
            price_series=price_series.fillna(method = 'ffill').fillna(method='bfill')
            logger.info("NaN values in 'close' column filled before decomposition.")


        period_deseason = int(window_days*period_multiplier[period])
        decomposition = seasonal_decompose(
            price_series,
            model='multiplicative',
            period = period_deseason
        )

        if plot==True:
            decomposition.plot()
            plt.show()

        price_deseason = price_series / decomposition.seasonal  
        price_deseason = price_deseason.ffill().bfill()
        logger.info(f"Price deseasonalized using period={period_deseason} and NaNs filled.")
        
        self.df['close_deseasonalized'] = price_deseason 

        return 
    
    def save_cleaned(self,filename: str) -> None:
        """
        Saves the current state of the cleaned DataFrame to a specified path 
        (e.g., using a pickle format for efficient storage).

        Args:
            filename (str): The name we want to store the picklefile as.
        """
        try:
            
            helper = PickleHelper(self.df)
            helper.pickle_dump(filename)
            
            # error handled by PickleHelper class
        except Exception as e:
            
            logger.error(f"Failed to save data: {e}") 
            raise
        
