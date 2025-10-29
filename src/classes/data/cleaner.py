from src.classes.utils import io as picklefunction
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import matplotlib.pyplot as plt


    
class DataCleaner:
    def __init__(self,csv_raw):
        try:
            self.df = pd.read_csv(csv_raw)
        except(FileNotFoundError):
            raise FileNotFoundError(f"Error: The file path '{csv_raw}' was not found.")
        except Exception as e:
            raise Exception(f"An error occurred while reading the CSV file: {e}")
        if self.df.empty:
            raise ValueError("Error: The DataFrame is empty after reading the CSV file.")
        
        if 'caldt' not in self.df.columns or 'close' not in self.df.columns:
            print('Warning, time and close columns not found in the dataframe or might have a different name, subsequent methods could fail')
        return
    

    def compute_missing_ratio(self):

        missing_ratio_series = self.df.isnull().mean()
    
        return missing_ratio_series



    def clean(self):
        if 'caldt' not in self.df.columns:
            raise ValueError(f'Time column "caldt" might be missing or have a different name')
        
        self.df['Datetime'] =pd.to_datetime(self.df['caldt'], errors='coerce')

        nat_count = self.df.Datetime.isna().sum()
        if nat_count>0:
            print(f"Warning: {nat_count} invalid date values in 'caldt' were coerced to NaT.")

        self.df = self.df.set_index('Datetime')

        print("Missing Ratios After Indexing:")
        print(self.compute_missing_ratio())
            
        return
    
    def fill_nan(self,column='close'):
        if column not in self.df.columns:
            raise KeyError(f'Error: required column {column} not found in the Dataframe')
        self.df[column] = self.df[column].fillna(method='ffill').fillna(method='bfill')


    def plot(self,column='close'):
        if column not in self.df.columns:
            raise KeyError(f'Error: required column {column} not found in the Dataframe')
        
        plt.hist(self.df[column],density=True, color='red', bins=50, edgecolor='black')
        plt.title(f'Distribuzione di {column}')
        plt.xlabel(column)
        plt.ylabel('Densità di Probabilità')
        plt.grid(axis='y', alpha=0.25) 

        if column == 'close':




        return 
    def deseasonalize(self, period='m', window_days=180, plot=False):

        period_multiplier={'w':1/7, 'd':1, 'h':6.5 , 'm':390}

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError('Dataframe must have a datetime index. Run the clean() method first.')
        if 'close' not in self.df.columns:
            raise KeyError("Error: Required column 'close' not found in the DataFrame.")
        if period not in self.period_multiplier: 
            raise ValueError(f"Invalid period '{period}'. Must be one of: {list(self.period_multiplier.keys())}")
        if not isinstance(window_days, (int, float)) or window_days <= 0:
            raise ValueError("window_days must be a positive number.")
        
        price_series=self.df['close'].copy()

        if price_series.isnull().any():
            price_series=price_series.fillna(method = 'ffill').fillna(method='bfill')


        period_deseason = window_days*period_multiplier[period] 
        decomposition = seasonal_decompose(
            price_series,
            model='multiplicative',
            period = period_deseason
        )
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        if plot==True:
            decomposition.plot()

        price_deseason = trend*residual
        
        self.df['close_deseasonalized'] = price_deseason 

        return 
    
    def save_cleaned(self,path):
        picklefunction(self.df, path)
        return


# TODO : check all the functions, create distirbution functions and better plots