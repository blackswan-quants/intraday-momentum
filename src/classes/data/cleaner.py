from src.classes.utils import io as picklefunction
import pandas_market_calendar as mcal
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

class DataCleaner:
    def __init__(self,df):
        self.df = df
        self.df_deseason = pd.DataFrame()
        return


    def clean(self, exchange, start_date, end_date, seasonality=False):
        # capisci come ti arrivano i dati e se devi collegare i giorni
        market_calendar = mcal.get_calendar(exchange)
        trading_days = market_calendar.schedule(start_date=start_date, end_date=end_date)
        # completa successivamente 
        self.df = self.df.set_index('t')
        return

    def compute_missing_ratio(self):
        missing_ratio_series = self.df.isnull().mean()
    
        return missing_ratio_series

    
    def deseasonalize(self, window_days=180, plot=False):

        if not isinstance(self.df.index , pd.DatetimeIndex):
            raise TypeError('Dataframe must have a datetime index')
        
        price_series=self.df['close']

        period_deseason = window_days*390 
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
        
        df_deseason = self.df.copy()
        df_deseason['close_deseasonalized'] = price_deseason 
        self.df_deseason=df_deseason
        return 
    
    def save_cleaned(self,path):
        picklefunction(self.df, path)
        return


# TODO : check all the functions, understand how to deal with time data in input, create distirbution functions 