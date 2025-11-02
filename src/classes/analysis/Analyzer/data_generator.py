import pandas as pd
import numpy as np

def create_sample_data(days, initial_aum, ret_mean, ret_std, 
                      spy_mean, spy_std):
    """
    Create sample financial data for testing.
    
    Args:
        days (int): Number of trading days (252 on average in a year)
        initial_aum (float): Initial Assets Under Management
        ret_mean (float): Mean daily return
        ret_std (float): Standard deviation of daily returns
        spy_mean (float): Mean daily SPY return
        spy_std (float): Standard deviation of SPY returns
    
    Returns:
        pd.DataFrame: DataFrame with simulated returns and AUM
    """
    ret = np.random.normal(ret_mean, ret_std, days)
    ret_spy = np.random.normal(spy_mean, spy_std, days)
    AUM = initial_aum * np.cumprod(1 + ret)
    data = {'ret': ret, 'ret_spy': ret_spy, 'AUM': AUM}

    return pd.DataFrame(data)