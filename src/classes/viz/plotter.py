
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import statsmodels.api as sm
import os

class Plotter:

  """
  A class whose methods have the aim to reproduce the visualizations shown in the reference paper: 
  "Beat the Market. An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"  by Carlo Zarattini, Andrew Aziz, Andrea Barbon"
  The class also has the aim to calculate some key statistics shown in the reference paper.

  Attributes: 
      dataframe (pandas.DataFrame): the DataFrame containing data with unique dates as indexes, daily returns of the index, daily AUM of the strategy and
      daily returns of the momentum strategy. 

  Methods:

      
  """

def __init__(self, dataframe):

  """
  Initialize the dataframe based on which we are going to visualize the graphs.

  Args: 
      dataframe (pandas.DataFrame): the DataFrame containing data with unique dates as indexes, daily returns of the index and daily AUM of the strategy.
  """

  self.dataframe = dataframe


def plot_cum_returns (self, ret_idx, AUM_strat, file_name, AUM_0 = 100000.0, commission = 0.0035):

    """
    Plot the cumulative returns of the Intraday strategy vs the cumulative returns of a Buy&Hold Strategy.

    Args: 
      dataframe (pandas.DataFrame): the DataFrame containing data with unique dates as indexes, daily returns of the index and daily AUM of the strategy.
      ret_idx (str): name of the column where the daily returns of the index are stored
      AUM_strat (str): name of the column where the daily AUM of the strategy are stored
      AUM_0 (float): Value of Assets Under Management (Equity) at time t = 0. This value must be the same AUM at time 0 used for the calculations
      of the strategy returns (default value is the same as the one in the reference paper).
      commission(float): commission ($) per share (default value is the same as the one in the reference paper)
      file_name (str) = Name with which the plot is saved in outputs/figures
   """
    
    if ret_idx not in self.dataframe.columns:
            raise ValueError(f"Column '{ret_idx}' not found in DataFrame columns.")
    
    if AUM_strat not in self.dataframe.columns:
            raise ValueError(f"Column '{AUM_strat}' not found in DataFrame columns.")
    
    if (AUM_0 != self.dataframe[AUM_strat].iloc[0]):
            raise ValueError(f"AUM at time zero for the two strategies are different.")
    
    
    """Create a copy of the dataframe which can be modified and calculate cumulative products for AUM calculations"""
    df_modified = self.dataframe.copy()
    df_modified['AUM_idx'] = AUM_0 * (1 + df_modified[ret_idx]).cumprod(skipna = True)


    fig, ax = plt.subplots()
    
    ax.plot(df_modified.index, df_modified[AUM_strat], label = 'Momentum', linewidth = 2, color = 'k')
    ax.plot(df_modified.index, df_modified['AUM_idx'], label = 'S&P500', linewidth = 1, color = 'r')

    ax.grid(True, linestyle=':')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.xticks(rotation=90)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_ylabel('AUM ($)')
    plt.legend(loc='upper left')
    plt.title('Intraday Momentum Strategy', fontsize=12, fontweight='bold')
    plt.suptitle(f'Commission = ${commission}/share', fontsize=9, verticalalignment='top')

    #save the plot
    destination_folder = os.path.join("..", "..", "..", "outputs","figures")
    #check id the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    complete_path = os.path.join(destination_folder, file_name)

    try:
            plt.savefig(complete_path, bbox_inches='tight')
            print(f"Plot saved in: {complete_path}")
    except Exception as e:
            print(f"Error in saving the plot: {e}")


    plt.show()
    plt.close()

    


def compute_statistics (self, ret, AUM):
     
     """
     Compute the same statistics used in the reference paper to calculate the returns of the strategy.

     Args:
        ret(str): name of the column in the dataframe where daily returns of the strategy are stored
        AUM(str): name of the column in the dataframe where daily AUM of the strategy is stored 

     Returns:
        pandas.DataFrame: A DataFrame containing the statistics.
     """
     
     if ret not in self.dataframe.columns:
            raise ValueError(f"Column '{ret}' not found in DataFrame columns.")
    
     if AUM not in self.dataframe.columns:
            raise ValueError(f"Column '{AUM}' not found in DataFrame columns.")

     stats = ['Total Return (%)', 'Annualized return (%)', 'Annualized volatility (%)', 'Sharpe Ratio', 'Hit Ratio (%)', 'Maximum Drawdown (%)']
     df_stats = pd.DataFrame(index = ['Value'], columns = stats)

     df_stats['Total Return (%)'] = round((np.prod(1 + self.dataframe[ret].dropna()) - 1) * 100, 0)
     df_stats['Annualized Return (%)'] = round((np.prod(1 + self.dataframe[ret]) ** (252 / len(self.dataframe[ret])) - 1) * 100, 1)
     df_stats['Annualized Volatility (%)'] = round(self.dataframe[ret].dropna().std() * np.sqrt(252) * 100, 1)
     df_stats['Sharpe Ratio'] = round(self.dataframe[ret].dropna().mean() / self.dataframe[ret].dropna().std() * np.sqrt(252), 2)
     df_stats['Hit Ratio (%)'] = round((self.dataframe[ret] > 0).sum() / (self.dataframe[ret].abs() > 0).sum() * 100, 0)
     df_stats['Maximum Drawdown (%)'] = round(self.dataframe[AUM].div(self.dataframe[AUM].cummax()).sub(1).min() * -100, 0)

     return df_stats


def compute_alpha_beta(self, ret_idx, ret_strat):
      
      
      """
      Compute alpha and beta.

      Args:
        ret_idx (str): name of the column where the daily returns of the index are stored
        ret_strat(str): name of the column where the daily returns of the momentum strategy are stored

      Return:
        pandas.DataFrame: A DataFrame containing the statistics.
      """
       
      if ret_idx not in self.dataframe.columns:
            raise ValueError(f"Column '{ret_idx}' not found in DataFrame columns.")
    
      if ret_strat not in self.dataframe.columns:
            raise ValueError(f"Column '{ret_strat}' not found in DataFrame columns.")
      
      stats = ['Alpha (%)', 'Beta']
      df_stats = pd.DataFrame(index = ['Value'], columns = stats)
      
      Y = self.dataframe[ret_strat].dropna()
      X = sm.add_constant(self.dataframe[ret_idx].dropna())

      model = sm.OLS(Y, X).fit()

      df_stats['Alpha (%)'] = round(model.params.const * 100 * 252, 2)
      df_stats['Beta'] = round(model.params[ret_idx], 2)

      return df_stats
