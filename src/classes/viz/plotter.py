import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import statsmodels.api as sm
import os
from pathlib import Path
import logging

class Plotter:

  """
  A class whose methods have the aim to reproduce the visualizations and compute the same statistics shown in the reference paper: 
  "Beat the Market. An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"  by Carlo Zarattini, Andrew Aziz, Andrea Barbon"

  Attributes
  ----------
  dataframe : pd.DataFrame 
      DataFrame containing data with unique dates as indexes, daily returns of the index, daily AUM of the strategy and
      daily returns of the momentum strategy.     
  """

  #Define the logger
  log_format = '%(levelname)s [%(asctime)s] %(message)s'
  date_format = '%Y-%m-%d %H:%M:%S'

  logging.basicConfig(
     level = logging.INFO,
     format = log_format,
     datefmt = date_format,
     handlers = [logging.StreamHandler()]
  )

  logger = logging.getLogger(__name__)


  def __init__(
      self, 
      dataframe: pd.DataFrame,
      ) -> None:
      """
      Initialize the dataframe based on which we are going to visualize the graphs.

      Parameters
      ---------- 
      dataframe : pd.dataframe
          DataFrame containing data with unique dates as indexes, daily returns of the index and daily AUM of the strategy.
      """
      #check if the index is of type pd.DatetimeIndex
      if not isinstance(dataframe.index, pd.DatetimeIndex):
        Plotter.logger.error("The index of the dataframe is not of type pd.DatetimeIndex. Type: %s", type(dataframe.index).__name__)
        raise ValueError("Index must be of type pd.DatetimeIndex")

      #check if there are any duplicates
      if not dataframe.index.is_unique:
        Plotter.logger.error("Dataframe index contains duplicate dates.")
        raise ValueError("Dataframe index must contain unique dates")

      self.dataframe = dataframe
      Plotter.logger.info("Dataframe correctly initialized")


  def plot_cum_returns (
      self, 
      ret_idx: str, 
      AUM_strat: str, 
      file_name: str, 
      AUM_0: float = 100000.0, 
      commission: float = 0.0035,
      ) -> None:
      """
      Plot the cumulative returns of the Intraday strategy vs. S&P 500.

      Parameters
      ---------- 
      ret_idx : str
          Column name of index returns.
      AUM_strat : str
          Column name of strategy AUM. 
      file_name : str
          Output filename for the figure.
      AUM_0 : float, default = 100_000.0 
          Initial AUM at t = 0. Default value is the same as in the reference paper.
      commission : float, default = 0.0035
          Commission per share. Default value is the same as in the reference paper.
      """
    
      if ret_idx not in self.dataframe.columns:
        raise ValueError(f"Column '{ret_idx}' not found in DataFrame columns.")
    
      if AUM_strat not in self.dataframe.columns:
        raise ValueError(f"Column '{AUM_strat}' not found in DataFrame columns.")
    
      if (AUM_0 != self.dataframe[AUM_strat].iloc[0]):
        raise ValueError(f"AUM at time zero for the two strategies are different.")
    
    
      #Create a copy of the dataframe which can be modified and calculate cumulative products for AUM calculations
      df_modified = self.dataframe.copy()
      Plotter.logger.info('Computing comulative AUM for SPY')
      df_modified['AUM_idx'] = AUM_0 * (1 + df_modified[ret_idx]).fillna(0).cumprod()

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
      out_dir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
      #check if the destination folder exists
      out_dir.mkdir(parents=True, exist_ok=True)
      complete_path = os.path.join(out_dir, file_name)

      try:
        plt.savefig(complete_path, bbox_inches='tight')
        Plotter.logger.info(f"Plot saved to {complete_path}")    
      except (IOError, OSError) as e:
        Plotter.logger.error(f"Failed to save plot to {complete_path}: {e}")
        
      plt.show()
      plt.close()

    


  def compute_statistics (
      self,
      ret: str,
      AUM:str,
      daily_risk_free_rate: float = 0.0,
      ) -> pd.DataFrame:
      """
      Compute the statistics used in the reference paper to calculate the returns.

      Parameters
      ----------
      ret : str 
          Column name of the daily returns of the strategy.
      AUM : str 
          Column name of strategy AUM.
      daily_risk_free_rate : float, default = 0.0
          Daily risk free rate assumed, default value is assumed to be 0.0.
          

      Returns
      -------
      pandas.DataFrame
          A DataFrame containing the statistics.
      """
     
      if ret not in self.dataframe.columns:
        raise ValueError(f"Column '{ret}' not found in DataFrame columns.")
    
      if AUM not in self.dataframe.columns:
        raise ValueError(f"Column '{AUM}' not found in DataFrame columns.")

      stats = ['Total Return (%)', 'Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Hit Ratio (%)', 'Maximum Drawdown (%)']
      df_stats = pd.DataFrame(index = ['Value'], columns = stats)

      Plotter.logger.info('Computing returns and performance statistics such as Sharpe, Hit, Drawdown)')
      #use np.log1p() and exponentiate to avoid potential overflow for long series
      df_stats['Total Return (%)'] = round((np.exp(np.sum(np.log1p(self.dataframe[ret].dropna()))) - 1) * 100, 0)
      df_stats['Annualized Return (%)'] = round((np.exp(np.sum(np.log1p(self.dataframe[ret]))) ** (252 / (self.dataframe[ret]).count()) - 1) * 100, 1)
      df_stats['Annualized Volatility (%)'] = round(self.dataframe[ret].dropna().std() * np.sqrt(252) * 100, 1)
      df_stats['Sharpe Ratio'] = round((self.dataframe[ret].dropna().mean() - daily_risk_free_rate) / self.dataframe[ret].dropna().std() * np.sqrt(252), 2)
      df_stats['Hit Ratio (%)'] = round((self.dataframe[ret] > 0).sum() / (self.dataframe[ret].abs() > 0).sum() * 100, 0)
      df_stats['Maximum Drawdown (%)'] = round(self.dataframe[AUM].div(self.dataframe[AUM].cummax()).sub(1).min() * -100, 0)
      
    
      return df_stats


  def compute_alpha_beta(
      self, 
      ret_idx: str, 
      ret_strat: str,
      ) -> pd.DataFrame:
      """
      Compute alpha and beta.

      Parameters
      ----------
      ret_idx : str 
          Column name of the daily returns of S&P500.
      ret_strat : str 
          Column name  of the daily returns of the strategy.

      Returns
      -------
      pandas.DataFrame
          A DataFrame containing the statistics.
      """
       
      if ret_idx not in self.dataframe.columns:
        raise ValueError(f"Column '{ret_idx}' not found in DataFrame columns.")
    
      if ret_strat not in self.dataframe.columns:
        raise ValueError(f"Column '{ret_strat}' not found in DataFrame columns.")
      
      stats = ['Alpha (%)', 'Beta']
      df_stats = pd.DataFrame(index = ['Value'], columns = stats)
      
      data = self.dataframe[[ret_idx, ret_strat]].dropna()
      Y = data[ret_strat]
      X = sm.add_constant(data[ret_idx])

      Plotter.logger.info('Running OLS Regression for Alpha/Beta estimation')
      model = sm.OLS(Y, X).fit()

      df_stats['Alpha (%)'] = round(model.params.const * 100 * 252, 2)
      df_stats['Beta'] = round(model.params[ret_idx], 2)

      return df_stats



