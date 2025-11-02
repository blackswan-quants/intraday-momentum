"""
Analyzer module for computing performance statistics, sensitivity analysis, and factor regressions.
"""

from computingstrat import compute_perf_stats
from regression import regressions
from sensitivity import sensitivity_sweep


class Analyzer:
    """
    A class for analyzing trading strategy performance, parameter sensitivity, and factor exposures.
    """
    
    def __init__(self):
        """Initialize the Analyzer class."""
        pass
    
    def compute_perf_stats(self, daily_pnl_df):
        """
        Compute performance metrics from daily P&L data.
        
        Args:
            daily_pnl_df (pd.DataFrame): DataFrame with 'ret', 'ret_spy', 'AUM' columns
        
        Returns:
            dict: Dictionary with performance statistics
        """
        return compute_perf_stats(daily_pnl_df)
    
    def sensitivity_sweep(self, param_grid):
        """
        Perform sensitivity sweeps across parameter combinations.
        
        Args:
            param_grid (dict): Dictionary of parameters with lists of values to test
        
        Returns:
            pd.DataFrame: DataFrame with parameters and performance metrics
        """
        return sensitivity_sweep(param_grid)
    
    def regressions(self, daily_returns, factors):
        """
        Run OLS regressions of strategy returns against market factors.
        
        Args:
            daily_returns (pd.Series): Daily returns of the strategy
            factors (pd.DataFrame): DataFrame with factor returns
        
        Returns:
            dict: Regression results including alpha, betas, p-values, R-squared
        """
        return regressions(daily_returns, factors)
