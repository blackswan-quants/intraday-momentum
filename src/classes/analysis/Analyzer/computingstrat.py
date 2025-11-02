
import pandas as pd
import numpy as np
import statsmodels.api as sm

def compute_perf_stats(daily_pnl_df):
    """
    Compute performance metrics from daily P&L data and print results clearly.
    
    Args:
        daily_pnl_df: DataFrame with columns like 'ret', 'ret_spy', 'AUM'
            - 'ret': portfolio daily return
            - 'ret_spy': market daily return
            - 'AUM': daily cumulative capital managed by strategy
    
    Returns:
        dict: Dictionary with performance statistics
    """
 

    # Extract returns and drop NaNs
    returns = daily_pnl_df['ret'].dropna()
    spy_returns = daily_pnl_df['ret_spy'].dropna() if 'ret_spy' in daily_pnl_df.columns else None
    
    # Calculate performance metrics
    total_return = (np.prod(1 + returns) - 1) * 100
    annualized_return = (np.prod(1 + returns) ** (252 / len(returns)) - 1) * 100
    annualized_vol = returns.std() * np.sqrt(252) * 100
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    hit_ratio = (returns > 0).sum() / (returns.abs() > 0).sum() * 100
    
    # Calculate max drawdown
    if 'AUM' in daily_pnl_df.columns:
        cumulative_return = daily_pnl_df['AUM'] / daily_pnl_df['AUM'].iloc[0]
    else:
        cumulative_return = (1 + returns).cumprod()
    rolling_max = cumulative_return.expanding().max()
    drawdowns = (cumulative_return - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * -100
    
    # Calculate alpha and beta if possible
    alpha = None
    beta = None
    if spy_returns is not None and len(spy_returns) == len(returns):
        Y = returns
        X = sm.add_constant(spy_returns)
        model = sm.OLS(Y, X).fit()
        alpha = model.params['const'] * 100 * 252
        beta = model.params['ret_spy']
    
    # Prepare stats dictionary
    stats = {
        'Total Return (%)': round(total_return, 1),
        'Annualized Return (%)': round(annualized_return, 1),
        'Annualized Volatility (%)': round(annualized_vol, 1),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Hit Ratio (%)': round(hit_ratio, 1),
        'Maximum Drawdown (%)': round(max_drawdown, 1)
    }
    
    if alpha is not None and beta is not None:
        stats['Alpha (%)'] = round(alpha, 2)
        stats['Beta'] = round(beta, 2)
    
    # ******Print results*********

       # Helper function to add comments on metrics
    def get_comment(metric, value):
        if metric == 'Sharpe Ratio':
            if value > 2: return "excellent!"
            if value > 1: return "good"
            if value > 0: return "positive"
            return "needs improvement"
        elif metric == 'Beta':
            if abs(value) < 0.3: return "low market correlation"
            if abs(value) < 0.7: return "moderate market correlation"
            return "high market correlation"
        elif metric == 'Hit Ratio (%)':
            if value > 55: return "good win rate"
            if value > 50: return "positive win rate"
            return "needs improvement"
        return ""
    
    print("\nPerformance Analysis Results")
    print("=" * 50)
    
    print(f"Total Return: {stats['Total Return (%)']:.1f}%")
    print(f"Annualized Return: {stats['Annualized Return (%)']:.1f}%")
    print(f"Annualized Volatility: {stats['Annualized Volatility (%)']:.1f}%")
    
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f} ({get_comment('Sharpe Ratio', stats['Sharpe Ratio'])})")
    
    print(f"Hit Ratio: {stats['Hit Ratio (%)']:.1f}% ({get_comment('Hit Ratio (%)', stats['Hit Ratio (%)'])})")
    
    print(f"Maximum Drawdown: {stats['Maximum Drawdown (%)']:.1f}%")
    
    if 'Alpha (%)' in stats:
        print(f"Alpha: {stats['Alpha (%)']:.2f}% (annualized)")
    
    if 'Beta' in stats:
        print(f"Beta: {stats['Beta']:.2f} ({get_comment('Beta', stats['Beta'])})")
    
    print("\nNote: Past performance does not guarantee future results.")
    
    return stats


    
