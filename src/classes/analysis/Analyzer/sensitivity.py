import pandas as pd
import numpy as np
import itertools
from computingstrat import compute_perf_stats

def sensitivity_sweep(param_grid) -> pd.DataFrame:
    """
    Perform sensitivity sweeps testing combinations of VM, lookback, commission.
    
    Args:
        param_grid (dict): e.g. {'VM': [10,20], 'lookback': [20,40], 'commission':[0.001,0.002]}
    
    Returns:
        pd.DataFrame: each row contains parameters + performance stats
    """
    results = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Sample backtest function (replace with your real backtest)
    def backtest(VM, lookback, commission):
        days = 252
        ret = np.random.normal(0.001 * (VM/20), 0.02 * (VM/20), days) - commission
        return pd.DataFrame({
            'ret': ret,
            'ret_spy': np.random.normal(0.001, 0.015, days),
            'AUM': 100000 * np.cumprod(1 + ret)
        })
    
    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        try:
            daily_pnl = backtest(**params)
            stats = compute_perf_stats(daily_pnl)
            results.append({**params, **stats})
            print(f"Tested {params}")
        except Exception as e:
            print(f"Error with {params}: {e}")
    
    df = pd.DataFrame(results).sort_values('Sharpe Ratio', ascending=False)
    # Salvataggio nel percorso desiderato
    df.to_csv('/intraday-momentum-spy-repro/outputs/sensitivity_sweep.csv', index=False)
    print(f"\nTotal tests: {len(df)}; Best Sharpe: {df['Sharpe Ratio'].max():.2f}")
    return df