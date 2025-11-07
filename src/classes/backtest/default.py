from dataclasses import dataclass

@dataclass(frozen=True)
class BacktestDefaults:
    minute_path: str = "cleaned_df.pkl"
    daily_path: str = "df_and_metrics.pkl"
    initial_aum: float = 100_000.0
    commission_rate: float = 0.0035
    min_comm_per_order: float = 0.35
    slippage_bps: int = 0
    band_mult: float = 1.0
    trade_freq: int = 30
    sizing_type: str = "vol_target"
    target_vol: float = 0.02
    max_leverage: float = 4.0
