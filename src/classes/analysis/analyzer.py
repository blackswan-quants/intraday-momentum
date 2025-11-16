"""
Analyzer module for computing performance statistics, sensitivity analysis.

"""
import logging
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from pathlib import Path
import concurrent.futures
import pickle
import os

# Ensure INFO messages include the actual message text in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Trading calendar constant used for annualization
TRADING_DAYS = 252

# Maximum worker cap to prevent excessive thread/process spawning and resource contention
MAX_WORKERS_CAP = 8


# Core stats computation extracted to module level to avoid circular imports in subprocesses
def _compute_perf_stats_core(daily_pnl_df: pd.DataFrame, trading_days: int) -> dict:
    """Compute key portfolio performance statistics (module-level core).

    This is used both by Analyzer.compute_perf_stats and by the parallel worker
    to avoid importing Analyzer from within this module (which can cause issues
    when unpickling in subprocesses).
    """
    # Input validation
    if daily_pnl_df.empty:
        raise ValueError("daily_pnl_df cannot be empty")
    if 'ret' not in daily_pnl_df.columns:
        raise ValueError("daily_pnl_df must contain 'ret' column")
    
    # Extract returns and drop NaNs
    returns = daily_pnl_df['ret'].ffill().bfill()
    spy_returns = daily_pnl_df['ret_spy'].ffill().bfill() if 'ret_spy' in daily_pnl_df.columns else None

    # Calculate performance metrics
    total_return = (np.prod(1 + returns) - 1) * 100
    annualized_return = (np.prod(1 + returns) ** (trading_days / len(returns)) - 1) * 100
    vol = returns.std()
    annualized_vol = vol * np.sqrt(trading_days) * 100
    hit_ratio = (returns > 0).sum() / len(returns) * 100
    # Compute Sharpe ratio with guard against zero volatility to avoid division error
    if vol < 1e-9:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (returns.mean() / vol) * np.sqrt(trading_days)

    # Benchmark regression (alpha/beta) only if benchmark provided
    alpha = None
    beta = None
    if spy_returns is not None:
        # Guard against degenerate benchmark: skip regression if near-constant
        if spy_returns.std() < 1e-9:
            logging.warning("Benchmark volatility ~0; skipping alpha/beta regression.")
        else:
            # Build a two-column DataFrame and drop rows where either return is missing
            aligned_df = pd.concat([returns, spy_returns], axis=1)
            aligned_df.columns = ['ret', 'ret_spy']
            # Use forward-fill to propagate last known value for missing dates instead of dropping rows
            aligned_df = aligned_df.ffill().bfill()

            if len(aligned_df) > 0:
                Y = aligned_df['ret']
                X = sm.add_constant(aligned_df['ret_spy'])
                model = sm.OLS(Y, X).fit()  # Fit regression (intercept already added by sm.add_constant())

                # model.params['const'] is the daily intercept (in decimal daily returns)
                alpha_daily = float(model.params.get('const', 0.0))
                # Annualize by compounding: (1 + daily_alpha)^trading_days - 1, then convert to percent
                try:
                    alpha = ((1.0 + alpha_daily) ** trading_days - 1.0) * 100.0
                except Exception:
                    # Fallback to linear approximation if something numeric goes wrong
                    logging.warning(
                        "Alpha compounding failed; falling back to linear annualization. "
                        f"alpha_daily={alpha_daily:.6f}. Review regression inputs if this persists."
                    )
                    alpha = alpha_daily * trading_days * 100.0

                # Extract beta safely: prefer 'ret_spy', else first non-const param
                if 'ret_spy' in model.params:
                    beta = float(model.params['ret_spy'])
                else:
                    # Fallback: use first non-const coefficient
                    beta = float(next((v for k, v in model.params.items() if k.lower() != 'const'), np.nan))
            else:
                logging.warning("No overlapping observations for regression; skipping alpha/beta.")

    # Drawdown calculation using AUM when available, else synthetic curve
    if 'AUM' in daily_pnl_df.columns and len(daily_pnl_df) > 0:
        aum_series = daily_pnl_df['AUM']
        # Use the first valid (non-NaN) AUM as baseline; if none, fall back to synthetic curve
        first_valid_idx = aum_series.first_valid_index()
        if first_valid_idx is not None:
            first_aum_raw = aum_series.loc[first_valid_idx]
            try:
                first_aum = float(first_aum_raw)
            except (ValueError, TypeError):
                first_aum = np.nan
            if np.isfinite(first_aum) and first_aum > 0.0:
                cumulative_return = aum_series / first_aum
            else:
                logging.warning(
                    "AUM is present but the first valid value is invalid (<=0 or non-finite). "
                    "Falling back to synthetic cumulative return from 'ret' to compute drawdown."
                )
                if (returns <= -1).any():
                    logging.warning("Returns contain values <= -1, which may cause invalid cumulative returns.")
                cumulative_return = (1 + returns).cumprod()
        else:
            logging.warning(
                "AUM column exists but contains no valid (non-NaN) entries. "
                "Falling back to synthetic cumulative return from 'ret' to compute drawdown."
            )
            if (returns <= -1).any():
                logging.warning("Returns contain values <= -1, which may cause invalid cumulative returns.")
            cumulative_return = (1 + returns).cumprod()
    else:
        # Use synthetic cumulative equity curve from returns when AUM is unavailable
        if (returns <= -1).any():
            logging.warning("Returns contain values <= -1, which may cause invalid cumulative returns.")
        cumulative_return = (1 + returns).cumprod()

    # Check for empty or all-NaN cumulative_return before drawdown calculation
    if len(cumulative_return) == 0 or cumulative_return.isna().all():
        logging.warning("Cumulative return is empty or all NaN; setting max_drawdown to 0")
        max_drawdown = 0.0
    else:
        rolling_max = cumulative_return.expanding().max()
        drawdowns = (cumulative_return - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) * 100.0  # Express as positive percentage

    # Prepare stats dictionary
    stats = {
        'Total Return (%)': round(total_return, 1),
        'Annualized Return (%)': round(annualized_return, 1),
        'Annualized Volatility (%)': round(annualized_vol, 1),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Hit Ratio (%)': round(hit_ratio, 1),
        'Maximum Drawdown (%)': round(max_drawdown, 1)
    }

    if alpha is not None:
        stats['Alpha (%)'] = round(alpha, 2)
    if beta is not None:
        stats['Beta'] = round(beta, 2)

    # Helper for logging annotations (kept for parity with Analyzer logging)
    def get_comment(metric: str, value: float) -> str:
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

    logging.info("Performance Analysis Results")
    logging.info("=" * 50)
    logging.info(f"Total Return: {stats['Total Return (%)']:.1f}%")
    logging.info(f"Annualized Return: {stats['Annualized Return (%)']:.1f}%")
    logging.info(f"Annualized Volatility: {stats['Annualized Volatility (%)']:.1f}%")
    logging.info(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f} ({get_comment('Sharpe Ratio', stats['Sharpe Ratio'])})")
    logging.info(f"Hit Ratio: {stats['Hit Ratio (%)']:.1f}% ({get_comment('Hit Ratio (%)', stats['Hit Ratio (%)'])})")
    logging.info(f"Maximum Drawdown: {stats['Maximum Drawdown (%)']:.1f}%")
    if 'Alpha (%)' in stats:
        logging.info(f"Alpha: {stats['Alpha (%)']:.2f}% (annualized)")
    if 'Beta' in stats:
        logging.info(f"Beta: {stats['Beta']:.2f} ({get_comment('Beta', stats['Beta'])})")
    logging.info("Note: Past performance does not guarantee future results.")

    return stats


# Module-level worker function for ProcessPoolExecutor compatibility
def _run_backtest_combo(args: tuple) -> tuple:
    """Worker function to run a single parameter combination.

    Must be at module level for ProcessPoolExecutor pickling.
    """
    idx, params_tuple, param_names, seed, backtest_fn, trading_days = args
    params = dict(zip(param_names, params_tuple))
    try:
        local_seed = seed + idx
        pnl = backtest_fn(local_seed=local_seed, **params)

        # Compute stats via module-level core function (no imports of Analyzer)
        stats = _compute_perf_stats_core(pnl, trading_days)

        merged = {**params, **stats}
        return (True, merged, None)
    except (ValueError, RuntimeError) as e:
        return (False, None, f"{type(e).__name__}: {e}")
    except Exception as e:
        return (False, None, f"Unexpected {type(e).__name__}: {e}")


class Analyzer:
    """Analyzer for computing performance statistics and parameter sensitivity sweeps.

    Encapsulates configuration (trading_days, output_dir, seed, save, log_every, max_workers)
    and exposes compute_perf_stats and sensitivity_sweep as instance methods.
    """

    def __init__(
        self,
        trading_days: int = TRADING_DAYS,
        output_dir: Path = Path('outputs'),
        seed: int = 42,
        save: bool = True,
        log_every: int = 25,
        max_workers: int | None = None,
    ) -> None:
        self.trading_days = trading_days
        self.output_dir = output_dir
        self.seed = seed
        self.save = save
        self.log_every = log_every
        cpu_count = os.cpu_count() or 1
        self.max_workers = max(1, min(cpu_count, MAX_WORKERS_CAP)) if max_workers is None else max(1, max_workers)

    def compute_perf_stats(self, daily_pnl_df: pd.DataFrame) -> dict:
        """Compute key portfolio performance statistics.

        Required columns:
            ret : daily strategy returns (decimal form)
        Optional columns:
            ret_spy : benchmark daily returns for alpha/beta (decimal)
            AUM : equity curve (used for drawdown if present)

        Metrics returned:
            Total Return (%), Annualized Return (%), Annualized Volatility (%), Sharpe Ratio,
            Hit Ratio (%), Maximum Drawdown (%), plus Alpha (%) and Beta if benchmark provided.

        Drawdown logic:
            Prefer AUM (normalized by first valid positive value). If missing or invalid, fall back to
            synthetic equity curve built from cumulative product of (1 + ret).

        Stability guards:
            - Sharpe set to NaN if volatility < 1e-9.
            - Alpha compounding fallback logs a warning if it fails.

        Returns:
            dict mapping metric name to numeric value.
        
        Raises:
            ValueError: If input validation fails (empty DataFrame or missing 'ret' column).
        """
        return _compute_perf_stats_core(daily_pnl_df, self.trading_days)


    def sensitivity_sweep(self, param_grid: dict, backtest_fn=None, parallel_backend: str = 'auto') -> pd.DataFrame:
        """Run a grid search over parameter combinations and collect performance stats.

        Parameters:
            param_grid: Dict of parameter names to lists of values to sweep.
                Example: {'VM': [10, 20], 'lookback': [20, 40], 'commission': [0.001, 0.002]}
            backtest_fn: Optional callable(local_seed, **params) -> pd.DataFrame.
                If None, uses a built-in synthetic placeholder for testing.
            parallel_backend: 'auto' | 'process' | 'thread'. Backend for parallel execution when
                max_workers > 1.
                - 'auto' (default): Uses threads for the built-in local placeholder backtest, and
                  processes for user-provided functions (with automatic fallback to threads if pickling fails).
                - 'process': Prefer process-based parallelism (may fail if functions are not picklable).
                - 'thread': Force thread-based parallelism (most compatible; subject to GIL for CPU-bound work).

        Returns:
            DataFrame with one row per successful combination (parameters merged with
            compute_perf_stats output). Rows are sorted by Sharpe Ratio when available.

        Validation:
            - param_grid must be a non-empty dict
            - each value must be a non-empty list/tuple

        Parallelism:
            - max_workers is clamped to min(cpu_count, MAX_WORKERS_CAP) to avoid oversubscription.
            - Note: Process-based parallelism requires picklable functions (module-level, no closures/lambdas).
              When using the default local placeholder backtest, threads are used for compatibility.

        Side effects:
            - Results saved to self.output_dir/sensitivity_sweep.csv if self.save=True
            - Progress logged every self.log_every combos
        """
        output_path = self.output_dir / 'sensitivity_sweep.csv'
        
        logging.info(f"max_workers={self.max_workers} (cpu_count={os.cpu_count() or 1}, cap={MAX_WORKERS_CAP})")
        
        # -------------------- Validation --------------------
        if not isinstance(param_grid, dict):
            raise TypeError("param_grid must be a dict of name -> list of values")
        if len(param_grid) == 0:
            logging.warning("param_grid is empty; returning empty DataFrame.")
            return pd.DataFrame()
        for k, v in param_grid.items():
            if not isinstance(v, (list, tuple)) or len(v) == 0:
                raise ValueError(f"param_grid['{k}'] must be a non-empty list or tuple of values")

        # Additional validation: check picklability of parameter values if using process backend
        # Only check if parallel_backend is 'process' (or will resolve to 'process')
        backend = parallel_backend
        if backend == 'auto' and not is_default_backtest:
            backend = 'process'
        if backend == 'process':
            for k, v in param_grid.items():
                for i, item in enumerate(v):
                    try:
                        pickle.dumps(item)
                    except Exception as e:
                        raise ValueError(
                            f"param_grid['{k}'][{i}] = {repr(item)} is not picklable and cannot be used with process-based parallelism. "
                            f"Error: {e}\n"
                            f"Consider using only simple types (int, float, str, etc.) or switching to parallel_backend='thread'."
                        )
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combos = list(itertools.product(*param_values))
        total = len(combos)
        logging.info(f"Starting sensitivity sweep over {total} combinations: {param_names}")

        # -------------------- Backtest function --------------------
        # Capture whether we're using the built-in placeholder before possibly defining it
        is_default_backtest = backtest_fn is None
        
        if backtest_fn is None:
            # Default placeholder for testing.
            # NOTE: This is a nested function and thus unpicklable, so it cannot be used with
            # ProcessPoolExecutor. This is safe because the code ensures the 'thread' backend
            # is used when this default is active (see is_default_backtest and backend selection).
            def backtest_fn(local_seed: int, **params) -> pd.DataFrame:
                """Synthetic placeholder backtest generating random returns."""
                VM = params.get('VM', 10)
                commission = params.get('commission', 0.001)
                rng = np.random.default_rng(local_seed)
                days = self.trading_days
                ret = rng.normal(0.001 * (VM / 20), 0.02 * (VM / 20), days) - commission
                return pd.DataFrame({
                    'ret': ret,
                    'ret_spy': rng.normal(0.001, 0.015, days),
                    'AUM': 100000 * np.cumprod(1 + ret)
                })

        # Validate/resolve parallel backend
        backend_normalized = (parallel_backend or 'auto').strip().lower()
        if backend_normalized not in {'auto', 'process', 'thread'}:
            raise ValueError("parallel_backend must be one of {'auto','process','thread'}")
        if backend_normalized == 'auto':
            # Use threads for local placeholder (not picklable), processes for user-provided functions
            resolved_backend = 'thread' if is_default_backtest else 'process'
        else:
            resolved_backend = backend_normalized  # 'thread' or 'process'

        # -------------------- Prepare worker arguments --------------------
        # Package arguments for module-level worker function
        worker_args = [
            (idx, combo, param_names, self.seed, backtest_fn, self.trading_days)
            for idx, combo in enumerate(combos)
        ]

        # -------------------- Execution (parallel or sequential) --------------------
        results = []
        
        if self.max_workers > 1:
            # Choose executor based on resolved backend
            if resolved_backend == 'thread':
                logging.info("Using ThreadPoolExecutor for parallel execution")
                executor_class = concurrent.futures.ThreadPoolExecutor
            else:
                logging.info("Using ProcessPoolExecutor for parallel execution")
                executor_class = concurrent.futures.ProcessPoolExecutor
            
            try:
                with executor_class(max_workers=self.max_workers) as executor:
                    for i, (success, data, err) in enumerate(executor.map(_run_backtest_combo, worker_args)):
                        if success:
                            results.append(data)
                        else:
                            logging.error(f"Combo failed: {err}")
                        if (i + 1) % self.log_every == 0 or (i + 1) == total:
                            logging.info(f"Progress {i+1}/{total} ({(i+1)/total:.1%})")
            except Exception as e:
                # Fallback to threads for common process-pool failures related to pickling or worker crashes
                is_broken_pool = isinstance(e, concurrent.futures.process.BrokenProcessPool)
                is_pickle_error = isinstance(e, (pickle.PickleError, AttributeError, TypeError))
                
                # Check error message for pickle-related keywords as additional fallback
                msg = str(e).lower()
                msg_has_pickle_hints = "pickle" in msg
                
                # Fallback to threads only when process backend was attempted
                should_fallback = resolved_backend == 'process' and (is_broken_pool or is_pickle_error or msg_has_pickle_hints)
                if should_fallback:
                    logging.warning("ProcessPoolExecutor failed (likely pickling/worker issue); falling back to ThreadPoolExecutor")
                    results = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        for i, (success, data, err) in enumerate(executor.map(_run_backtest_combo, worker_args)):
                            if success:
                                results.append(data)
                            else:
                                logging.error(f"Combo failed: {err}")
                            if (i + 1) % self.log_every == 0 or (i + 1) == total:
                                logging.info(f"Progress {i+1}/{total} ({(i+1)/total:.1%})")
                else:
                    raise
        else:
            # Sequential execution
            for i, args in enumerate(worker_args):
                success, data, err = _run_backtest_combo(args)
                if success:
                    results.append(data)
                else:
                    logging.error(f"Combo failed: {err}")
                if (i + 1) % self.log_every == 0 or (i + 1) == total:
                    logging.info(f"Progress {i+1}/{total} ({(i+1)/total:.1%})")

        # -------------------- Result handling --------------------
        if not results:
            logging.warning("No successful combinations; returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        if 'Sharpe Ratio' in df.columns:
            df = df.sort_values('Sharpe Ratio', ascending=False)
        else:
            logging.warning("'Sharpe Ratio' column missing; results left unsorted.")

        if self.save:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logging.info(f"Saved sensitivity sweep to: {output_path.resolve()}")

        best_sharpe = df['Sharpe Ratio'].max() if 'Sharpe Ratio' in df.columns else None
        logging.info(f"Completed sweep. Total successful: {len(df)}; Best Sharpe: {best_sharpe}")

        return df

# Backward-compatible free functions
def compute_perf_stats(daily_pnl_df: pd.DataFrame) -> dict:
    """Backward-compatible wrapper. Prefer using Analyzer class directly."""
    return _compute_perf_stats_core(daily_pnl_df, TRADING_DAYS)


def sensitivity_sweep(param_grid: dict) -> pd.DataFrame:
    """Backward-compatible wrapper. Prefer using Analyzer class directly."""
    return Analyzer().sensitivity_sweep(param_grid)
