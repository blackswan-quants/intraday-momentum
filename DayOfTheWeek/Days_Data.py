from pathlib import Path

import pandas as pd

# Trade lists live next to this script, so the module runs from any directory
DATA_DIR = Path(__file__).parent

# List of strategy numbers to process
strategies = [0, 1, 2, 3, 4]

# Define the logical order for the days of the week once
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

for strat_id in strategies:
    # 1. Dynamic file loading
    file_path = DATA_DIR / f'Trades_Strat{strat_id}_8y.csv'

    try:
        df = pd.read_csv(file_path)
        
        # --- DAY OF THE WEEK EXTRACTION ---
        df['Entry Time'] = pd.to_datetime(df['Entry Time'])
        df['Day_of_Week'] = df['Entry Time'].dt.day_name()
        df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=days_order, ordered=True)

        # --- CALCULATE GROUPED STATISTICS ---
        stats = df.groupby('Day_of_Week', observed=False).agg(
            Total_Trades=('P&L', 'count'),
            Win_Rate_Pct=('IsWin', lambda x: x.mean() * 100),
            Avg_Profit_per_Trade=('P&L', 'mean'),
            Total_Profit_per_Day=('P&L', 'sum')
        ).reset_index()

        # Formatting for the report
        stats['Win_Rate_Pct'] = stats['Win_Rate_Pct'].round(2).astype(str) + '%'
        stats['Avg_Profit_per_Trade'] = stats['Avg_Profit_per_Trade'].round(2)
        stats['Total_Profit_per_Day'] = stats['Total_Profit_per_Day'].round(2)

        # --- DISPLAY RESULTS ---
        print("\n" + "="*60)
        print(f" 📊 DAY OF THE WEEK RESULTS (STRATEGY {strat_id} - TRADE LIST) ")
        print("="*60)
        print(stats.to_string(index=False))
        
    except FileNotFoundError:
        print(f"\n⚠️ Warning: File {file_path.name} not found. Skipping Strategy {strat_id}.")