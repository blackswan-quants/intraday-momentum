# region imports
from AlgorithmImports import *
import numpy as np

# endregion


class IntradayMomentum_0(QCAlgorithm):
    def Initialize(self):
        # 1. Setup Basics
        self.SetStartDate(2017, 5, 10)
        self.SetEndDate(2025, 5, 10)
        self.SetCash(100000)

        # 2. Subscribe to Data
        self.symbol_name = "SPY"
        self.spy = self.AddEquity(self.symbol_name, Resolution.Minute)
        self.spy_symbol = self.spy.Symbol

        # 3. Strategy Parameters
        self.lookback = 14
        self.vol_target = 0.02
        self.max_leverage = 1.5  # Reduced from 2.0 to give more margin headroom

        # Data Storage
        self.daily_returns = RollingWindow[float](self.lookback)
        self.minute_stats = {}  # Dictionary of RollingWindows for each minute

        # State Variables
        self.todays_open = 0.0
        self.yesterdays_close = 0.0
        self.last_day = -1

        # 4. Indicators
        self.vwap = self.VWAP(self.spy_symbol)

        # Warm up for daily volatility
        self.SetWarmUp(self.lookback, Resolution.Daily)

    def OnData(self, data: Slice):
        if self.spy_symbol not in data or data[self.spy_symbol] is None:
            return

        current_time = self.Time
        current_price = data[self.spy_symbol].Close

        # 1. AUTO-DETECT OPEN (First minute of the trading day)
        if current_time.day != self.last_day:
            if self.last_day != -1:
                self.RecordEndOfDay()

            self.todays_open = data[self.spy_symbol].Open
            self.last_day = current_time.day
            self.Debug(f"New Day: {current_time.date()} | Open: {self.todays_open}")
            return

        # 2. Safety Check: Need Open and Yesterday's Close
        if self.todays_open <= 0 or self.yesterdays_close <= 0:
            return

        # 3. Capture Minute Move for Statistics
        time_key = current_time.strftime("%H:%M")
        if time_key not in self.minute_stats:
            self.minute_stats[time_key] = RollingWindow[float](self.lookback)

        current_move = abs(current_price / self.todays_open - 1.0)

        # 4. Strategy Execution (Every 30 Minutes)
        if current_time.minute % 30 == 0:
            self.ExecuteStrategy(current_price, time_key)

        # 5. Always update stats after potential execution
        self.minute_stats[time_key].Add(current_move)

        # 6. Exit near Market Close (3:58 PM ET)
        if current_time.hour == 15 and current_time.minute >= 58:
            if self.Portfolio.Invested:
                self.Liquidate(self.spy_symbol)

    def ExecuteStrategy(self, price, time_key):
        # Calculate Sigma (Mean of historical moves for this specific minute)
        history = self.minute_stats[time_key]
        sigma = np.mean([x for x in history]) if history.Count > 0 else 0.0015

        upper_bound = max(self.todays_open, self.yesterdays_close) * (1.0 + sigma)
        lower_bound = min(self.todays_open, self.yesterdays_close) * (1.0 - sigma)
        vwap_val = self.vwap.Current.Value

        if not self.Portfolio.Invested:
            leverage = self.CalculateDynamicSize()

            if price > upper_bound:
                self.SetHoldings(self.spy_symbol, leverage)
                self.Debug(f"BUY at {price} | Upper {upper_bound:.2f}")
            elif price < lower_bound:
                self.SetHoldings(self.spy_symbol, -leverage)
                self.Debug(f"SELL at {price} | Lower {lower_bound:.2f}")
        else:
            # Exit Logic
            is_long = self.Portfolio[self.spy_symbol].IsLong
            if is_long and price < max(upper_bound, vwap_val):
                self.Liquidate(self.spy_symbol)
            elif not is_long and price > min(lower_bound, vwap_val):
                self.Liquidate(self.spy_symbol)

    def CalculateDynamicSize(self):
        # Conservative default when not enough history
        if not self.daily_returns.IsReady:
            return 0.5

        returns_list = [x for x in self.daily_returns]
        vol = np.std(returns_list)

        if vol == 0:
            return 0.5

        # Calculate target leverage based on vol targeting
        target_leverage = self.vol_target / vol

        # Cap at max_leverage, then apply a 95% margin buffer to avoid hitting limits
        margin_buffer = 0.95
        capped_leverage = min(self.max_leverage, target_leverage) * margin_buffer

        # Dynamic check: if free margin is tight, reduce leverage further
        total_portfolio_value = self.Portfolio.TotalPortfolioValue
        free_margin = self.Portfolio.MarginRemaining

        if total_portfolio_value > 0:
            free_margin_ratio = free_margin / total_portfolio_value
            # If free margin is below 60%, back off aggressively
            if free_margin_ratio < 0.6:
                capped_leverage = min(capped_leverage, free_margin_ratio * 0.9)

        # Always trade at least a minimal size, never go negative
        return max(0.1, capped_leverage)

    def RecordEndOfDay(self):
        # The price at the end of the previous bar
        current_close = self.Securities[self.spy_symbol].Close
        if self.yesterdays_close > 0:
            daily_ret = (current_close / self.yesterdays_close) - 1.0
            self.daily_returns.Add(daily_ret)

        self.yesterdays_close = current_close