# region imports
from AlgorithmImports import *
from collections import deque
import numpy as np
# endregion

class IntradayMomentum_1(QCAlgorithm):
    def Initialize(self):
        # 1. Setup Basics
        self.SetStartDate(2017, 5, 10) 
        self.SetEndDate(2025, 5, 10)
        self.SetCash(100000)
        
        # 2. Subscribe to Data
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        self.spy_symbol = self.spy.Symbol
        
        # 3. Strategy Parameters
        self.lookback = 14
        self.vol_target = 0.02 # 2% daily target
        self.minute_stats = {} # Stores deques of minute-by-minute moves
        self.daily_returns = deque(maxlen=self.lookback)

        self.entry_check_interval = 30     #Asimmetria delle tempistiche di entry e exit
        self.exit_check_interval = 5
        
        # 4. Indicators & State
        self.vwap = self.VWAP(self.spy_symbol)
        self.todays_open = None
        self.yesterdays_close = None
        
        # 5. Warm up 14 days of Daily data for Volatility
        self.SetWarmUp(self.lookback, Resolution.Daily)
        
        # 6. Scheduled Events
        # Capture the "Yesterday's Close" at the exact market close
        self.Schedule.On(self.DateRules.EveryDay(self.spy_symbol), 
                         self.TimeRules.BeforeMarketClose(self.spy_symbol, 0), 
                         self.RecordEndOfDay)

    def RecordEndOfDay(self):
        """Captures closing price and updates daily volatility deque."""
        if not self.spy.HasData: return
        
        current_close = self.spy.Close
        if self.yesterdays_close is not None:
            daily_ret = (current_close / self.yesterdays_close) - 1
            self.daily_returns.append(daily_ret)
        
        self.yesterdays_close = current_close
        # Reset today's open for the next session
        self.todays_open = None

    def OnData(self, data: Slice):
        if self.spy_symbol not in data or data[self.spy_symbol] is None:
            return

        current_time = self.Time
        current_price = data[self.spy_symbol].Close
        time_key = current_time.strftime("%H:%M")

        # 1. Set Today's Open (9:31 AM is the first usable minute bar)
        if current_time.hour == 9 and current_time.minute == 31:
            self.todays_open = data[self.spy_symbol].Open

        # 2. Safety Check: Need Open, Prev Close, and Volatility Data
        if self.todays_open is None or self.yesterdays_close is None or len(self.daily_returns) < self.lookback:
            return

        # 3. Calculate Boundaries
        historical_moves = self.minute_stats.get(time_key, deque(maxlen=self.lookback))
        sigma = np.mean(historical_moves) if len(historical_moves) > 0 else 0

        # Only trade if we have at least some historical "noise" data
        if sigma == 0 and not self.IsWarmingUp:
            current_move = abs(current_price / self.todays_open - 1)
            historical_moves.append(current_move)
            self.minute_stats[time_key] = historical_moves
            return

        upper_bound = max(self.todays_open, self.yesterdays_close) * (1 + sigma)
        lower_bound = min(self.todays_open, self.yesterdays_close) * (1 - sigma)

        # 4. Update Historical Stats (for tomorrow's sigma)
        current_move = abs(current_price / self.todays_open - 1)
        historical_moves.append(current_move)
        self.minute_stats[time_key] = historical_moves

        # 5. Exit at Market Close
        if current_time.hour == 15 and current_time.minute >= 58:
            if self.Portfolio.Invested:
                self.Liquidate(self.spy_symbol)
            return

        # 6. Read VWAP once
        vwap_val = self.vwap.Current.Value

        # 7. Exit logic checked every self.exit_check_interval minutes
        if self.Portfolio.Invested and current_time.minute % self.exit_check_interval == 0:
            if self.Portfolio[self.spy_symbol].IsLong:
                if current_price < max(upper_bound, vwap_val):
                    self.Liquidate(self.spy_symbol)
                    return
            elif self.Portfolio[self.spy_symbol].IsShort:
                if current_price > min(lower_bound, vwap_val):
                    self.Liquidate(self.spy_symbol)
                    return

        # 8. Entry logic checked every self.entry_check_interval minutes
        if (not self.Portfolio.Invested) and current_time.minute % self.entry_check_interval == 0:
            target_leverage = self.CalculateDynamicSize()
            if target_leverage == 0:
                return

            if current_price > upper_bound:
                self.SetHoldings(self.spy_symbol, target_leverage)
            elif current_price < lower_bound:
                self.SetHoldings(self.spy_symbol, -target_leverage)
    def CalculateDynamicSize(self):
        """Returns the leverage factor based on target volatility."""
        if len(self.daily_returns) < self.lookback: return 0
        
        current_vol = np.std(list(self.daily_returns))
        if current_vol == 0: return 0
        
        # Target Vol / Current Vol = Leverage (capped at 3.8 to leave room for fees)
        leverage = min(2, self.vol_target / current_vol)
        return leverage    