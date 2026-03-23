# region imports
from AlgorithmImports import *
from collections import deque
import numpy as np
# endregion


class IntradayMomentum_4(QCAlgorithm):

    def Initialize(self):
        # 1. Setup Basics
        self.SetStartDate(2025, 2, 10)
        self.SetEndDate(2025, 5, 10)
        self.SetCash(100000)

        # 2. Subscribe to Data
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        self.spy_symbol = self.spy.Symbol

        # 3. Strategy Parameters
        self.lookback = 14
        self.vol_target = 0.02                  # 2% daily target
        self.minute_stats = {}                  # { "HH:MM" : deque of abs moves }
        self.daily_returns = deque(maxlen=self.lookback)

        self.entry_check_interval = 30          # entry every 30 minutes
        self.exit_check_interval = 5            # exit every 5 minutes

        # Exit confirmation counters
        self.exit_confirmation_bars = 4
        self.long_exit_counter = 0
        self.short_exit_counter = 0

        # 4. Indicators & State
        self.vwap = self.VWAP(self.spy_symbol)
        self.ema = self.EMA(self.spy_symbol, 100, Resolution.Minute)
        self.todays_open = None
        self.yesterdays_close = None

        # 5. Warm up 14 days of Daily data for Volatility
        self.SetWarmUp(self.lookback, Resolution.Daily)

        # 6. Scheduled Events
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy_symbol),
            self.TimeRules.BeforeMarketClose(self.spy_symbol, 0),
            self.RecordEndOfDay
        )

    def RecordEndOfDay(self):
        """Captures closing price and updates daily volatility deque."""
        if not self.spy.HasData:
            return

        current_close = self.spy.Close

        if self.yesterdays_close is not None:
            daily_ret = (current_close / self.yesterdays_close) - 1
            self.daily_returns.append(daily_ret)

        self.yesterdays_close = current_close
        self.todays_open = None

        # Reset counters at end of session
        self.long_exit_counter = 0
        self.short_exit_counter = 0

    def OnData(self, data: Slice):
        if self.spy_symbol not in data or data[self.spy_symbol] is None:
            return

        current_time = self.Time
        current_bar = data[self.spy_symbol]
        current_price = current_bar.Close
        time_key = current_time.strftime("%H:%M")

        # 1. Set today's open using the first usable minute bar
        if current_time.hour == 9 and current_time.minute == 31:
            self.todays_open = current_bar.Open

        # 2. Safety check
        if (
            self.todays_open is None
            or self.yesterdays_close is None
            or len(self.daily_returns) < self.lookback
            or not self.ema.IsReady
        ):
            return

        # 3. Calculate intraday sigma for this specific minute-of-day
        historical_moves = self.minute_stats.get(time_key, deque(maxlen=self.lookback))
        sigma = np.mean(historical_moves) if len(historical_moves) > 0 else 0

        if sigma == 0 and not self.IsWarmingUp:
            current_move = abs(current_price / self.todays_open - 1)
            historical_moves.append(current_move)
            self.minute_stats[time_key] = historical_moves
            return

        upper_bound = max(self.todays_open, self.yesterdays_close) * (1 + sigma)
        lower_bound = min(self.todays_open, self.yesterdays_close) * (1 - sigma)

        # 4. Update historical stats for future sessions
        current_move = abs(current_price / self.todays_open - 1)
        historical_moves.append(current_move)
        self.minute_stats[time_key] = historical_moves

        # 5. Forced exit near market close
        if current_time.hour == 15 and current_time.minute >= 58:
            if self.Portfolio.Invested:
                self.Liquidate(self.spy_symbol)
            self.long_exit_counter = 0
            self.short_exit_counter = 0
            return

        # 6. Read VWAP and EMA
        vwap_val = self.vwap.Current.Value
        ema_val = self.ema.Current.Value

        # 7. EXIT LOGIC: checked every 5 minutes with confirmation counter
        if self.Portfolio.Invested and current_time.minute % self.exit_check_interval == 0:

            if self.Portfolio[self.spy_symbol].IsLong:
                exit_level = max(upper_bound, vwap_val)
                if current_price < exit_level:
                    self.long_exit_counter += 1
                else:
                    self.long_exit_counter = 0

                if self.long_exit_counter >= self.exit_confirmation_bars:
                    self.Liquidate(self.spy_symbol)
                    self.long_exit_counter = 0
                    self.short_exit_counter = 0
                    return

            elif self.Portfolio[self.spy_symbol].IsShort:
                exit_level = min(lower_bound, vwap_val)
                if current_price > exit_level:
                    self.short_exit_counter += 1
                else:
                    self.short_exit_counter = 0

                if self.short_exit_counter >= self.exit_confirmation_bars:
                    self.Liquidate(self.spy_symbol)
                    self.long_exit_counter = 0
                    self.short_exit_counter = 0
                    return

        # 8. ENTRY LOGIC: dual confirmation — band breakout AND EMA alignment
        if (not self.Portfolio.Invested) and current_time.minute % self.entry_check_interval == 0:
            target_leverage = self.CalculateDynamicSize()
            if target_leverage == 0:
                return

            if current_price > upper_bound and current_price > ema_val:
                self.SetHoldings(self.spy_symbol, target_leverage)
                self.long_exit_counter = 0
                self.short_exit_counter = 0

            elif current_price < lower_bound and current_price < ema_val:
                self.SetHoldings(self.spy_symbol, -target_leverage)
                self.long_exit_counter = 0
                self.short_exit_counter = 0

    def CalculateDynamicSize(self):
        """Returns the leverage factor based on target volatility."""
        if len(self.daily_returns) < self.lookback:
            return 0

        current_vol = np.std(list(self.daily_returns))
        if current_vol == 0:
            return 0

        leverage = min(2, self.vol_target / current_vol)
        return leverage