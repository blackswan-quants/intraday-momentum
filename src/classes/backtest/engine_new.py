from typing import Tuple
from backtesting import Strategy 

"""
Recall main idea of the strat 
If we are outside noise area 
    -> at the first hh.30 we buy / short
    -> we close when either vwap becomes equal to the stock or we are back in the noise area
    -> leverage is set dynamically at the beginning of each day 
"""


class Momentum_Band(Strategy) :

    def init(self):
        self.traded = False
        pass

    def out_of_noise_area(self) -> Tuple[bool, bool]:
        upper_bound = self.data.Upper_bnd[-1]
        lower_bound = self.data.Lower_bnd[-1]
        price=self.data.Open[-1]
        
        if upper_bound < price:
            return(True, True)

        if lower_bound > price :
            return(True, False)
        
        return (False, False)

    def close_position(self) -> bool:
        out_of_bnd, _ = self.out_of_noise_area()
        return not out_of_bnd
    
    def next(self):
        
        current_time = self.data.Minute_of_day[-1]
        price = self.data.Close[-1]
        MARKET_MINUTES = 390  # 9:30-16:00 in minutes from market open   
        EXIT_TIME=MARKET_MINUTES-5 #5 minutes buffer


        # (Daily Reset)
        if current_time == 1:
            self.traded = False
        # Liquidation EOD (End Of Day)
        if current_time >= EXIT_TIME: 
            if self.position:
                self.position.close()
            return 

        # Entry Positions
        if not self.traded and (current_time % 30 == 0) and current_time<300:
            out_of_bnd, is_long = self.out_of_noise_area()
            if out_of_bnd:
                if is_long:
                    self.buy() 
                else:
                    self.sell()
                self.traded = True
                
        # Open Postions (Exit & Stop Loss)
        if self.position:
        # Noise area exit
            out_of_bnd, _ = self.out_of_noise_area()
            if not out_of_bnd:
                 self.position.close()


class Momentum_Band_VWAP(Strategy) :

    def init(self):
        self.traded = False
        pass

    def out_of_noise_area(self) -> Tuple[bool, bool]:
        upper_bound = self.data.Upper_bnd[-1]
        lower_bound = self.data.Lower_bnd[-1]
        price=self.data.Open[-1]
        
        if upper_bound < price:
            return(True, True)

        if lower_bound > price :
            return(True, False)
        
        return (False, False)

    def close_position(self) -> bool:
        out_of_bnd, _ = self.out_of_noise_area()
        return not out_of_bnd
    
    def next(self):
        current_time = self.data.Minute_of_day[-1]
        price = self.data.Close[-1]
        VWAP = self.data.Vwap[-1]
        MARKET_MINUTES = 390  # 9:30-16:00 in minutes from market open   
        EXIT_TIME=MARKET_MINUTES-5 #5 minutes buffer


        # (Daily Reset)
        if current_time == 1:
            self.traded = False
        # Liquidation EOD (End Of Day)
        if current_time >= EXIT_TIME: 
            if self.position:
                self.position.close()
             

        # Entry Positions
        if not self.traded and (current_time % 30 == 0) and current_time<300:
            out_of_bnd, is_long = self.out_of_noise_area()
            if out_of_bnd:
                if is_long:
                    self.buy() 
                else:
                    self.sell()
                self.traded = True
                
        # Open Postions (Exit & Stop Loss)
        if self.position:
            # VWAP Stop Loss 
             # If we are Long and price drops below VWAP -> Immediate Stop Loss
            if self.position.is_long and price < VWAP:
                self.position.close()
                
             # If we are Short and price jumps above VWAP -> Immediate Stop Loss
            elif self.position.is_short and price > VWAP:
                self.position.close()
                
         # Noise area exit
            out_of_bnd, _ = self.out_of_noise_area()
            if not out_of_bnd:
                 self.position.close()
                 


class Momentum_Band_VWAP_Lev(Strategy) :

    def init(self):
        self.traded = False
        self.sig_target = 0.02
        self.max_leverage = 4
        self.lev = 1
        pass

    def out_of_noise_area(self) -> Tuple[bool, bool]:
        upper_bound = self.data.Upper_bnd[-1]
        lower_bound = self.data.Lower_bnd[-1]
        price=self.data.Open[-1]
        
        if upper_bound < price:
            return(True, True)

        if lower_bound > price :
            return(True, False)
        
        return (False, False)

    def close_position(self) -> bool:
        out_of_bnd, _ = self.out_of_noise_area()
        return not out_of_bnd
    
    def set_leverage(self, open) -> float :
        
        return min ( self.max_leverage , self.sig_target / self.data.Sigma[-1])

    
    def next(self):
        current_time = self.data.Minute_of_day[-1]
        price = self.data.Close[-1]
        VWAP = self.data.Vwap[-1]
        MARKET_MINUTES = 390  # 9:30-16:00 in minutes from market open   
        EXIT_TIME=MARKET_MINUTES-5 #5 minutes buffer

        if current_time == 0 :  
            self.lev = self.set_leverage(open= self.data.Open[-1])

        # (Daily Reset)
        if current_time == 1:
            self.traded = False
        # Liquidation EOD (End Of Day)
        if current_time >= EXIT_TIME: 
            if self.position:
                self.position.close()
            return 

        # Entry Positions
        if not self.traded and (current_time % 30 == 0) and current_time<300:
            out_of_bnd, is_long = self.out_of_noise_area()
            if out_of_bnd:
                equity = self.equity
                price = self.data.Close[-1]

                position_value = equity * self.lev
                size = int(position_value / price)

                if is_long:
                    self.buy(size= size) 
                else:
                    self.sell(size = size)
                self.traded = True
                
        # Open Postions (Exit & Stop Loss)
        if self.position:
            # VWAP Stop Loss 
             # If we are Long and price drops below VWAP -> Immediate Stop Loss
            if self.position.is_long and price < VWAP:
                self.position.close()
                return
             # If we are Short and price jumps above VWAP -> Immediate Stop Loss
            elif self.position.is_short and price > VWAP:
                self.position.close()
                return
         # Noise area exit
            out_of_bnd, _ = self.out_of_noise_area()
            if not out_of_bnd:
                 self.position.close()
                 return


       