from typing import Tuple
from backtesting import Backtest , Strategy 


"""
Recall main idea of the strat 
If we are outside noise area 
    -> at the first hh.30 we buy / short
    -> we close when either vwap becomes equal to the stock or we are back in the noise area
    -> leverage is set dynamically at the beginning of each day 
"""

class Momentum(Strategy) :

    def __init__(self):
        vol_target = 4
        max_leverage = 2
        pass
    
    def out_of_noise_area(self) -> Tuple[bool, bool]:

        if self.data.upper_bnd < self.data.open :
            out_of_bnd = True
            over = True
            return(out_of_bnd, over)
        
        if self.data.lower_bnd > self.data.open :
            out_of_bnd = True
            over = False
            return(out_of_bnd, over)
        
        return (False, False)
    
    def set_leverage(self) -> float :


        pass
    
    def close_position(self) -> bool:
        pass
    
    def next(self):
        Out_of_bnd , over = self.outside_of_noise_area(self.data)

        if Out_of_bnd and self.data.index.date.minutes == 30:
            if not over : 
                self.position.short()
            else :
                self.position.buy()
        
        if self.poistion() and self.close_position(self.data):
            self.position.close()
        
        pass
