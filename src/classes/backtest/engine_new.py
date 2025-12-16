from typing import Tuple
from backtesting import Strategy 


"""
Recall main idea of the strat 
If we are outside noise area 
    -> at the first hh.30 we buy / short
    -> we close when either vwap becomes equal to the stock or we are back in the noise area
    -> leverage is set dynamically at the beginning of each day 
"""

class Momentum(Strategy) :

    def init(self):
        self.traded = False
        pass
    
    def out_of_noise_area(self) -> Tuple[bool, bool]:

        if self.data.Upper_bnd < self.data.Open[-1]:
            out_of_bnd = True
            over = True
            return(out_of_bnd, over)
        
        if self.data.Lower_bnd > self.data.Open[-1] :
            out_of_bnd = True
            over = False
            return(out_of_bnd, over)
        
        return (False, False)
    
    def set_leverage(self) -> float :
        pass
    
    def close_position(self) -> bool:
        out_of_bnd, _ = self.out_of_noise_area()
        return not out_of_bnd
    
    def next(self):

        if self.data.Minute_of_day[-1] == 0 :
            self.traded = False
        
        if not self.traded :
            if self.data.Minute_of_day[-1] % 30 == 0 : 

                Out_of_bnd , Over = self.out_of_noise_area()

                if Out_of_bnd and not self.position:

                    if Over : 
                        self.buy()
                    else :
                        self.sell()
                
                elif self.position and self.close_position() :
                    self.position.close()
                    self.traded = True
        
        elif self.data.Minute_of_day == 389 :
            self.position.close()
        
        pass
