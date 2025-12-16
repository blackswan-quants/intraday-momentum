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

        self.data = self.data

    def out_of_noise_area() :
        pass
    
    def set_leverage() :
        pass
    
    def close_position():
        pass
    
    def next(self):

        if self.outside_of_noise_area(self.data):
            if below : 
                self.position.short()
            else :
                self.position.buy()
        
        if self.close_position(self.data):
            self.position.close()
        
        pass
