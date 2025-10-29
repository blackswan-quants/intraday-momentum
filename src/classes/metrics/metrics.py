import pandas as pd

class MetricsCalculator :

    
    def from_clean_df (self, df : pd.DataFrame) -> tuple[pd.DataFrame , pd.DataFrame] :
        """
        Standart OHVLC df. 
        Goal here is to compute all the relevant features

        """
        #Initialisation of the two desired data sets
        df['day'] = pd.to_datetime(df['caldt']).dt.date
        df_all_days = df.copy()

        #Apply Helpers
        self.compute_RV(df_all_days)
        self.compute_BV(df_all_days)
        self.compute_vwap(df_all_days)

        #daily_groups
        df_daily_groups = self.compute_intraday_profiles(df_all_days)
        
        return df_all_days , df_daily_groups
    
    #Helpers Functions

    def compute_RV (self, df : pd.DataFrame) -> None :
        pass
    
    def compute_BV (self, df : pd.DataFrame) -> None :
        pass
    
    def compute_vwap (self, df : pd.DataFrame) -> None :
        pass
    
    def compute_intraday_profiles (self , df : pd.DataFrame) -> pd.DataFrame :
        out = df.copy()
        return df
    
    
    