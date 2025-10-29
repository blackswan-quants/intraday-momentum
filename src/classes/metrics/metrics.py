import pandas as pd
import numpy as np

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

        #Drop unecessary columns 
        df_all_days.drop(columns=["log_returns" , "price"], inplace= True)

        #daily_groups
        df_daily_groups = self.compute_intraday_profiles(df_all_days)
    
        return df_all_days , df_daily_groups
    
    #Helpers Functions

    def compute_RV (self, df : pd.DataFrame) -> None :
        #Compute log_returns 
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1) )
        #Apply RV formula
        rv = df.groupby("day")["log_returns"].apply(
            lambda x: np.sqrt(np.sum(x.dropna() ** 2))
            )
        # Map back to each row
        df["RV"] = df["day"].map(rv, na_action="ignore")
    
    def compute_BV (self, df : pd.DataFrame) -> None :
        #Log returns are already computed
        #Compute Bipower varation
        bv = df.groupby("day")["log_returns"].apply(
            lambda x : (np.pi / 2) * (1 / (len(x.dropna()) - 1)) * 
            np.sum(np.abs(x.dropna()) * np.abs(x.dropna().shift(1)))
            )
        # Map back to each row
        df["BV"] = df["day"].map(bv, na_action="ignore")
    
    def compute_vwap (self, df : pd.DataFrame) -> None :
        #As in the notebook can aslo use close tho
        df["price"] = (df["high"] + df["close"] + df["low"])/3
        #Compute vwap
        vwap = df.groupby("day")[["price" , "volume"]].apply(
            lambda x : (np.sum(x["volume"] * x["price"] )) / np.sum(x["volume"])
        )
        # Map back to each row
        df["vwap"] = df["day"].map(vwap , na_action= "ignore")
    
    def compute_intraday_profiles (self , df : pd.DataFrame) -> pd.DataFrame :
    
        out = df.groupby("minute_of_day").mean(numeric_only=True)
        return out
