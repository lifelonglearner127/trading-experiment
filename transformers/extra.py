import numpy as np
import pandas as pd
import talib


class Transformer:
    def __init__(self, price_column: str = "close"):
        self.price_column = price_column

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        n_bb = 20
        n_kc = 20
        mult_bb = 2.0
        mult_kc = 1.5
        df["Typical_Price"] = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate Bollinger Bands
        df["BB_Mid"] = df["close"].rolling(n_bb).mean()
        df["BB_Std"] = df["close"].rolling(n_bb).std()
        df["BB_Upper"] = df["BB_Mid"] + (mult_bb * df["BB_Std"])
        df["BB_Lower"] = df["BB_Mid"] - (mult_bb * df["BB_Std"])

        # Calculate Keltner Channels
        df["ATR"] = (df["high"] - df["low"]).rolling(n_kc).mean()
        df["KC_Upper"] = df["Typical_Price"].rolling(n_kc).mean() + (mult_kc * df["ATR"])
        df["KC_Lower"] = df["Typical_Price"].rolling(n_kc).mean() - (mult_kc * df["ATR"])

        # Identify squeeze condition (BB inside KC)
        df["Squeeze"] = ((df["BB_Lower"] > df["KC_Lower"]) & (df["BB_Upper"] < df["KC_Upper"])).astype(int)

        # Momentum (difference of moving averages)
        df["Momentum"] = df["close"] - df["close"].rolling(n_bb).mean()

        # Smooth Momentum
        df["Momentum_Smooth"] = df["Momentum"].rolling(4).mean()

        # Histogram (difference of smoothed momentum)
        df["Histogram"] = df["Momentum_Smooth"] - df["Momentum_Smooth"].shift(1)

        # Histogram change (to determine momentum speed)
        df["Histogram_Change"] = df["Histogram"] - df["Histogram"].shift(1)

        # Color based on Histogram and its change
        df["Color"] = np.select(
            condlist=[
                (df["Histogram"] > 0) & (df["Histogram_Change"] > 0),  # Strong upward momentum
                (df["Histogram"] > 0) & (df["Histogram_Change"] <= 0),  # Slowing upward momentum
            ],
            choicelist=["Bright Green", "Light Green"],  # Two shades of bright green
            default="Dark Green",  # Default for negative momentum
        )
        df["ema_200"] = talib.EMA(df["close"], timeperiod=200)
        return df
