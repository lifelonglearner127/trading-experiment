import pandas as pd
import talib


class Transformer:
    def __init__(self, price_column: str = "close"):
        self.price_column = price_column

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["ema_200"] = talib.EMA(df["close"], timeperiod=200)
        return df
