import talib


class Transformer:
    def __init__(self, price_column: str = "close"):
        self.price_column = price_column

    def transform(self, df, **kwargs):
        df["rsi"] = talib.RSI(df[self.price_column], timeperiod=14)

        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df[self.price_column], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower

        macd, macd_signal, macd_hist = talib.MACD(df[self.price_column], fastperiod=12, slowperiod=26, signalperiod=9)
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        return df
