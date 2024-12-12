import numpy as np
import pandas as pd
from scipy.stats import linregress


class Transformer:
    def __init__(self, price_high="high", price_low="low", price_close="close", volume="volume"):
        self.price_high = price_high
        self.price_low = price_low
        self.price_close = price_close
        self.volume = volume

    def calculate_mfi(self, df, periods=(14, 28, 56)):
        typical_price = (df[self.price_high] + df[self.price_low] + df[self.price_close]) / 3
        money_flow = typical_price * df[self.volume]

        for period in periods:
            positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0))
            negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0))

            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()

            mfi = 100 - (100 / (1 + positive_mf / negative_mf))

            df[f"mfi_{period}"] = mfi
            df[f"mfi_ma_{period}"] = mfi.rolling(window=period).mean()
            df[f"mfi_slope_{period}"] = self.calculate_slope(mfi, period)
            df[f"mfi_divergence_{period}"] = self.calculate_divergence(df[self.price_close], mfi, period)

    def calculate_vwap(self, df, windows=(5, 10, 20, 30)):
        typical_price = (df[self.price_high] + df[self.price_low] + df[self.price_close]) / 3

        for window in windows:
            vwap = (typical_price * df[self.volume]).rolling(window).sum() / df[self.volume].rolling(window).sum()

            df[f"vwap_{window}"] = vwap
            df[f"price_to_vwap_{window}"] = df[self.price_close] / vwap
            df[f"vwap_slope_{window}"] = self.calculate_slope(vwap, window)
            df[f"vwap_volatility_{window}"] = self.calculate_volatility(vwap, window)

    def calculate_ad_line(self, df, windows=(5, 10, 20, 30)):
        clv = ((df[self.price_close] - df[self.price_low]) - (df[self.price_high] - df[self.price_close])) / (
            df[self.price_high] - df[self.price_low]
        )
        ad = (clv * df[self.volume]).cumsum()

        for window in windows:
            df[f"ad_line_{window}"] = ad.rolling(window=window).mean()
            df[f"ad_momentum_{window}"] = self.calculate_momentum(ad, window)
            df[f"ad_trend_{window}"] = self.calculate_trend_strength(ad, window)
            df[f"ad_divergence_{window}"] = self.calculate_divergence(df[self.price_close], ad, window)

    def calculate_klinger(self, df, short_period=34, long_period=55, signal_period=13):
        trend = np.where(df[self.price_close] > df[self.price_close].shift(1), 1, -1)
        dm = df[self.price_high] - df[self.price_low]

        sv = df[self.volume] * abs(2 * (dm / dm.shift(1) - 1))
        sv = sv * trend

        ema34 = sv.ewm(span=short_period).mean()
        ema55 = sv.ewm(span=long_period).mean()
        kvo = ema34 - ema55

        df["klinger"] = kvo
        df["klinger_signal"] = kvo.ewm(span=signal_period).mean()
        df["klinger_hist"] = df["klinger"] - df["klinger_signal"]
        df["klinger_slope"] = self.calculate_slope(kvo, signal_period)
        df["klinger_divergence"] = self.calculate_divergence(df[self.price_close], kvo, signal_period)

    @staticmethod
    def calculate_slope(series, period):
        return series.diff(period) / period

    @staticmethod
    def calculate_momentum(series, period):
        return series.diff(period)

    @staticmethod
    def calculate_volatility(series, period):
        return series.rolling(window=period).std() / series.rolling(window=period).mean()

    @staticmethod
    def calculate_trend_strength(series, period):
        def linear_regression_slope(x):
            if len(x) < 2:
                return np.nan
            return linregress(np.arange(len(x)), x)[0]

        return series.rolling(period).apply(linear_regression_slope)

    @staticmethod
    def calculate_divergence(price, indicator, period):
        price_trend = price.rolling(period).apply(lambda x: linregress(np.arange(len(x)), x)[0])
        indicator_trend = indicator.rolling(period).apply(lambda x: linregress(np.arange(len(x)), x)[0])
        return np.where(price_trend * indicator_trend < 0, 1, 0)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.calculate_mfi(df)
        self.calculate_vwap(df)
        self.calculate_ad_line(df)
        self.calculate_klinger(df)

        windows = [5, 10, 20, 30]
        for w in windows:
            if f"mfi_{w}" in df.columns and f"ad_line_{w}" in df.columns:
                df[f"mfi_ad_ratio_{w}"] = df[f"mfi_{w}"] / df[f"ad_line_{w}"]

            if f"vwap_{w}" in df.columns:
                df[f"klinger_vwap_cross_{w}"] = np.where(
                    (df["klinger"].shift(1) < df[f"vwap_{w}"].shift(1)) & (df["klinger"] > df[f"vwap_{w}"]),
                    1,
                    np.where(
                        (df["klinger"].shift(1) > df[f"vwap_{w}"].shift(1)) & (df["klinger"] < df[f"vwap_{w}"]), -1, 0
                    ),
                )

        return df
