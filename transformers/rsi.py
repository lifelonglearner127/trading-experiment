import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


class Transformer:
    def __init__(self, price_column: str = "close"):
        self.price_column = price_column

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        delta = df[self.price_column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        windows = [5, 10, 20, 30]
        for window in windows:
            gain_ma = gain.rolling(window=window).mean()
            loss_ma = loss.rolling(window=window).mean()
            rs = gain_ma / loss_ma
            rsi = 100 - (100 / (1 + rs))

            df[f"rsi_pattern_skew_{window}"] = rsi.rolling(window=window).apply(skew)
            df[f"rsi_pattern_kurtosis_{window}"] = rsi.rolling(window=window).apply(kurtosis)
            df[f"rsi_volatility_{window}"] = rsi.rolling(window=window).std()
            df[f"rsi_trend_{window}"] = (rsi - rsi.shift(window)) / rsi.shift(window)

        for window in windows:
            rsi_series = df[f"rsi_pattern_skew_{window}"]
            df[f"rsi_extreme_ratio_{window}"] = rsi_series.rolling(window=window).apply(
                lambda x: np.sum(np.abs(x - x.mean()) > 2 * x.std()) / len(x)
            )

            df[f"rsi_inflection_{window}"] = (
                ((rsi_series > rsi_series.shift(1)) & (rsi_series > rsi_series.shift(-1)))
                | ((rsi_series < rsi_series.shift(1)) & (rsi_series < rsi_series.shift(-1)))
            ).astype(int)

        for window in windows:
            df[f"rsi_relative_position_{window}"] = (
                df[f"rsi_pattern_skew_{window}"] - df[f"rsi_pattern_skew_{window}"].rolling(window=window).min()
            ) / (
                df[f"rsi_pattern_skew_{window}"].rolling(window=window).max()
                - df[f"rsi_pattern_skew_{window}"].rolling(window=window).min()
            )

            df[f"rsi_momentum_acc_{window}"] = df[f"rsi_trend_{window}"].diff() / df[f"rsi_trend_{window}"].shift(1)

        for fast_window, slow_window in zip(windows[:-1], windows[1:]):
            df[f"rsi_crossover_{fast_window}_{slow_window}"] = (
                (df[f"rsi_pattern_skew_{fast_window}"] > df[f"rsi_pattern_skew_{slow_window}"])
                & (df[f"rsi_pattern_skew_{fast_window}"].shift(1) <= df[f"rsi_pattern_skew_{slow_window}"].shift(1))
            ).astype(int)

        return df
