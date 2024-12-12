import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


class Transformer:
    def __init__(self, hist_column: str = "macd_hist"):
        self.hist_column = hist_column
        self.windows = (5, 10, 20)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for window in self.windows:
            df[f"hist_ma_{window}"] = df[self.hist_column].rolling(window).mean()
            df[f"hist_std_{window}"] = df[self.hist_column].rolling(window).std()
            df[f"hist_skew_{window}"] = df[self.hist_column].rolling(window).apply(skew)
            df[f"hist_kurt_{window}"] = df[self.hist_column].rolling(window).apply(kurtosis)

        df["hist_momentum"] = df[self.hist_column].diff()
        df["hist_acceleration"] = df["hist_momentum"].diff()
        df["hist_roc"] = df[self.hist_column].pct_change()
        df["hist_zero_cross"] = np.where(
            (df[self.hist_column] > 0) & (df[self.hist_column].shift(1) < 0)
            | (df[self.hist_column] < 0) & (df[self.hist_column].shift(1) > 0),
            1,
            0,
        )

        for window in self.windows:
            df[f"hist_pos_sum_{window}"] = (
                df[self.hist_column].apply(lambda x: x if x > 0 else 0).rolling(window).sum()
            )

            df[f"hist_neg_sum_{window}"] = (
                df[self.hist_column].apply(lambda x: abs(x) if x < 0 else 0).rolling(window).sum()
            )

        df["hist_consec_pos"] = (
            (df[self.hist_column] > 0).astype(int).groupby((df[self.hist_column] <= 0).astype(int).cumsum()).cumsum()
        )

        df["hist_consec_neg"] = (
            (df[self.hist_column] < 0).astype(int).groupby((df[self.hist_column] >= 0).astype(int).cumsum()).cumsum()
        )

        for window in self.windows:
            df[f"hist_oscillation_{window}"] = (
                df[self.hist_column].rolling(window).max() - df[self.hist_column].rolling(window).min()
            )

        for window in self.windows:
            pos_counts = (df[self.hist_column] > 0).rolling(window).sum()
            total_counts = window
            df[f"hist_pos_ratio_{window}"] = pos_counts / total_counts

        df["hist_reversal"] = np.where(
            (df[self.hist_column] > 0) & (df[self.hist_column].shift(1) < 0) & (df[self.hist_column].shift(2) < 0)
            | (df[self.hist_column] < 0) & (df[self.hist_column].shift(1) > 0) & (df[self.hist_column].shift(2) > 0),
            1,
            0,
        )

        return df
