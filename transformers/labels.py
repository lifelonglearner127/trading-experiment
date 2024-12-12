import time

import numpy as np
import pandas as pd

from constants import TimeInterval


class Transformer:
    def __init__(self, price_column: str = "close", n: int = 7, k: float = 0.015):
        self.price_column = price_column
        self.n = n
        self.k = k

    def calculate_label(self, df: pd.DataFrame):
        labels = []
        for i in range(len(df)):
            current_close = df[self.price_column][i]
            past_close = df[self.price_column][max(0, i - self.n) : i]
            future_close = df[self.price_column][i + 1 :]

            if not past_close.empty and not future_close.empty:
                min_past_close = past_close.min()
                for future in future_close:
                    if future < min_past_close:
                        labels.append(0)
                        break

                    if future >= current_close * (1 + self.k):
                        labels.append(1)
                        break
                else:
                    labels.append(0)
            else:
                labels.append(0)
        return labels

    @staticmethod
    def calculate_bullish_label(df: pd.DataFrame, k: float):
        labels = []
        for i in range(len(df)):
            current_close = df["close"][i]
            current_low = df["low"][i]
            future_close_prices = df["close"][i + 1 :]
            future_low_prices = df["low"][i + 1 :]
            if not future_close_prices.empty:
                for future_close, future_low in zip(future_close_prices, future_low_prices):
                    if future_low < current_low:
                        labels.append(0)
                        break
                    if future_close >= current_close * (1 + k):
                        labels.append(1)
                        break
                else:
                    labels.append(0)
            else:
                labels.append(0)

        return labels

    @staticmethod
    def calculate_bearish_label(df: pd.DataFrame, k: float):
        labels = []
        for i in range(len(df)):
            current_close = df["close"][i]
            current_high = df["high"][i]
            future_close_prices = df["close"][i + 1 :]
            future_high_prices = df["high"][i + 1 :]
            if not future_close_prices.empty:
                for future_close, future_high in zip(future_close_prices, future_high_prices):
                    if future_high > current_high:
                        labels.append(0)
                        break

                    if future_close <= current_close * (1 - k):
                        labels.append(-1)
                        break
                else:
                    labels.append(0)
            else:
                labels.append(0)

        return labels

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # df["label"] = self.calculate_label(df)
        # df["label_bullish_0"] = self.calculate_bullish_label(df, 0.01)
        # df["label_bullish_1"] = self.calculate_bullish_label(df, 0.02)
        # df["label_bullish_2"] = self.calculate_bullish_label(df, 0.03)
        # df["label_bullish_3"] = self.calculate_bullish_label(df, 0.04)
        # df["label_bullish_4"] = self.calculate_bullish_label(df, 0.05)
        # df["label_bearish_0"] = self.calculate_bearish_label(df, 0.01)
        # df["label_bearish_1"] = self.calculate_bearish_label(df, 0.02)
        # df["label_bearish_2"] = self.calculate_bearish_label(df, 0.03)
        # df["label_bearish_3"] = self.calculate_bearish_label(df, 0.04)
        # df["label_bearish_4"] = self.calculate_bearish_label(df, 0.05)

        interval = kwargs.get("interval")
        m = 1
        k = 0
        # if interval == TimeInterval.INTERVAL_1D:
        #     k = 14
        #     m = 7
        # elif interval == TimeInterval.INTERVAL_12H:
        #     k = 9.8
        #     m = 8
        # elif interval == TimeInterval.INTERVAL_8H:
        #     k = 7.5
        #     m = 8
        # elif interval == TimeInterval.INTERVAL_4H:
        if interval == TimeInterval.INTERVAL_4H:
            k = 4.9
            m = 8
        # elif interval == TimeInterval.INTERVAL_1H:
        #     k = 2.2
        #     m = 7
        # elif interval == TimeInterval.INTERVAL_15M:
        #     k = 1
        #     m = 7
        n = m
        df["label"] = self.advanced_label_candles(df, n, m, k, False)
        return df

    @staticmethod
    def advanced_label_candles(candles, n, m, k, volatility_factor=True):
        labels = np.zeros(len(candles))

        if volatility_factor:
            rolling_std = candles["close"].rolling(window=n).std()
            dynamic_k = k * (1 + rolling_std / candles["close"])
            dynamic_l = 0.8 * dynamic_k
        else:
            dynamic_k = pd.Series([k] * len(candles), index=candles.index)
            dynamic_l = pd.Series([0.8 * k] * len(candles), index=candles.index)

        for i in range(n, len(candles) - m):
            current_close = candles.loc[i, "close"]
            recent_lows = candles.loc[i - n : i, "low"]
            min_low = recent_lows.min()

            is_local_bottom = min_low <= current_close <= min_low * (1 + dynamic_k[i] / 100)

            if is_local_bottom:
                future_close = candles.loc[i + 1 : i + m, "close"]
                future_low = candles.loc[i + 1 : i + m, "low"]

                future_gain = ((future_close.max() - current_close) / current_close) * 100
                future_drawdown = ((future_low.min() - current_close) / current_close) * 100

                if future_drawdown <= -dynamic_l[i]:
                    labels[i] = -1
                elif future_gain >= dynamic_k[i]:
                    labels[i] = 1

        return labels


if __name__ == "__main__":
    import constants as c
    from serializers import CSVSerializer

    symbol = "BTCUSDT"

    start_time = time.perf_counter()
    serializer = CSVSerializer(output_dir=c.CSV_OUTPUT_DIR_PATH)
    df = serializer.from_csv(file_name=f"{symbol}_{c.TimeInterval.INTERVAL_12H.value}_cleaned.csv")

    total_data = len(df)
    n_draft = 7
    k_draft = 0.015
    transformer = Transformer(n=n_draft, k=k_draft)
    df = transformer.transform(df)
    positives = df["label"].sum()
    print(f"{positives} out of {total_data}: {positives * 100 / total_data}")
    end_time = time.perf_counter()
    print(f"It took {end_time - start_time} seconds")
