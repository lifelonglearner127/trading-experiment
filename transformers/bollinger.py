import numpy as np
import pandas as pd


class Transformer:
    def __init__(self, price_column: str = "close"):
        self.price_column = price_column
        self.window = 20
        self.num_std = 2

    def calculate_basic_bollinger_features(self, df):
        prices = df[self.price_column]
        middle_band = prices.rolling(window=self.window).mean()
        std = prices.rolling(window=self.window).std()
        upper_band = middle_band + (std * self.num_std)
        lower_band = middle_band - (std * self.num_std)

        features = pd.DataFrame(index=df.index)
        features["bb_position"] = (prices - middle_band) / (upper_band - middle_band)
        features["bb_width"] = (upper_band - lower_band) / middle_band
        features["bb_ratio"] = (prices - lower_band) / (upper_band - lower_band)

        return features, (middle_band, upper_band, lower_band)

    def calculate_pattern_features(self, df, bands):
        middle_band, upper_band, lower_band = bands
        prices = df[self.price_column]

        features = pd.DataFrame(index=df.index)
        features["bb_upper_break"] = (prices > upper_band).astype(int)
        features["bb_lower_break"] = (prices < lower_band).astype(int)
        conditions = [
            (prices > upper_band),
            (prices > middle_band) & (prices <= upper_band),
            (prices > lower_band) & (prices <= middle_band),
            (prices <= lower_band),
        ]
        choices = [3, 2, 1, 0]
        features["bb_zone"] = np.select(conditions, choices, default=np.nan)
        band_width = upper_band - lower_band
        features["bb_squeeze"] = band_width.pct_change()
        features["consecutive_upper"] = ((prices > upper_band) & (prices.shift(1) > upper_band)).astype(int)
        features["consecutive_lower"] = ((prices < lower_band) & (prices.shift(1) < lower_band)).astype(int)

        return features

    def calculate_momentum_features(self, df, bands):
        middle_band, upper_band, lower_band = bands
        prices = df[self.price_column]

        features = pd.DataFrame(index=df.index)
        features["bb_middle_slope"] = middle_band.diff() / middle_band
        features["bb_width_slope"] = ((upper_band - lower_band) / middle_band).diff()
        features["price_to_bb_momentum"] = (prices - middle_band) / prices.shift(1)
        features["bb_volatility"] = (upper_band - lower_band).rolling(self.window).std() / middle_band
        features["trend_strength"] = abs(features["bb_middle_slope"]).rolling(self.window).mean()

        return features

    def calculate_volatility_features(self, df, bands):
        middle_band, upper_band, lower_band = bands
        prices = df[self.price_column]

        features = pd.DataFrame(index=df.index)
        band_width = upper_band - lower_band
        features["bb_vol_ratio"] = prices.rolling(self.window).std() / band_width
        touch_threshold = 0.05  # 5% 이내 근접을 터치로 간주
        upper_touch = (prices - upper_band).abs() / prices < touch_threshold
        lower_touch = (prices - lower_band).abs() / prices < touch_threshold
        features["bb_touch_count"] = (upper_touch | lower_touch).rolling(self.window).sum()
        features["bb_crossing_count"] = (
            ((prices.shift(1) - middle_band.shift(1)) * (prices - middle_band) < 0).rolling(self.window).sum()
        )
        features["volatility_trend"] = band_width.pct_change().rolling(self.window).mean()

        return features

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        basic_features, bands = self.calculate_basic_bollinger_features(df)
        pattern_features = self.calculate_pattern_features(df, bands)
        momentum_features = self.calculate_momentum_features(df, bands)
        volatility_features = self.calculate_volatility_features(df, bands)

        return pd.concat([df, basic_features, pattern_features, momentum_features, volatility_features], axis=1)
