import numpy as np
import pandas as pd
import talib


class Transformer:
    def __init__(self, price_column: str = "close"):
        self.price_column = price_column
        self.window = 20
        self.num_std = 2

    def calculate_gpt_features(self, df):
        length_bb = 20
        mult_bb = 2.0
        length_kc = 20
        mult_kc = 1.5
        use_true_range = True

        close = df["close"]
        sma_bb = close.rolling(window=length_bb).mean()
        stdev_bb = close.rolling(window=length_bb).std()

        upper_bb = sma_bb + mult_bb * stdev_bb
        lower_bb = sma_bb - mult_bb * stdev_bb

        # Calculate Keltner Channels
        sma_kc = close.rolling(window=length_kc).mean()
        if use_true_range:
            true_range = np.maximum(
                df["high"] - df["low"],
                np.abs(df["high"] - df["close"].shift(1)),
                np.abs(df["low"] - df["close"].shift(1)),
            )
            range_kc = true_range.rolling(window=length_kc).mean()
        else:
            range_kc = (df["high"] - df["low"]).rolling(window=length_kc).mean()

        upper_kc = sma_kc + mult_kc * range_kc
        lower_kc = sma_kc - mult_kc * range_kc

        # Squeeze Logic
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
        no_sqz = ~(sqz_on | sqz_off)

        # Calculate Squeeze Momentum
        highest_high = df["high"].rolling(window=length_kc).max()
        lowest_low = df["low"].rolling(window=length_kc).min()
        linreg_input = close - (0.5 * (highest_high + lowest_low) + sma_kc)
        val = linreg_input.rolling(window=length_kc).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)

        # Calculate the momentum colors (simplified, without plot)
        momentum_colors = []
        for i in range(1, len(val)):
            if val[i] > 0:
                if val[i] > val[i - 1]:
                    momentum_colors.append("lime")  # Green for increasing
                else:
                    momentum_colors.append("green")  # Darker green for decreasing
            else:
                if val[i] < val[i - 1]:
                    momentum_colors.append("red")  # Red for decreasing
                else:
                    momentum_colors.append("maroon")  # Darker red for increasing

        # Handle first value (set it to default)
        momentum_colors.insert(0, "green" if val[0] > 0 else "maroon")

        # Calculate sColor (Squeeze state color)
        sColor = []
        for i in range(len(df)):
            if no_sqz[i]:
                sColor.append("blue")  # No squeeze
            elif sqz_on[i]:
                sColor.append("black")  # Squeeze on
            elif sqz_off[i]:
                sColor.append("gray")  # Squeeze off

        momentum_colors = pd.Series(momentum_colors, name="gpt_momentum_colors")
        sColor = pd.Series(sColor, name="gpt_sColor")
        return pd.concat(
            [
                df,
                val.rename("gpt_val"),
                sqz_on.rename("gpt_sqz_on"),
                sqz_off.rename("gpt_sqz_off"),
                momentum_colors,
                sColor,
            ],
            axis=1,
        )

    def calculate_claud_features(self, df):
        bb_length = 20
        # bb_mult = 2.0
        kc_length = 20
        kc_mult = 1.5
        use_true_range = True
        color_map = {0: "black", 1: "gray", 2: "blue"}

        basis = talib.SMA(df["close"], bb_length)
        kc_ma = talib.SMA(df["close"], kc_length)

        dev = kc_mult * talib.STDDEV(df["close"], bb_length)

        upper_bb = basis + dev
        lower_bb = basis - dev

        if use_true_range:
            range_values = talib.TRANGE(df["high"], df["low"], df["close"])
        else:
            range_values = df["high"] - df["low"]

        range_ma = talib.SMA(range_values, kc_length)

        upper_kc = kc_ma + range_ma * kc_mult
        lower_kc = kc_ma - range_ma * kc_mult

        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
        no_sqz = ~(sqz_on | sqz_off)

        val = np.zeros_like(df["close"])
        for i in range(kc_length, len(df["close"])):
            window_high = np.max(df["high"][i - kc_length : i])
            window_low = np.min(df["low"][i - kc_length : i])
            window_sma = np.mean(df["close"][i - kc_length : i])

            val[i] = df["close"][i] - np.mean([window_high, window_low, window_sma])

        s_color = np.zeros_like(df["close"])
        s_color[:] = 2

        s_color[no_sqz] = 2
        s_color[sqz_on] = 0
        s_color[sqz_off] = 1
        no_sqz = pd.Series(no_sqz, name="claud_no_sqz")
        val = pd.Series(val, name="claud_val")
        s_color = pd.Series(s_color, name="claud_sColor")
        s_color = s_color.map(color_map)
        return pd.concat(
            [
                df,
                val.rename("claud_val"),
                sqz_on.rename("claud_sqz_on"),
                sqz_off.rename("claud_sqz_off"),
                no_sqz.rename("claud_no_sqz"),
                s_color.rename("claud_sColor"),
            ],
            axis=1,
        )

    def calculate_features(self, df):
        length = 20
        mult = 2
        length_KC = 20
        mult_KC = 1.5

        # calculate BB
        m_avg = df["close"].rolling(window=length).mean()
        m_std = df["close"].rolling(window=length).std(ddof=0)
        upper_BB = m_avg + mult * m_std
        lower_BB = m_avg - mult * m_std

        # calculate true range
        df["tr0"] = abs(df["high"] - df["low"])
        df["tr1"] = abs(df["high"] - df["close"].shift())
        df["tr2"] = abs(df["low"] - df["close"].shift())
        df["tr"] = df[["tr0", "tr1", "tr2"]].max(axis=1)

        # calculate KC
        range_ma = df["tr"].rolling(window=length_KC).mean()
        upper_KC = m_avg + range_ma * mult_KC
        lower_KC = m_avg - range_ma * mult_KC

        # calculate bar value
        highest = df["high"].rolling(window=length_KC).max()
        lowest = df["low"].rolling(window=length_KC).min()
        m1 = (highest + lowest) / 2
        df["value"] = df["close"] - (m1 + m_avg) / 2
        fit_y = np.array(range(0, length_KC))
        df["value"] = (
            df["value"]
            .rolling(window=length_KC)
            .apply(lambda x: np.polyfit(fit_y, x, 1)[0] * (length_KC - 1) + np.polyfit(fit_y, x, 1)[1], raw=True)
        )

        # check for 'squeeze'
        df["squeeze_on"] = (lower_BB > lower_KC) & (upper_BB < upper_KC)
        df["squeeze_off"] = (lower_BB < lower_KC) & (upper_BB > upper_KC)

        # buying window for long position:
        # 1. black cross becomes gray (the squeeze is released)
        # long_cond1 = (df["squeeze_off"].iloc[-2] == False) & (df["squeeze_off"].iloc[-1] == True)

        # 2. bar value is positive => the bar is light green k
        # long_cond2 = df["value"].iloc[-1] > 0
        # enter_long = long_cond1 and long_cond2

        # buying window for short position:
        # 1. black cross becomes gray (the squeeze is released)
        # short_cond1 = (df["squeeze_off"].iloc[-2] == False) & (df["squeeze_off"].iloc[-1] == True)

        # 2. bar value is negative => the bar is light red
        # short_cond2 = df["value"].iloc[-1] < 0
        # enter_short = short_cond1 and short_cond2
        return df.drop(["tr0", "tr1", "tr2", "tr"], axis=1)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = self.calculate_features(df)
        # df = self.calculate_gpt_features(df)
        # df = self.calculate_claud_features(df)
        return df
