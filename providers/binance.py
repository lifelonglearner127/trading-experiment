import pandas as pd
from binance.client import Client


class DataFetcher:
    def __init__(self):
        self.client = Client()

    def get_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol, interval=interval, start_str=start_date, end_str=end_date
            )

            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].astype(float)
            return df

        except Exception as e:
            print(f"Error: {e}")
            raise e
