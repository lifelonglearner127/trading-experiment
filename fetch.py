import datetime
import random
import time

import constants as c
from providers import DataFetcher as BinanceDataFetcher
from serializers import CSVSerializer

if __name__ == "__main__":
    fetcher = BinanceDataFetcher()
    serializer = CSVSerializer(output_dir=c.CSV_OUTPUT_DIR_PATH)

    for interval in c.TimeInterval:
        start_date = datetime.date(year=2018, month=5, day=1)
        end_date = datetime.date.today()

        while start_date < end_date:
            current_end_date = start_date + datetime.timedelta(days=90)
            current_end_date = min(current_end_date, end_date)
            print(f"Getting {interval} Data from {start_date.isoformat()} - {current_end_date.isoformat()}")
            data = fetcher.get_data(
                symbol=c.SYMBOL,
                interval=interval.value,
                start_date=start_date.isoformat(),
                end_date=current_end_date.isoformat(),
            )
            serializer.to_csv(data=data, file_name=f"{c.SYMBOL}_{interval.value}.csv")
            print(f"Loaded {len(data)} Data from {start_date.isoformat()} - {current_end_date.isoformat()}")
            wait_seconds = random.randrange(30, 60)
            print(f"Sleeping {wait_seconds} seconds")
            print("=" * 30)
            time.sleep(wait_seconds)
            start_date = current_end_date
