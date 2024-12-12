import pandas as pd

import constants as c
from serializers import CSVSerializer

serializer = CSVSerializer(output_dir=c.CSV_OUTPUT_DIR_PATH)
symbol = "BTCUSDT"

start_date = "2017-08-17"
end_date = "2024-11-18"
frequencies = {
    c.TimeInterval.INTERVAL_5M: "5min",
    c.TimeInterval.INTERVAL_15M: "15min",
    c.TimeInterval.INTERVAL_1H: "1h",
    c.TimeInterval.INTERVAL_4H: "4h",
    c.TimeInterval.INTERVAL_8H: "8h",
    c.TimeInterval.INTERVAL_12H: "12h",
    c.TimeInterval.INTERVAL_1D: "1D",
}

for interval in c.TimeInterval:
    df = serializer.from_csv(file_name=f"{symbol}_{interval.value}_cleaned.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    expected_range = pd.date_range(start=start_date, end=end_date, freq=frequencies[interval])
    missing_timestamps = set(expected_range) - set(df["timestamp"])

    if missing_timestamps:
        print(f"Missing {len(missing_timestamps)} timestamps: ")
        for timestamp in sorted(missing_timestamps):
            print(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print("No missing timestamps")
