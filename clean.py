import constants as c
from serializers import CSVSerializer

serializer = CSVSerializer(output_dir=c.CSV_OUTPUT_DIR_PATH)

for interval in c.TimeInterval:
    df = serializer.from_csv(file_name=f"{c.SYMBOL}_{interval.value}.csv")
    df = df.drop_duplicates(keep="first")
    serializer.to_csv(df, f"{c.SYMBOL}_{interval.value}_cleaned.csv")
