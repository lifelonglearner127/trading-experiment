import constants as c
from serializers import CSVSerializer, DBSerializer

if __name__ == "__main__":
    db_serializer = DBSerializer(c.DB_URI)
    csv_serializer = CSVSerializer(output_dir=c.CSV_OUTPUT_DIR_PATH)

    for interval in c.TimeInterval:
        df = db_serializer.load_from_db(table_name=f"binance_crypto_prices_in_{interval.value}_rev02")
        df.drop("id", axis=1, inplace=True)
        csv_serializer.to_csv(df, f"btc_{interval.value}.csv")
