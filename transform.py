import constants as c
from serializers import CSVSerializer, DBSerializer
from transformers import (
    BollingerTransformer,
    EmaTransformer,
    IndicatorTransformer,
    LabelTransformer,
    MACDTransformer,
    RSITransformer,
    SqueezeMomentTransformer,
    VolumeTransformer,
)

if __name__ == "__main__":
    csv_serializer = CSVSerializer(output_dir=c.CSV_OUTPUT_DIR_PATH)
    db_serializer = DBSerializer(c.DB_URI)

    transformers = [
        LabelTransformer(),
        IndicatorTransformer(),
        VolumeTransformer(),
        BollingerTransformer(),
        MACDTransformer(),
        RSITransformer(),
        EmaTransformer(),
        SqueezeMomentTransformer(),
    ]

    for interval in c.TimeInterval:
        print(f"Processing {interval} data")
        df = csv_serializer.from_csv(file_name=f"{c.SYMBOL}_{interval.value}_cleaned.csv")

        for i, transformer in enumerate(transformers):
            print(f"Applying {i + 1}th transformer")
            df = transformer.transform(df, interval=interval)
            print(f"Applied {i + 1}th transformer")

        print("Saving to the table")
        db_serializer.save_to_db(f"{c.SYMBOL}_in_{interval.value}", df)
