import constants as c
from serializers import DBSerializer
from transformers import SqueezeMomentTransformer  # EmaTransformer,

if __name__ == "__main__":
    db_serializer = DBSerializer(c.DB_URI)
    transformers = [
        # EmaTransformer(),
        SqueezeMomentTransformer(),
    ]

    for interval in c.TimeInterval:
        print(f"Reading {interval} data")
        df = db_serializer.load_from_db(table_name=f"{c.SYMBOL}_in_{interval.value}")
        # df.drop("label", axis=1, inplace=True)
        for i, transformer in enumerate(transformers):
            print(f"Applying {i + 1}th transformer")
            df = transformer.transform(df, interval=interval)
            print(f"Applied {i + 1}th transformer")

        print("Saving to the table")
        db_serializer.save_to_db(f"{c.SYMBOL}_in_{interval.value}_latest", df)
