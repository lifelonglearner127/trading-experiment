import pandas as pd
from sqlalchemy import BigInteger, Column, Float, MetaData, String, Table, create_engine


class DBSerializer:
    def __init__(self, db_uri: str):
        self.engine = create_engine(db_uri)
        self.metadata = MetaData()

    def create_table(self, table_name: str, df: pd.DataFrame):
        columns = [Column("id", BigInteger, primary_key=True, autoincrement=True)]
        for col in df.columns:
            if col == "id":
                continue
            if pd.api.types.is_integer_dtype(df[col]):
                col_type = BigInteger
            elif pd.api.types.is_float_dtype(df[col]):
                col_type = Float
            else:
                col_type = String
            columns.append(Column(col, col_type))

        table = Table(table_name, self.metadata, *columns)
        self.metadata.create_all(self.engine)
        return table

    def save_to_db(self, table_name: str, df: pd.DataFrame):
        with self.engine.connect() as connection:
            if not self.engine.dialect.has_table(connection, table_name):
                self.create_table(table_name, df)

        df.to_sql(table_name, self.engine, if_exists="append", index=False)
        print(f"DataFrame saved to table: {table_name}")

    def load_from_db(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql_table(table_name, self.engine)


if __name__ == "__main__":
    DBSerializer
