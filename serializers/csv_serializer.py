import os

import pandas as pd


class CSVSerializer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def to_csv(self, data: pd.DataFrame, file_name: str):
        file_path = os.path.join(self.output_dir, file_name)
        file_exists = os.path.exists(file_path)
        data.to_csv(file_path, index=False, mode="a", header=not file_exists)

    def from_csv(self, file_name: str) -> pd.DataFrame:
        file_path = os.path.join(self.output_dir, file_name)
        return pd.read_csv(file_path)
