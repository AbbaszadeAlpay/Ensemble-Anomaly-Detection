import os

import pandas as pd

ALLOWED_EXTENSIONS = [".csv", ".txt", "parquet", "xlsx"]

DATA_READ_MAPPING = {
    ".txt": pd.read_csv,
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
}


class DatasetIO:

    """This class implements the Dataset IO class."""

    def __init__(self, DATA_PATH):
        self.DATA_PATH = DATA_PATH
        self.extension = os.path.splitext(self.DATA_PATH)[-1]
        self.__validate_extensions()

    def __validate_extensions(self):
        """Validate the extension."""

        if not isinstance(self.DATA_PATH, str):
            raise TypeError(f"Expected data_path as str, found {type(self.DATA_PATH)}")

        if self.extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File extension is not supported. found {self.extension}")

    def read_data(self):
        """Read the data."""

        read_func = DATA_READ_MAPPING.get(self.extension, None)

        if read_func is None:
            raise ValueError(f"Received unsuported extension, found {self.extension}")

        else:
            data = read_func(self.DATA_PATH)

        return data
