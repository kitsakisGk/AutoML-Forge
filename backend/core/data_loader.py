"""
Data Loader - Multi-format data loading
"""
import pandas as pd
from pathlib import Path
from typing import Optional


class DataLoader:
    """Load data from various file formats"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_extension = self.file_path.suffix.lower()

    def load(self) -> pd.DataFrame:
        """
        Load data based on file extension

        Returns:
            pandas DataFrame

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        loaders = {
            ".csv": self._load_csv,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".json": self._load_json,
            ".parquet": self._load_parquet,
        }

        loader = loaders.get(self.file_extension)
        if loader is None:
            raise ValueError(
                f"Unsupported file format: {self.file_extension}. "
                f"Supported: {list(loaders.keys())}"
            )

        return loader()

    def _load_csv(self) -> pd.DataFrame:
        """Load CSV file"""
        return pd.read_csv(self.file_path)

    def _load_excel(self) -> pd.DataFrame:
        """Load Excel file"""
        return pd.read_excel(self.file_path)

    def _load_json(self) -> pd.DataFrame:
        """Load JSON file"""
        return pd.read_json(self.file_path)

    def _load_parquet(self) -> pd.DataFrame:
        """Load Parquet file"""
        return pd.read_parquet(self.file_path)
