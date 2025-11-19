"""
Tests for DataLoader
"""
import pytest
import pandas as pd
from pathlib import Path
from backend.core.data_loader import DataLoader


def test_load_csv():
    """Test loading CSV file"""
    test_file = Path("tests/fixtures/sample_data.csv")
    if not test_file.exists():
        pytest.skip("Test fixture not found")

    loader = DataLoader(str(test_file))
    df = loader.load()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert len(df.columns) > 0


def test_unsupported_format(tmp_path):
    """Test loading unsupported file format"""
    # Create a temporary .txt file
    test_file = tmp_path / "test.txt"
    test_file.write_text("some content")

    loader = DataLoader(str(test_file))

    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load()


def test_file_not_found():
    """Test loading non-existent file"""
    loader = DataLoader("nonexistent.csv")

    with pytest.raises(FileNotFoundError):
        loader.load()
