"""
Tests for DataProfiler
"""
import pytest
import pandas as pd
from backend.core.data_profiler import DataProfiler


def test_generate_profile():
    """Test generating data profile"""
    # Create sample dataframe
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, None],
        'income': [50000, 60000, 70000, 80000, 90000],
        'city': ['Zurich', 'Geneva', 'Basel', 'Zurich', 'Geneva']
    })

    profiler = DataProfiler(df)
    profile = profiler.generate_profile()

    # Check structure
    assert 'shape' in profile
    assert 'dtypes' in profile
    assert 'missing' in profile
    assert 'numeric_stats' in profile
    assert 'correlations' in profile
    assert 'unique_values' in profile
    assert 'memory_usage' in profile
    assert 'column_details' in profile

    # Check shape
    assert profile['shape']['rows'] == 5
    assert profile['shape']['columns'] == 3

    # Check missing values
    assert profile['missing']['counts']['age'] == 1
    assert profile['missing']['percentages']['age'] == 20.0


def test_empty_dataframe():
    """Test profiling empty dataframe"""
    df = pd.DataFrame()
    profiler = DataProfiler(df)
    profile = profiler.generate_profile()

    assert profile['shape']['rows'] == 0
    assert profile['shape']['columns'] == 0
