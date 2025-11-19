"""
Data Profiler - Automatic data analysis and profiling
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class DataProfiler:
    """Generate comprehensive data profile"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate comprehensive data profile

        Returns:
            Dictionary containing:
            - shape: rows and columns
            - dtypes: data types per column
            - missing: missing values analysis
            - numeric_stats: statistics for numeric columns
            - correlations: correlation matrix
            - unique_values: unique value counts
            - memory_usage: memory consumption
            - column_details: detailed analysis per column
        """
        profile = {
            "shape": {"rows": int(self.df.shape[0]), "columns": int(self.df.shape[1])},
            "dtypes": self._get_dtypes(),
            "missing": self._analyze_missing(),
            "numeric_stats": self._get_numeric_stats(),
            "correlations": self._get_correlations(),
            "unique_values": self._count_unique(),
            "memory_usage": self._get_memory_usage(),
            "column_details": self._get_column_details(),
        }

        return profile

    def _get_dtypes(self) -> Dict[str, str]:
        """Get data types for each column"""
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    def _analyze_missing(self) -> Dict[str, Any]:
        """Analyze missing values"""
        missing_counts = self.df.isnull().sum()
        total_rows = len(self.df)

        return {
            "counts": {col: int(count) for col, count in missing_counts.items()},
            "percentages": {
                col: float(round(count / total_rows * 100, 2))
                for col, count in missing_counts.items()
            },
            "total_missing": int(missing_counts.sum()),
        }

    def _get_numeric_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {}

        stats = numeric_df.describe().to_dict()

        # Convert numpy types to Python types for JSON serialization
        return {
            col: {stat: float(value) for stat, value in stat_dict.items()}
            for col, stat_dict in stats.items()
        }

    def _get_correlations(self) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.empty or len(numeric_df.columns) < 2:
            return {}

        corr = numeric_df.corr()

        return {
            col: {other_col: float(value) for other_col, value in row.items()}
            for col, row in corr.iterrows()
        }

    def _count_unique(self) -> Dict[str, int]:
        """Count unique values per column"""
        return {col: int(self.df[col].nunique()) for col in self.df.columns}

    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        memory = self.df.memory_usage(deep=True)

        return {
            "total_bytes": int(memory.sum()),
            "total_mb": float(round(memory.sum() / 1024 / 1024, 2)),
            "per_column": {col: int(mem) for col, mem in memory.items()},
        }

    def _get_column_details(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed analysis for each column"""
        details = {}

        for col in self.df.columns:
            col_data = self.df[col]
            col_details = {
                "dtype": str(col_data.dtype),
                "unique_count": int(col_data.nunique()),
                "null_count": int(col_data.isnull().sum()),
                "null_percentage": float(
                    round(col_data.isnull().sum() / len(col_data) * 100, 2)
                ),
            }

            # Add numeric-specific details
            if pd.api.types.is_numeric_dtype(col_data):
                col_details.update(
                    {
                        "min": float(col_data.min()) if not col_data.empty else None,
                        "max": float(col_data.max()) if not col_data.empty else None,
                        "mean": float(col_data.mean()) if not col_data.empty else None,
                        "median": (
                            float(col_data.median()) if not col_data.empty else None
                        ),
                        "std": float(col_data.std()) if not col_data.empty else None,
                    }
                )

            # Add categorical-specific details
            elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(
                col_data
            ):
                value_counts = col_data.value_counts().head(10)
                col_details.update(
                    {
                        "top_values": {
                            str(k): int(v) for k, v in value_counts.items()
                        },
                        "cardinality": "high"
                        if col_data.nunique() > len(col_data) * 0.5
                        else "low",
                    }
                )

            details[col] = col_details

        return details
