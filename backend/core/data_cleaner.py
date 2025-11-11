"""
Smart Data Cleaner - Automatic data cleaning suggestions with explanations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats


class SmartDataCleaner:
    """
    Analyzes data and suggests intelligent cleaning strategies with explanations
    """

    def __init__(self, df: pd.DataFrame, profile: Optional[Dict] = None):
        """
        Initialize the cleaner

        Args:
            df: DataFrame to clean
            profile: Optional pre-generated data profile
        """
        self.df = df.copy()
        self.profile = profile
        self.suggestions = []

    def analyze_and_suggest(self) -> List[Dict[str, Any]]:
        """
        Analyze data and generate cleaning suggestions

        Returns:
            List of suggestion dictionaries with structure:
            {
                'issue': 'description',
                'column': 'column_name',
                'severity': 'high|medium|low',
                'suggestion': 'what to do',
                'reason': 'why this suggestion',
                'details': {...},
                'alternatives': [...]
            }
        """
        self.suggestions = []

        # Analyze each column
        for col in self.df.columns:
            # Check for missing values
            self._check_missing_values(col)

            # Check for outliers (numeric columns only)
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self._check_outliers(col)

            # Check data type issues
            self._check_data_types(col)

            # Check for high cardinality categoricals
            self._check_cardinality(col)

        return self.suggestions

    def _check_missing_values(self, col: str):
        """Check for missing values and suggest imputation strategies"""
        missing_count = self.df[col].isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100

        if missing_count == 0:
            return

        is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
        col_data = self.df[col].dropna()

        if missing_pct < 5:
            severity = "low"
        elif missing_pct < 30:
            severity = "medium"
        else:
            severity = "high"

        if is_numeric:
            # Numeric column
            median_val = col_data.median()
            mean_val = col_data.mean()
            std_val = col_data.std()

            # Determine if there are outliers
            has_outliers = self._has_outliers(col_data)

            if has_outliers:
                suggestion = f"Impute with median ({median_val:.2f})"
                reason = "Numerical column with outliers detected. Median is robust to outliers and won't be skewed by extreme values."
            else:
                suggestion = f"Impute with mean ({mean_val:.2f})"
                reason = "Numerical column without significant outliers. Mean preserves the overall distribution."

            alternatives = [
                {
                    "method": "Drop rows",
                    "impact": f"-{missing_count} rows ({missing_pct:.1f}% data loss)",
                    "description": "Remove all rows with missing values"
                },
                {
                    "method": "Use mode" if not has_outliers else "Use median",
                    "value": col_data.mode()[0] if len(col_data.mode()) > 0 else None,
                    "description": "Use most frequent value" if not has_outliers else "Use middle value"
                },
                {
                    "method": "Forward fill",
                    "description": "Use previous valid value (works for time series)"
                }
            ]

        else:
            # Categorical column
            mode_val = col_data.mode()[0] if len(col_data.mode()) > 0 else "N/A"
            unique_count = col_data.nunique()

            suggestion = f"Impute with mode ('{mode_val}')"
            reason = f"Categorical column with {unique_count} unique values. Mode (most frequent value) is the standard approach for categorical data."

            alternatives = [
                {
                    "method": "Drop rows",
                    "impact": f"-{missing_count} rows ({missing_pct:.1f}% data loss)",
                    "description": "Remove all rows with missing values"
                },
                {
                    "method": "Create 'Unknown' category",
                    "description": "Replace missing values with explicit 'Unknown' or 'Missing' label"
                },
                {
                    "method": "Predictive imputation",
                    "description": "Use other columns to predict missing values (advanced)"
                }
            ]

        self.suggestions.append({
            "issue": f"Missing values in '{col}'",
            "column": col,
            "severity": severity,
            "suggestion": suggestion,
            "reason": reason,
            "details": {
                "missing_count": int(missing_count),
                "missing_percentage": float(round(missing_pct, 2)),
                "total_rows": len(self.df),
                "data_type": str(self.df[col].dtype)
            },
            "alternatives": alternatives,
            "action": "impute_missing"
        })

    def _check_outliers(self, col: str):
        """Check for outliers in numeric columns"""
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            return

        col_data = self.df[col].dropna()

        if len(col_data) < 10:  # Not enough data
            return

        # IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_iqr = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        outlier_count = len(outliers_iqr)
        outlier_pct = (outlier_count / len(col_data)) * 100

        if outlier_count == 0:
            return

        if outlier_pct < 2:
            severity = "low"
        elif outlier_pct < 5:
            severity = "medium"
        else:
            severity = "high"

        suggestion = f"Cap outliers at IQR boundaries (lower: {lower_bound:.2f}, upper: {upper_bound:.2f})"
        reason = f"Detected {outlier_count} outliers ({outlier_pct:.1f}%) using IQR method. Capping preserves all data while reducing the impact of extreme values."

        alternatives = [
            {
                "method": "Remove outliers",
                "impact": f"-{outlier_count} rows ({outlier_pct:.1f}% data loss)",
                "description": "Delete rows containing outliers"
            },
            {
                "method": "Log transformation",
                "description": "Apply log transformation to reduce skewness (if positive values)"
            },
            {
                "method": "Z-score method",
                "description": "Remove values beyond 3 standard deviations"
            },
            {
                "method": "Keep outliers",
                "description": "Outliers might be legitimate extreme values"
            }
        ]

        self.suggestions.append({
            "issue": f"Outliers detected in '{col}'",
            "column": col,
            "severity": severity,
            "suggestion": suggestion,
            "reason": reason,
            "details": {
                "outlier_count": int(outlier_count),
                "outlier_percentage": float(round(outlier_pct, 2)),
                "iqr_lower_bound": float(round(lower_bound, 2)),
                "iqr_upper_bound": float(round(upper_bound, 2)),
                "min_value": float(col_data.min()),
                "max_value": float(col_data.max())
            },
            "alternatives": alternatives,
            "action": "handle_outliers"
        })

    def _check_data_types(self, col: str):
        """Check for potential data type issues"""
        col_data = self.df[col].dropna()

        if len(col_data) == 0:
            return

        # Check if string column could be datetime
        if pd.api.types.is_object_dtype(self.df[col]):
            # Try to detect dates
            sample_size = min(100, len(col_data))
            sample = col_data.sample(sample_size) if len(col_data) > sample_size else col_data

            try:
                pd.to_datetime(sample, errors='raise')
                # If successful, it's likely a date column

                suggestion = f"Convert to datetime type"
                reason = f"Column contains date/time strings. Converting to datetime enables time-based operations and proper sorting."

                self.suggestions.append({
                    "issue": f"Potential datetime column '{col}' stored as text",
                    "column": col,
                    "severity": "medium",
                    "suggestion": suggestion,
                    "reason": reason,
                    "details": {
                        "current_type": str(self.df[col].dtype),
                        "suggested_type": "datetime64",
                        "sample_values": sample.head(3).tolist()
                    },
                    "alternatives": [
                        {
                            "method": "Keep as string",
                            "description": "No conversion needed if not using date operations"
                        }
                    ],
                    "action": "convert_dtype"
                })
            except:
                pass

    def _check_cardinality(self, col: str):
        """Check for high cardinality categorical columns"""
        if pd.api.types.is_numeric_dtype(self.df[col]):
            return

        col_data = self.df[col].dropna()
        unique_count = col_data.nunique()
        unique_ratio = unique_count / len(col_data) if len(col_data) > 0 else 0

        # High cardinality (probably an ID column or needs encoding)
        if unique_ratio > 0.9 and unique_count > 50:
            suggestion = f"Consider removing '{col}' (likely an ID column)"
            reason = f"Column has {unique_count} unique values ({unique_ratio*100:.1f}% of rows). High cardinality columns are typically IDs and don't help ML models."

            self.suggestions.append({
                "issue": f"High cardinality in '{col}' (possible ID column)",
                "column": col,
                "severity": "low",
                "suggestion": suggestion,
                "reason": reason,
                "details": {
                    "unique_count": int(unique_count),
                    "unique_ratio": float(round(unique_ratio, 3)),
                    "total_rows": len(col_data)
                },
                "alternatives": [
                    {
                        "method": "Keep column",
                        "description": "Keep if it's actually meaningful for analysis"
                    },
                    {
                        "method": "Feature engineering",
                        "description": "Extract useful features (e.g., extract domain from email)"
                    }
                ],
                "action": "consider_removal"
            })

        # Medium cardinality (good for encoding)
        elif 2 < unique_count <= 20:
            suggestion = f"Good candidate for one-hot encoding"
            reason = f"Column has {unique_count} categories - perfect for one-hot encoding to make it usable in ML models."

            self.suggestions.append({
                "issue": f"Categorical column '{col}' needs encoding for ML",
                "column": col,
                "severity": "low",
                "suggestion": suggestion,
                "reason": reason,
                "details": {
                    "unique_count": int(unique_count),
                    "categories": col_data.value_counts().head(10).to_dict()
                },
                "alternatives": [
                    {
                        "method": "Label encoding",
                        "description": "Assign numbers (0, 1, 2...) - use if ordinal"
                    },
                    {
                        "method": "Target encoding",
                        "description": "Encode based on target variable (for supervised learning)"
                    }
                ],
                "action": "encode_categorical"
            })

    def _has_outliers(self, data: pd.Series) -> bool:
        """Quick check if data has outliers"""
        if len(data) < 10:
            return False

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return len(outliers) > 0

    def execute_cleaning(self, suggestions_to_apply: List[str]) -> pd.DataFrame:
        """
        Execute selected cleaning suggestions

        Args:
            suggestions_to_apply: List of issue descriptions to apply

        Returns:
            Cleaned DataFrame
        """
        cleaned_df = self.df.copy()

        for suggestion in self.suggestions:
            if suggestion['issue'] in suggestions_to_apply:
                col = suggestion['column']
                action = suggestion['action']

                if action == "impute_missing":
                    cleaned_df = self._apply_imputation(cleaned_df, suggestion)
                elif action == "handle_outliers":
                    cleaned_df = self._apply_outlier_handling(cleaned_df, suggestion)
                elif action == "convert_dtype":
                    cleaned_df = self._apply_dtype_conversion(cleaned_df, suggestion)

        return cleaned_df

    def _apply_imputation(self, df: pd.DataFrame, suggestion: Dict) -> pd.DataFrame:
        """Apply missing value imputation"""
        col = suggestion['column']

        if pd.api.types.is_numeric_dtype(df[col]):
            # Use median or mean based on suggestion
            if "median" in suggestion['suggestion'].lower():
                fill_value = df[col].median()
            else:
                fill_value = df[col].mean()
        else:
            # Use mode for categorical
            fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"

        df[col].fillna(fill_value, inplace=True)
        return df

    def _apply_outlier_handling(self, df: pd.DataFrame, suggestion: Dict) -> pd.DataFrame:
        """Apply outlier capping"""
        col = suggestion['column']
        details = suggestion['details']

        lower_bound = details['iqr_lower_bound']
        upper_bound = details['iqr_upper_bound']

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df

    def _apply_dtype_conversion(self, df: pd.DataFrame, suggestion: Dict) -> pd.DataFrame:
        """Apply data type conversion"""
        col = suggestion['column']

        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass  # Keep original if conversion fails

        return df

    def generate_cleaning_script(self, suggestions_to_apply: List[str]) -> str:
        """
        Generate executable Python script for the cleaning pipeline

        Args:
            suggestions_to_apply: List of issues to include in script

        Returns:
            Python script as string
        """
        script_lines = [
            "# AutoML Forge - Generated Data Cleaning Script",
            "# Generated automatically based on data analysis",
            "",
            "import pandas as pd",
            "import numpy as np",
            "",
            "def clean_data(df):",
            '    """Apply data cleaning transformations"""',
            "    df = df.copy()",
            ""
        ]

        for suggestion in self.suggestions:
            if suggestion['issue'] in suggestions_to_apply:
                col = suggestion['column']
                action = suggestion['action']

                script_lines.append(f"    # {suggestion['issue']}")
                script_lines.append(f"    # {suggestion['reason']}")

                if action == "impute_missing":
                    if "median" in suggestion['suggestion'].lower():
                        script_lines.append(f"    df['{col}'].fillna(df['{col}'].median(), inplace=True)")
                    elif "mean" in suggestion['suggestion'].lower():
                        script_lines.append(f"    df['{col}'].fillna(df['{col}'].mean(), inplace=True)")
                    else:
                        script_lines.append(f"    df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)")

                elif action == "handle_outliers":
                    lower = suggestion['details']['iqr_lower_bound']
                    upper = suggestion['details']['iqr_upper_bound']
                    script_lines.append(f"    df['{col}'] = df['{col}'].clip(lower={lower}, upper={upper})")

                elif action == "convert_dtype":
                    script_lines.append(f"    df['{col}'] = pd.to_datetime(df['{col}'])")

                script_lines.append("")

        script_lines.extend([
            "    return df",
            "",
            "",
            "# Usage:",
            "# df = pd.read_csv('your_data.csv')",
            "# df_cleaned = clean_data(df)",
            "# df_cleaned.to_csv('cleaned_data.csv', index=False)"
        ])

        return "\n".join(script_lines)
