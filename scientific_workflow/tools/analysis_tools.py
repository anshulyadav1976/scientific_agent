"""Tools for statistical analysis and pattern detection."""

import logging
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from scipy.stats import skew, kurtosis # For descriptive stats

from portia.tool import Tool, ToolRunContext, ToolHardError, ToolSoftError

logger = logging.getLogger(__name__)

# --- AnalysisTool --- #

class AnalysisInputSchema(BaseModel):
    """Input schema for the AnalysisTool."""
    # Option 1: Pass file path (tool re-reads)
    file_path: str = Field(..., description="The absolute path to the CSV file to analyze.")
    # Option 2: Pass data reference (requires previous step to output df reference - more complex)
    # data_reference: Any = Field(..., description="Reference to the dataframe in memory (if possible).")
    correlation_threshold: float = Field(default=0.7, description="Absolute threshold for reporting strong correlations.")
    outlier_iqr_multiplier: float = Field(default=1.5, description="IQR multiplier for detecting outliers.")

class CorrelationInfo(BaseModel):
    column_1: str
    column_2: str
    correlation: float

class OutlierInfo(BaseModel):
    column: str
    lower_bound: float
    upper_bound: float
    outliers_low: List[Any]
    outliers_high: List[Any]
    outlier_count: int

class DescriptiveStatInfo(BaseModel):
    metric: str
    value: Any

class ColumnStatInfo(BaseModel):
    column_name: str
    stats: List[DescriptiveStatInfo]

class AnalysisOutputSchema(BaseModel):
    """Output schema for the AnalysisTool."""
    strong_correlations: List[CorrelationInfo] = Field(description="List of strongly correlated column pairs.")
    outliers: List[OutlierInfo] = Field(description="Details about outliers detected in numerical columns.")
    descriptive_stats: List[ColumnStatInfo] = Field(description="Descriptive statistics for each column.")
    categorical_info: Dict[str, Dict[str, Any]] = Field(description="Information about dominant categories in object/categorical columns.")


class AnalysisTool(Tool[AnalysisOutputSchema]):
    """Performs correlation analysis, outlier detection (IQR), and calculates descriptive statistics on tabular data."""
    id: str = "analysis_tool"
    name: str = "Tabular Data Analysis Tool"
    description: str = (
        "Analyzes tabular data (CSV) to find strong correlations (Pearson), detect outliers (IQR method), "
        "calculate comprehensive descriptive statistics (mean, median, std, skew, kurtosis, missing%, etc.), "
        "and identify key categorical features. Requires the file path of the CSV."
    )
    args_schema: type[BaseModel] = AnalysisInputSchema
    output_schema: tuple[str, str] = (
        "AnalysisOutputSchema", # String representation of the output type
        "A dictionary containing strong correlations, outliers, descriptive statistics, and categorical info."
    )

    def _calculate_descriptive_stats(self, series: pd.Series) -> List[DescriptiveStatInfo]:
        """Helper to calculate detailed descriptive stats for a series."""
        stats = []
        if pd.api.types.is_numeric_dtype(series):
            stats.extend([
                DescriptiveStatInfo(metric="Mean", value=series.mean()),
                DescriptiveStatInfo(metric="Median", value=series.median()),
                DescriptiveStatInfo(metric="Std Dev", value=series.std()),
                DescriptiveStatInfo(metric="Min", value=series.min()),
                DescriptiveStatInfo(metric="Max", value=series.max()),
                DescriptiveStatInfo(metric="Range", value=series.max() - series.min()),
                DescriptiveStatInfo(metric="Skewness", value=skew(series.dropna())),
                DescriptiveStatInfo(metric="Kurtosis", value=kurtosis(series.dropna())),
                DescriptiveStatInfo(metric="Zeros Count", value=(series == 0).sum()),
            ])
        # Stats applicable to all types
        stats.extend([
             DescriptiveStatInfo(metric="Missing Percentage", value=series.isnull().mean() * 100),
             DescriptiveStatInfo(metric="Unique Count", value=series.nunique()),
             DescriptiveStatInfo(metric="Is Constant", value=series.std() == 0 if pd.api.types.is_numeric_dtype(series) else series.nunique() <= 1),
        ])
        # Format numbers nicely
        for stat in stats:
            if isinstance(stat.value, (float, np.floating)):
                stat.value = round(stat.value, 3)
            elif isinstance(stat.value, (int, np.integer, bool, np.bool_)):
                 stat.value = stat.value # Keep as is
            else:
                stat.value = str(stat.value) # Convert others to string
        return stats

    def run(self, ctx: ToolRunContext, file_path: str, correlation_threshold: float, outlier_iqr_multiplier: float) -> AnalysisOutputSchema:
        """Executes the analysis steps."""
        logger.info(f"[{self.id}] Running analysis on file: {file_path}")
        try:
            # Read CSV - consider adding encoding detection like in ingestion tool if needed
            df = pd.read_csv(file_path)
            if df.empty:
                raise ToolHardError("Input DataFrame is empty.")

            results = {
                "strong_correlations": [],
                "outliers": [],
                "descriptive_stats": [],
                "categorical_info": {}
            }

            # --- 1. Correlation Analysis --- 
            numerical_cols = df.select_dtypes(include=np.number).columns
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i): # Avoid duplicate pairs and self-correlation
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        correlation = corr_matrix.iloc[i, j]
                        if abs(correlation) >= correlation_threshold:
                            results["strong_correlations"].append(CorrelationInfo(
                                column_1=col1,
                                column_2=col2,
                                correlation=round(correlation, 3)
                            ))
                logger.info(f"[{self.id}] Found {len(results['strong_correlations'])} strong correlations.")
            else:
                logger.info(f"[{self.id}] Skipping correlation analysis: Less than 2 numerical columns.")

            # --- 2. Outlier Detection (IQR) --- 
            for col in numerical_cols:
                series = df[col].dropna()
                if not pd.api.types.is_numeric_dtype(series) or series.empty:
                    continue
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_iqr_multiplier * IQR
                upper_bound = Q3 + outlier_iqr_multiplier * IQR
                
                outliers_series = series[(series < lower_bound) | (series > upper_bound)]
                outliers_low = series[series < lower_bound].tolist()
                outliers_high = series[series > upper_bound].tolist()

                if not outliers_series.empty:
                    results["outliers"].append(OutlierInfo(
                        column=col,
                        lower_bound=round(lower_bound, 3),
                        upper_bound=round(upper_bound, 3),
                        outliers_low=outliers_low[:10], # Limit displayed outliers
                        outliers_high=outliers_high[:10],
                        outlier_count=len(outliers_series)
                    ))
            logger.info(f"[{self.id}] Found outliers in {len(results['outliers'])} columns.")

            # --- 3. Descriptive Statistics & Categorical Info --- 
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in df.columns:
                 col_stats = self._calculate_descriptive_stats(df[col])
                 results["descriptive_stats"].append(ColumnStatInfo(
                     column_name=col,
                     stats=col_stats
                 ))
                 # Categorical specific info
                 if col in categorical_cols:
                     mode_info = df[col].mode()
                     mode_val = mode_info.iloc[0] if not mode_info.empty else "N/A"
                     mode_freq = (df[col] == mode_val).mean() * 100 if mode_val != "N/A" else 0
                     results["categorical_info"][col] = {
                         "dominant_category": mode_val,
                         "dominant_frequency_percent": round(mode_freq, 2),
                         "unique_count": df[col].nunique()
                     }

            logger.info(f"[{self.id}] Calculated descriptive stats for {len(df.columns)} columns.")
            
            return AnalysisOutputSchema(**results)

        except FileNotFoundError:
            logger.error(f"[{self.id}] Input file not found: {file_path}")
            raise ToolHardError(f"Analysis failed: File not found at {file_path}")
        except pd.errors.EmptyDataError:
            logger.error(f"[{self.id}] Input file is empty: {file_path}")
            raise ToolHardError("Analysis failed: Input CSV file is empty.")
        except Exception as e:
            logger.error(f"[{self.id}] Unexpected error during analysis: {e}", exc_info=True)
            raise ToolHardError(f"An unexpected error occurred during analysis: {e}") 