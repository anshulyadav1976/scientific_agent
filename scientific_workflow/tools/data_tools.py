"""Tools for Data Ingestion and Cleaning"""

import logging
import pandas as pd
from io import StringIO
from pydantic import BaseModel, Field

from portia.tool import Tool, ToolRunContext, ToolHardError

logger = logging.getLogger(__name__)

# --- DataIngestionTool --- #

class DataIngestionInputSchema(BaseModel):
    """Input schema for the DataIngestionTool."""
    file_path: str = Field(..., description="The absolute path to the CSV file to ingest.")

class DataIngestionTool(Tool[dict]):
    """Reads a CSV file from the specified path and returns a basic summary."""
    id: str = "data_ingestion_tool"
    name: str = "Data Ingestion Tool"
    description: str = (
        "Reads a CSV file from a given file path, performs basic validation, "
        "and returns a summary including shape, columns, head, info, and describe."
    )
    args_schema: type[BaseModel] = DataIngestionInputSchema
    output_schema: tuple[str, str] = (
        "dict",
        "A dictionary containing summary information: 'shape', 'columns', 'head', 'info', 'describe'"
    )

    def run(self, ctx: ToolRunContext, file_path: str) -> dict:
        """Reads the CSV and generates a summary."""
        logger.info(f"[{self.id}] Running for file: {file_path}")
        try:
            # Attempt to read the CSV
            # Add error_bad_lines=False or on_bad_lines='skip' for robustness if needed
            df = pd.read_csv(file_path)

            if df.empty:
                logger.warning(f"[{self.id}] Ingested file is empty: {file_path}")
                raise ToolHardError("The provided CSV file is empty.")

            logger.info(f"[{self.id}] Successfully read {df.shape} dataframe from {file_path}")

            # Generate summary components
            shape = df.shape
            columns = df.columns.tolist()
            head = df.head().to_string()
            describe = df.describe(include='all').to_string() # include='all' for non-numeric too

            # Capture df.info() output
            buffer = StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()

            summary = {
                "file_path": file_path,
                "shape": f"{shape[0]} rows x {shape[1]} columns",
                "columns": columns,
                "head": head,
                "info": info_str,
                "describe": describe,
            }

            logger.info(f"[{self.id}] Generated summary for {file_path}")
            return summary

        except FileNotFoundError:
            logger.error(f"[{self.id}] File not found: {file_path}")
            raise ToolHardError(f"File not found at path: {file_path}")
        except pd.errors.EmptyDataError:
            logger.error(f"[{self.id}] No data or columns found in file: {file_path}")
            raise ToolHardError("The CSV file is empty or contains no columns.")
        except pd.errors.ParserError as e:
            logger.error(f"[{self.id}] Error parsing CSV file: {file_path} - {e}")
            raise ToolHardError(f"Error parsing the CSV file: {e}. Please ensure it's a valid CSV.")
        except Exception as e:
            logger.error(f"[{self.id}] Unexpected error during ingestion for {file_path}: {e}", exc_info=True)
            raise ToolHardError(f"An unexpected error occurred during data ingestion: {e}") 