"""Tools for Data Ingestion and Cleaning"""

import logging
import pandas as pd
from io import StringIO
from pathlib import Path # Use pathlib for path operations
import chardet # For detecting text encoding

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional # For type hint

from portia.tool import Tool, ToolRunContext, ToolHardError

logger = logging.getLogger(__name__)

# --- DataIngestionTool --- #

class DataIngestionInputSchema(BaseModel):
    """Input schema for the DataIngestionTool."""
    file_path: str = Field(..., description="The absolute path to the file to ingest.")
    # Added optional hint
    data_type_hint: Optional[Literal['tabular', 'unstructured']] = Field(
        default=None,
        description="Optional hint from the user about the data type."
    )

    # Add validation if needed, e.g., check if file_path exists
    # @validator('file_path')
    # def file_path_must_exist(cls, v):
    #     if not Path(v).is_file():
    #         raise ValueError(f'File not found at path: {v}')
    #     return v

class DataIngestionTool(Tool[dict]):
    """Reads a CSV or TXT file from the specified path, detects type, and returns a basic summary."""
    id: str = "data_ingestion_tool"
    name: str = "Data Ingestion Tool"
    description: str = (
        "Reads a CSV or TXT file from a given file path. Detects if the content is likely tabular or unstructured text. "
        "Performs basic validation and returns a summary including shape/columns/head/info/describe for tabular data, "
        "or line count/word count/snippet for unstructured text. Also returns the detected data type."
    )
    args_schema: type[BaseModel] = DataIngestionInputSchema
    # Updated output schema description
    output_schema: tuple[str, str] = (
        "dict",
        "A dictionary containing summary information (structure depends on detected type) "
        "and a 'detected_type' ('tabular' or 'unstructured')."
    )

    def _detect_encoding(self, file_path: Path) -> str:
        """Attempt to detect the file encoding."""
        try:
            with file_path.open('rb') as f:
                raw_data = f.read(5000) # Read first 5KB to guess encoding
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            logger.debug(f"Detected encoding: {encoding} with confidence {confidence} for {file_path.name}")
            # Use utf-8 as fallback if detection is uncertain or fails
            return encoding if encoding and confidence > 0.7 else 'utf-8'
        except Exception as e:
            logger.warning(f"Could not detect encoding for {file_path.name}, falling back to utf-8. Error: {e}")
            return 'utf-8'

    def run(self, ctx: ToolRunContext, file_path: str, data_type_hint: Optional[str] = None) -> dict:
        """Reads the file, detects type, and generates a summary."""
        logger.info(f"[{self.id}] Running for file: {file_path} with hint: {data_type_hint}")
        path = Path(file_path)
        detected_type = "unknown"
        summary = {}
        final_output = {
            "file_path": file_path,
            "summary": summary,
            "detected_type": detected_type
        }

        if not path.is_file():
            logger.error(f"[{self.id}] File not found: {file_path}")
            raise ToolHardError(f"File not found at path: {file_path}")

        # 1. Determine Data Type
        file_ext = path.suffix.lower()
        if data_type_hint:
            detected_type = data_type_hint
            logger.info(f"[{self.id}] Using provided hint: '{detected_type}'")
        elif file_ext == '.csv':
            detected_type = 'tabular'
            logger.info(f"[{self.id}] Detected type 'tabular' based on .csv extension.")
        elif file_ext == '.txt':
            detected_type = 'unstructured'
            logger.info(f"[{self.id}] Detected type 'unstructured' based on .txt extension.")
        else:
            # Basic fallback: Try reading as CSV, if it fails assume text? Risky.
            # For now, stick to extensions or hint.
            logger.warning(f"[{self.id}] Could not determine type from extension '{file_ext}' or hint. Cannot process.")
            final_output["detected_type"] = "unknown"
            # Raise error or return unknown? Returning for now.
            # raise ToolHardError(f"Could not determine file type for {path.name}. Please use .csv or .txt or provide a hint.")
            return final_output

        # 2. Process Based on Type
        try:
            if detected_type == 'tabular':
                # --- Tabular Processing --- 
                encoding = self._detect_encoding(path)
                logger.info(f"[{self.id}] Reading CSV with encoding: {encoding}")
                try:
                    # Try standard separator first
                    df = pd.read_csv(path, encoding=encoding)
                except pd.errors.ParserError:
                    logger.warning(f"[{self.id}] CSV parsing failed with standard separator, trying common alternatives (;, \t)")
                    try:
                        df = pd.read_csv(path, encoding=encoding, sep=';')
                    except pd.errors.ParserError:
                         df = pd.read_csv(path, encoding=encoding, sep='\t')
                
                if df.empty:
                    logger.warning(f"[{self.id}] Ingested CSV file is empty: {file_path}")
                    raise ToolHardError("The provided CSV file is empty.")

                logger.info(f"[{self.id}] Successfully read {df.shape} dataframe from {file_path}")

                # Generate summary components
                shape = df.shape
                columns = df.columns.tolist()
                head = df.head().to_string()
                describe = df.describe(include='all').to_string()
                buffer = StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()

                summary = {
                    "data_format": "tabular",
                    "shape": f"{shape[0]} rows x {shape[1]} columns",
                    "columns": columns,
                    "head": head,
                    "info": info_str,
                    "describe": describe,
                }
                final_output["detected_type"] = "tabular"

            elif detected_type == 'unstructured':
                # --- Unstructured Text Processing --- 
                encoding = self._detect_encoding(path)
                logger.info(f"[{self.id}] Reading TXT with encoding: {encoding}")
                content = path.read_text(encoding=encoding)
                lines = content.splitlines()
                line_count = len(lines)
                word_count = len(content.split())
                char_count = len(content)
                snippet_length = 500 # Show first 500 chars as snippet
                snippet = content[:snippet_length] + ("..." if char_count > snippet_length else "")
                
                if char_count == 0:
                     logger.warning(f"[{self.id}] Ingested TXT file is empty: {file_path}")
                     raise ToolHardError("The provided TXT file is empty.")

                summary = {
                    "data_format": "unstructured",
                    "line_count": line_count,
                    "word_count": word_count,
                    "character_count": char_count,
                    "snippet": snippet
                }
                final_output["detected_type"] = "unstructured"
            
            final_output["summary"] = summary
            logger.info(f"[{self.id}] Generated {detected_type} summary for {file_path}")
            return final_output

        except FileNotFoundError: # Should be caught earlier, but defensive check
            logger.error(f"[{self.id}] File not found error during processing: {file_path}")
            raise ToolHardError(f"File not found at path: {file_path}")
        except UnicodeDecodeError as ude:
            logger.error(f"[{self.id}] Encoding error reading file {file_path} with detected encoding {encoding}: {ude}")
            raise ToolHardError(f"Failed to decode file {path.name} with detected encoding {encoding}. Try saving as UTF-8.")
        except pd.errors.EmptyDataError:
            logger.error(f"[{self.id}] No data or columns found in CSV file: {file_path}")
            raise ToolHardError("The CSV file is empty or contains no columns.")
        except pd.errors.ParserError as e:
            logger.error(f"[{self.id}] Error parsing CSV file: {file_path} - {e}")
            raise ToolHardError(f"Error parsing the CSV file: {e}. Please ensure it's a valid CSV.")
        except Exception as e:
            logger.error(f"[{self.id}] Unexpected error during ingestion for {file_path}: {e}", exc_info=True)
            raise ToolHardError(f"An unexpected error occurred during data ingestion: {e}") 