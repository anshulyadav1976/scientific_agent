# Scientific Workflow Agent - Tool Definitions

This document outlines the custom Python tools needed for the Scientific Workflow Agent, based on the features described in `user_flowand_features.md` and the `roadmap.md`. The goal is to define the core functionality, inputs, and outputs for each tool so they can be implemented in standard Python first, before being wrapped as Portia Tools.

## Custom Tools to Build

### 1. DataIngestionTool (Existing, needs enhancement)

*   **Purpose**: Load, validate, and perform initial processing/summarization of input data.
*   **Core Functionality**:
    *   Read tabular data (CSV currently supported). 
    *   *Enhancement*: Add support for reading unstructured text (.txt files or direct input).
    *   *Enhancement*: Attempt basic data type detection (tabular vs. unstructured).
    *   *Enhancement*: Perform basic cleaning (e.g., handle common missing value representations if possible).
    *   Generate a textual summary of the data (shape, columns, head, info, describe for tabular; basic stats like word count, unique tokens for unstructured).
    *   Handle file/parsing errors gracefully.
*   **Inputs**:
    *   `file_path`: Path to the uploaded file.
    *   `data_type_hint` (Optional): User-provided hint ('tabular', 'unstructured').
*   **Outputs**:
    *   `summary`: Dictionary or string containing the data summary.
    *   `detected_type`: String indicating the detected data type ('tabular', 'unstructured', 'unknown').
    *   `data_reference` (Internal): A way to reference the loaded data (e.g., DataFrame in memory, path to cleaned file) for subsequent tools.

### 2. AnalysisTool

*   **Purpose**: Perform statistical analysis on tabular data to identify patterns.
*   **Core Functionality**:
    *   Calculate pairwise correlations between numerical columns.
    *   Identify correlations exceeding a defined threshold.
    *   Detect potential outliers/anomalies in numerical columns (e.g., using IQR, Z-score).
    *   Identify basic trends if applicable (e.g., for time-series data).
    *   (Future) Basic analysis for unstructured text (keyword frequency, etc.).
*   **Inputs**:
    *   `data_reference`: Reference to the loaded tabular data (e.g., DataFrame).
    *   `analysis_config` (Optional): Dictionary with parameters (e.g., correlation threshold, outlier method).
*   **Outputs**:
    *   `analysis_summary`: Dictionary or string detailing findings (e.g., list of strong correlations, columns with outliers and example values, identified trends).

### 3. VisualizationTool

*   **Purpose**: Generate visualizations based on data and analysis insights.
*   **Core Functionality**:
    *   Generate plots relevant to findings (e.g., correlation heatmap, distribution plots for columns with outliers, time-series plots for trends).
    *   Use libraries like Matplotlib or Seaborn.
    *   Save generated plots as image files (e.g., PNG) to a designated directory.
*   **Inputs**:
    *   `data_reference`: Reference to the loaded data.
    *   `analysis_summary`: Findings from the AnalysisTool to guide plot selection.
    *   `plot_request` (Optional): Specific plot type requested by the agent or user.
*   **Outputs**:
    *   `plot_file_paths`: List of paths to the generated image files.
    *   `plot_descriptions` (Optional): Textual descriptions of what each plot shows.

### 4. KeywordExtractionTool

*   **Purpose**: Identify key terms from text for use in external research.
*   **Core Functionality**:
    *   Process input text (e.g., user context, analysis summary).
    *   Extract relevant nouns, phrases, or named entities.
    *   Could use basic NLP (like NLTK, spaCy) or an LLM call for more sophisticated extraction.
*   **Inputs**:
    *   `text_to_analyze`: String containing the context or analysis summary.
    *   `max_keywords` (Optional): Number of keywords to return.
*   **Outputs**:
    *   `keywords`: List of extracted keyword strings.

### 5. ResearchTool

*   **Purpose**: Perform external searches based on keywords and summarize findings.
*   **Core Functionality**:
    *   Take a list of keywords.
    *   Formulate search queries.
    *   Utilize an external search API (initially Portia's `SearchTool`/Tavily, later potentially academic APIs).
    *   Process search results (e.g., extract snippets, titles, links).
    *   Optionally summarize the collective findings from multiple results.
*   **Inputs**:
    *   `keywords`: List of keyword strings.
    *   `max_results` (Optional): Number of search results to fetch/process.
*   **Outputs**:
    *   `search_findings`: List of dictionaries (each with title, snippet, link) or a summary string.

### 6. HypothesisGenerationTool

*   **Purpose**: Generate research questions or hypotheses based on available information.
*   **Core Functionality**:
    *   Synthesize information from data summary, analysis findings, and external research.
    *   Construct a prompt for an LLM.
    *   Call the LLM (via Portia's `LLMTool` or direct API) to generate hypotheses/questions.
    *   Ensure questions are grounded in the provided evidence.
*   **Inputs**:
    *   `data_summary`: Text/Dict from DataIngestionTool.
    *   `analysis_summary`: Text/Dict from AnalysisTool.
    *   `research_findings`: Text/List from ResearchTool.
*   **Outputs**:
    *   `hypotheses`: List of generated hypothesis/question strings.

## Portia Built-in Tools to Utilize

*   **`SearchTool` (via `open_source_tool_registry`)**: 
    *   Will be used internally by the `ResearchTool` to perform web searches (likely using Tavily based on SDK defaults).
*   **`LLMTool` (via `open_source_tool_registry`)**: 
    *   Can be used by `LLMSummarizationTool` (if built), `KeywordExtractionTool` (potentially), and `HypothesisGenerationTool` to interact with configured LLMs for text generation tasks. 