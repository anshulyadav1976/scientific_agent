# Deep Research Agents (CSV & Neo4j) with Simulated Portia Planning

This project demonstrates two deep research agent systems:

1.  **CSV Research Agent:** Conducts research on CSV data locally, using LLMs for summary generation, insight extraction, and query intent classification. It simulates Portia planning for its research steps.
2.  **Graph Database Research Agent:** Translates natural language research questions into Cypher queries for execution against a Neo4j graph database, leveraging LLMs for generation, validation, and error handling. It also simulates Portia planning.

Both systems showcase how Portia AI could orchestrate complex, multi-step deep research workflows involving LLMs and interaction with external data sources/databases.

## Features

### CSV Research Agent (`portia_csv_analyzer.py`)

*   **Local Data Handling:** Parses CSV data directly within the script, keeping raw data private during the research process.
*   **LLM-Powered Insights:** Uses LLMs (e.g., OpenAI GPT) for:
    *   Generating concise data summaries to inform research direction.
    *   Identifying user query intent for focused investigation.
*   **Local Analysis:** Identifies column types and basic data characteristics (missing values, numeric ranges, potential categories) locally as part of the research.
*   **(Simulated) Portia Planning:** Visualizes the research workflow (e.g., Parse Data, Generate Summary, Identify Types, Process Query) as a simulated Portia plan.

### Graph Database Research Agent (`cypher_query_system_v3.py`)

*   **Natural Language Research Questions:** Ask complex research questions about your Neo4j graph data in plain English.
*   **Cypher Query Generation:** Uses LLMs (configurable, defaults to OpenAI GPT models) to generate appropriate Cypher queries for graph exploration.
*   **Schema Validation:** Checks user terminology against the database schema to identify potential mismatches before formulating queries.
*   **Query Verification:** Uses an LLM to verify if the generated query accurately reflects the user's research question.
*   **Error Handling & Correction:** Attempts to automatically correct failed queries or reformulate them using LLMs to ensure research continuity.
*   **Alternative Suggestions:** Provides fallback suggestions if direct query attempts fail.
*   **(Simulated) Portia Planning:** Visualizes the research execution flow as a simulated Portia plan, showing the status of each conceptual step (Validation, Generation, Verification, Execution, Correction, Response).
*   **Result Visualization:** Attempts to visualize successful graph query results using Cytoscape.js (opens an HTML file in your browser).

## How it Leverages Portia AI (Simulation)

Both systems in this project are designed to *demonstrate* how Portia AI could be used to orchestrate complex, multi-step workflows involving LLMs for deep research tasks.

Instead of deeply integrating Portia's execution engine for this version, the scripts:

1.  **Define Conceptual Plans:** They outline the logical steps involved in processing a research request (e.g., parsing, summarizing, generating queries, verifying, executing, etc.).
2.  **Execute the Workflows:** They run the actual research and query logic using LLM wrappers and data connections (local CSV parsing or Neo4j).
3.  **Simulate Plan Status:** Based on the success or failure points of the actual workflow execution, they determine the status (`COMPLETE`, `FAILED`, `Skipped`) for each step in the *conceptual* Portia plan.
4.  **Visualize the Simulated Plan:** They print a step-by-step breakdown of the simulated plan and its status, mimicking the kind of observability Portia provides for complex agentic processes.

This approach allows us to showcase the *structure* and *potential* of using Portia for such tasks within the hackathon context, showing how Portia could manage the transitions, inputs, and outputs between different agents or tools (LLMs, database executors, local functions) in a real implementation.

## Prerequisites

*   Python 3.8+
*   Access to a Neo4j database instance (for the Graph Database Research Agent).
*   API Key for an LLM provider (e.g., OpenAI). Both systems default to OpenAI but can be configured.
*   `pip` for installing dependencies.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Install Dependencies:**
    Create a `requirements.txt` file (or use the one provided if available) with the following content:
    ```txt
    # Common
    python-dotenv
    portia-ai  # Recommended, though handled gracefully if missing
    httpx
    pydantic
    openai
    google-generativeai

    # For Cypher System
    py2neo

    # For CSV Analyzer
    pandas # Though not explicitly imported, often useful for CSV tasks
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `portia-ai` is included as recommended. The scripts include error handling if not installed, but having it allows the Portia client initialization checks to pass.)*

3.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory and add your configuration:
    ```dotenv
    # === LLM Configuration (Used by both systems) ===
    OPENAI_API_KEY=your_openai_api_key
    # Optional: Specify models if different from default (used by Cypher system)
    # AGENT1_MODEL=gpt-4o-mini
    # AGENT2_MODEL=gpt-4o
    # Optional: Specify model for CSV Analyzer (if different from OpenAI default)
    # OPENAI_MODEL=gpt-4 # Or your preferred model

    # If using Google Gemini:
    # GOOGLE_API_KEY=your_google_api_key
    # You might need to adjust provider/model settings within the Python scripts if using Google

    # === Neo4j Connection (Used by Cypher system) ===
    NEO4J_URI=neo4j://localhost:7687 # Or your bolt/neo4j URI
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_neo4j_password
    ```
    *   Modify LLM model settings directly in the Python scripts if needed, especially if using different models/providers for each system.

4.  **Prompts (for Cypher System):** Ensure the `prompts` directory exists and contains the necessary `.txt` files referenced in `cypher_query_system_v3.py` (e.g., `schema_validation_prompt.txt`, `cypher_generation_prompt_common.txt`, etc.).

5.  **Visualization Directory (for Cypher System):** Create a directory for the visualization output:
    ```bash
    mkdir graph_vis
    ```

## How to Use

### Running the CSV Research Agent

Provide the path to your CSV file as a command-line argument:

```bash
python portia_csv_analyzer.py /path/to/your/data.csv
```

The script will:
1.  Parse the CSV data locally.
2.  Perform initial analysis and research (summary, column types, insights) using local logic and LLM calls.
3.  Print the initial findings.
4.  Enter an interactive mode where you can ask research questions about the CSV.
5.  For each question, it will print the **Simulated Portia Plan** and the final response.

### Running the Graph Database Research Agent

Run the script directly from your terminal:

```bash
python cypher_query_system_v3.py
```

The script will:
1.  Connect to Neo4j and extract the schema.
2.  Attempt to identify database sources from node IDs (using an LLM call).
3.  Process the example research questions defined in the `if __name__ == "__main__":` block one by one.
4.  For each question:
    *   Print the natural language research question.
    *   Execute the workflow (validation, generation, verification, execution, error handling).
    *   Print the final natural language response generated by the LLM based on the research results.
    *   Print the Cypher query that was successfully executed (or the last one attempted if failed).
    *   If successful and results were found, attempt to generate and open an HTML file (`graph_vis/cypher_results_viz.html`) with an interactive graph visualization.
    *   Print the **Simulated Portia Plan** showing the status of each conceptual step.
    *   Print details of the workflow execution for transparency.

Modify the `queries_to_run` list in the `if __name__ == "__main__":` block of `cypher_query_system_v3.py` to test your own natural language research questions against your Neo4j database. 