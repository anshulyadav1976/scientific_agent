# AI Agent for Scientific Workflows: Feature Overview & User Flow (v2)

## Core Concept

A conversational AI agent designed to assist with scientific research workflows. The user interacts via a chat interface, providing input files and prompts. The agent executes a multi-step plan involving data analysis, literature/web searches, code execution (for visualization/simulation), citation management, and report generation. Key agent outputs (summaries, results, plots, final report) are presented directly in the chat, while the detailed step-by-step reasoning and tool execution logs are available in a collapsible side panel.

## Core Features

1.  **Data Input & Initial Analysis:**
    *   Supports uploading various file types (CSV, TXT initially).
    *   Performs automatic data cleaning (e.g., numeric conversion for CSVs).
    *   Detects data type (tabular, unstructured) using `data_ingestion_tool`.
    *   Generates an initial summary of the ingested data.

2.  **LLM-Powered Planning & Reasoning:**
    *   Uses a powerful LLM (e.g., Gemini 1.5 Pro) to generate a research plan based on user prompt and initial data assessment.
    *   LLM synthesizes information from multiple sources (data analysis, scientific literature, web search).
    *   LLM generates Python code for analysis, visualization, or simulation tasks.
    *   LLM generates final reports and summaries.

3.  **Integrated Scientific Toolkit:**
    *   **Statistical Analysis:** Performs correlations, outlier detection, descriptive statistics (`analysis_tool`).
    *   **Targeted Research:** Queries scientific databases (PubMed, ArXiv, etc.) and general web search (`scientific_search_tool`). Prioritizes scientific sources.
    *   **Code Execution:** Runs LLM-generated Python code in a sandboxed environment (`code_execution_tool`) for calculations, simulations, and generating visualizations.
    *   **Citation Management:** Automatically identifies sources used and formats citations (`citation_tool`).

4.  **Interactive Chat Interface:**
    *   Primary user interaction occurs through a familiar chat UI.
    *   Agent presents key findings, summaries, requests for clarification, and final results as chat messages.
    *   Supports interactive elements like displaying generated plots or data tables directly within the chat flow.
    *   Handles user responses for clarifications or follow-up prompts.

5.  **Transparent Reasoning Panel:**
    *   A collapsible side panel displays the detailed execution trace (Portia's "Thinking Process").
    *   Shows each step planned, the tool used, input arguments, raw output, and status.
    *   Allows users to inspect the agent's detailed work without cluttering the main chat conversation.

6.  **Reproducible Output:**
    *   Provides formatted citations for sources used in the synthesis/report.
    *   Includes results from executed code (stdout, stderr, file paths to plots).
    *   Generates a final, structured report in Markdown.
    *   (Future) Export functionality for the chat log and/or reasoning trace.

## User Flow Example

1.  **Start Session:** User opens the web interface.
2.  **Initiate Task:**
    *   User uploads a file (e.g., `research_data.csv` or `literature_notes.txt`).
    *   User provides an initial prompt in the chat input (e.g., "Analyze this data to find correlations related to gene X and summarize recent relevant literature.").
3.  **Agent Processing (Steps shown in Side Panel):**
    *   Agent uses `data_ingestion_tool` (Step 1a).
    *   Agent uses `llm_tool` to create initial plan based on prompt and ingested data (Step 1b). Side panel shows the planned steps.
    *   *Side Panel Updates:* Shows "Step 1: Ingest & Initial Assessment - Executed".
    *   Agent executes `analysis_tool` (Step 2, if required by plan).
    *   *Side Panel Updates:* Shows "Step 2: Data Analysis - Executed" with input/output details.
    *   Agent uses `llm_tool` to generate search queries (Step 3).
    *   *Side Panel Updates:* Shows "Step 3: Generate Search Queries - Executed".
    *   Agent executes `scientific_search_tool` (Step 4).
    *   *Side Panel Updates:* Shows "Step 4: Execute Search - Executed".
    *   Agent uses `llm_tool` to synthesize findings and list sources (Step 5).
    *   *Side Panel Updates:* Shows "Step 5: Synthesize Findings - Executed".
    *   Agent uses `llm_tool` to generate Python code for visualization (Step 6).
    *   *Side Panel Updates:* Shows "Step 6: Generate Code - Executed".
    *   Agent executes `code_execution_tool` (Step 7).
    *   *Side Panel Updates:* Shows "Step 7: Execute Code - Executed".
    *   Agent executes `citation_tool` (Step 8).
    *   *Side Panel Updates:* Shows "Step 8: Format Citations - Executed".
    *   Agent uses `llm_tool` to create the final report (Step 9).
    *   *Side Panel Updates:* Shows "Step 9: Create Final Report - Executed".
4.  **Agent Output (Displayed in Main Chat):**
    *   Agent might post intermediate summaries or key findings (e.g., after synthesis).
    *   Agent displays visualizations generated by the code execution step (e.g., embedding the plot image `/plots/figure1.png`).
    *   Agent displays the final report (`$final_report`) as a formatted Markdown message.
    *   Agent displays the formatted citations (`$formatted_citations`).
5.  **User Interaction:**
    *   User reviews the report, plots, and citations in the chat.
    *   User can expand the side panel at any time to see the detailed execution steps.
    *   User can ask follow-up questions or provide new prompts in the chat input to refine the research or start a new analysis based on the results.
    *   If the agent requires clarification during the process, the chat interface will prompt the user, halting execution until a response is provided.


