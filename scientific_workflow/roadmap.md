# Scientific Workflow Agent - Roadmap (New Workflow)

This roadmap outlines the development plan based on the refined scientific workflow defined in `newworkflow.md`.

## Milestone 1: Implement Core Scientific Tools

*(Goal: Create the backend Python classes for the new specialized tools)*

- [ ] **`ScientificSearchTool` (`scientific_workflow/tools/search_tools.py`)**
    - [ ] Define Input/Output Schemas (Input: list of query strings; Output: list of structured results with source type, title, snippet, url/doi).
    - [ ] Implement logic to query relevant scientific APIs (e.g., Semantic Scholar, PubMed - requires API keys/libs).
    - [ ] Implement logic for general web search (e.g., using Tavily or Google Search API).
    - [ ] Implement routing/prefix handling based on query input.
- [ ] **`CodeExecutionTool` (`scientific_workflow/tools/code_execution_tool.py`)**
    - [ ] Define Input/Output Schemas (Input: code string; Output: dict with stdout, stderr, list of output file paths).
    - [ ] Implement sandboxed execution environment (e.g., using Docker or `restrictedpython`). **Security is critical.**
    - [ ] Handle capture of stdout/stderr.
    - [ ] Implement mechanism to detect and return paths of files created in a designated output directory (e.g., `/plots`).
- [ ] **`CitationTool` (`scientific_workflow/tools/citation_tool.py`)**
    - [ ] Define Input/Output Schemas (Input: list of source URLs/DOIs, optional format string; Output: formatted citation string/list).
    - [ ] Implement logic using `llm_tool` (or a direct LLM API call) to fetch metadata and format citations based on input identifiers.
    - [ ] Handle potential errors during metadata fetching or formatting.
- [ ] **Tool Registration (`scientific_workflow/tools/__init__.py`)**
    - [ ] Import and register `ScientificSearchTool`, `CodeExecutionTool`, `CitationTool` in `create_tool_registry`.

## Milestone 2: Refine Existing Tools & Agent Logic

*(Goal: Adapt existing components and implement the new plan prompt)*

- [ ] **Refine `DataIngestionTool` (`scientific_workflow/tools/data_tools.py`?)**
    - [ ] Ensure output format matches `$ingestion_output` structure expected by the new plan.
    - [ ] Verify cleaning logic is robust.
- [ ] **Refine `AnalysisTool` (`scientific_workflow/tools/analysis_tools.py`)**
    - [ ] Confirm input argument (`file_path`) and output structure (`$analysis_results`) match the new plan.
- [ ] **Configure `LLMTool`**
    - [ ] Ensure it's configured to use the desired model (e.g., Gemini 1.5 Pro via appropriate API wrapper).
- [ ] **Implement New Plan Prompt (`scientific_workflow/agents/research_agent.py`)**
    - [ ] Replace the existing `plan_prompt` string in `ResearchAgent.start_analysis` with the detailed, multi-step prompt derived from `newworkflow.md`, Section 4.
    - [ ] Ensure variable names (`$ingestion_output`, `$initial_plan`, `$analysis_results`, etc.) are consistent between the prompt and tool outputs.

## Milestone 3: Frontend Adaptation & Integration Testing

*(Goal: Connect the backend changes to the UI and test the end-to-end flow)*

- [ ] **Frontend Display (`static/js/main.js`, `templates/index.html`)**
    - [ ] Adapt `displayThinkingProcess` or results display logic to handle potential new output types, especially file paths from `CodeExecutionTool` (e.g., display links or embed images if possible).
    - [ ] Ensure clear display of the final report and citations.
- [ ] **API Key Configuration**
    - [ ] Ensure all necessary API keys (LLM, Search, Scientific Databases) are correctly configured in `.env` and accessible by the relevant tools.
- [ ] **End-to-End Testing**
    - [ ] Test workflow with tabular data input.
    - [ ] Test workflow with unstructured text input.
    - [ ] Verify correct tool execution order.
    - [ ] Verify variable passing between steps.
    - [ ] Check output quality (report, citations, code results).
    - [ ] Debug any planning, execution, or tool errors.

## Milestone 4: Iteration and Refinement

*(Goal: Improve the workflow based on testing and user feedback)*

- [ ] **Prompt Engineering:** Refine LLM prompts for planning, synthesis, code generation, and report writing for better results.
- [ ] **Tool Improvement:** Enhance the capabilities or error handling of custom tools.
- [ ] **Workflow Logic:** Adjust the plan structure or conditional logic based on observed behavior.
- [ ] **UI/UX Enhancements:** Improve the display of complex information (e.g., interactive plots, better reasoning view).
- [ ] **Add Features:** Consider adding features from the old roadmap that are still relevant (e.g., Notebook system, Hypothesis testing prompts).

**(Previous Roadmap Phases are now superseded by this new plan)** 