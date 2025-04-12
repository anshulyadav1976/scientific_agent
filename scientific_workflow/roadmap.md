# Scientific Workflow Agent - Interactive UI Roadmap

This roadmap outlines the phased development plan to transform the agent into an interactive, step-by-step chat application.

## Phase 1: Backend API & Control Flow Refactoring

*(Goal: Change the backend to support step-by-step execution initiated by the frontend)*

- [x] **Modify `ResearchAgent.start_analysis`**:
    - [x] Change method to only generate Plan (`portia.plan()`).
    - [x] Create PlanRun (`portia.create_plan_run()`).
    - [x] **Remove** `portia.run()` or `portia.resume()` call.
    - [x] Return initial `PlanRun` object (state: `NOT_STARTED`).
- [x] **Modify `/upload` Endpoint (`main.py`)**:
    - [x] Call the refactored `agent.start_analysis`.
    - [x] Return only `run_id` and initial status/message.
- [x] **Implement `/resume/{run_id}` Endpoint (POST) (`main.py`)**:
    - [x] Accepts `run_id` as path parameter.
    - [x] Calls `agent.portia.resume(plan_run_id=run_id)`.
    - [x] Returns latest `PlanRun` state/status after resume completes.
- [X] `/clarify` Endpoint (POST)
  - [X] Endpoint at `/clarify/{run_id}`.
  - [X] Accepts `run_id` in path and JSON body with `clarification_id` and `response`.
  - [X] Finds the specific `Clarification` object within the `PlanRun`.
  - [X] Calls `agent.portia.resolve_clarification(clarification, response, plan_run)`.
  - [X] Returns a simple success JSON response.
- [ ] **Modify `/status/{run_id}` Endpoint (`main.py`)**:
    - [ ] Ensure it accurately reflects current `PlanRun` state (`NOT_STARTED`, `IN_PROGRESS`, `NEED_CLARIFICATION`, `COMPLETE`, `FAILED`).
    - [ ] Ensure `thinking_process` reflects steps executed *so far* based on `current_step_index`.
    - [ ] Ensure `clarification` field is correctly populated when state is `NEED_CLARIFICATION`.
    - [ ] Ensure final `formatted_summary`/`raw_output`/`error` fields are correct for terminal states.

## Phase 2: Plan Modification for Interaction

*(Goal: Insert deliberate pause points into the agent's workflow)*

- [ ] **Create `ClarificationTool` (`tools/interaction_tools.py`)**:
    - [ ] Implement simple tool that raises `InputClarification` with a configurable prompt.
- [ ] **Update Tool Registry (`tools/__init__.py`)**: Register `ClarificationTool`.
- [ ] **Update `plan_prompt` (`research_agent.py`)**: 
    - [ ] Add step *after* ingestion (Step 1) calling `ClarificationTool` ("Proceed with analysis?").
    - [ ] Add step *after* analysis (Step 3) calling `ClarificationTool` ("Proceed with LLM summary?").
    - [ ] Add step *after* LLM summary (Step 4) calling `ClarificationTool` ("Specify follow-up or type 'done'?").
- [ ] **Refine Step Numbering/Naming**: Ensure prompt and output variables are clear.

## Phase 3: Frontend Revamp - Core Chat Interface

*(Goal: Replace the current static display with a dynamic chat log)*

- [ ] **HTML (`index.html`)**: Redesign layout with chat log container and user input area.
- [ ] **CSS (`static/css/main.css`)**: Style chat messages (user vs. agent).
- [ ] **JavaScript (`static/js/main.js`) - State Management**: Add variables for `run_id` and interaction state (e.g., `WAITING_FOR_AGENT`, `WAITING_FOR_USER_CLARIFICATION`).
- [ ] **JavaScript - Initial Request**: Handle initial form submission:
    - [ ] Call `/upload`.
    - [ ] On success, get `run_id`, display "Starting...", call `/resume/{run_id}`.
- [ ] **JavaScript - User Input Handling**: Handle user messages/form submission:
    - [ ] If state is `WAITING_FOR_USER_CLARIFICATION`, call `/clarify`.
    - [ ] On `/clarify` success, call `/resume/{run_id}`.
- [ ] **JavaScript - Polling (`/status`)**: Implement polling loop driven by interaction state.
- [ ] **JavaScript - Message Display**: Append new thinking steps/outputs from `/status` to chat log.
- [ ] **JavaScript - State Transitions**: Update interaction state based on `/status` response (`NEED_CLARIFICATION`, `COMPLETE`, `FAILED`) and stop/start polling appropriately.

## Phase 4: Frontend Revamp - Detailed Output & Reasoning View

*(Goal: Show the full details within the chat and provide optional reasoning view)*

- [ ] **JavaScript - Detailed Output**: Enhance agent message display to nicely format the *full* structured output from tools like `AnalysisTool` (received via `thinking_process` in `/status`).
- [ ] **JavaScript - Final Result**: Display `formatted_summary` from `/status` as final agent message on `COMPLETE`.
- [ ] **(Optional) Reasoning View**: Add UI element (button/link) to toggle visibility of detailed step reasoning (tool calls, raw I/O) possibly derived from `thinking_process`.

## Phase 5: Implement Follow-up Loop

*(Goal: Allow user to act on LLM suggestions or provide new analysis prompts)*

- [ ] **Refine Clarification Prompt (Step 4 in Plan)**: Ensure it clearly asks for follow-up query or "done".
- [ ] **Backend Logic (`/clarify` & `/resume`)**: 
    - [ ] Handle "done" response to end the run.
    - [ ] Handle new analysis request:
        - [ ] **Approach A (Pre-defined Tools)**: Parse request, map to existing/new tool (e.g., `RegressionTool`), potentially add step to plan dynamically, call resume. (Requires building specific tools).
        - [ ] **Approach B (Code Interpreter Tool)**: Implement custom tool to call external code interpreter API. (Deferred complexity).
- [ ] **Frontend Loop**: Ensure chat interface allows multiple follow-up interactions.

## Feature: 1. Data Input and Classification

- [x] **Upload**: Support for uploading tabular data (CSV) **and unstructured text (TXT)**. (`/upload` endpoint, `index.html` form)
- [x] **Upload**: Allow `.txt` file uploads (`accept` attribute updated).
- [x] **Backend Validation**: `/upload` endpoint accepts `text/plain`.
- [x] **Data Type Hint**: Added radio buttons for user hint. (`index.html`)
- [x] **Backend Hint Handling**: `/upload` endpoint receives hint and passes to agent.
- [x] **Tool Logic**: `DataIngestionTool` handles both CSV and TXT, performs basic type detection, and basic summarization for both.
- [x] **Tool Output**: `DataIngestionTool` includes `detected_type` in output.
- [x] **Frontend Display**: Basic display logic handles both tabular and unstructured summaries.
- [ ] **Classification**: Automatic data type detection (beyond extension).
- [ ] **Classification**: Allow user override/confirmation of data type (UI only provides hint currently).

## Feature: 2. Context-Aware Processing

- [x] **Context Input**: Prompt user for initial research context via text area. (`index.html` form)
- [x] **Tabular - Initial Analysis**: Perform basic local Python-based analysis (current `DataIngestionTool` does this). 
    - [x] Read data (pandas).
    - [x] Basic summary (shape, columns, head, info, describe).
- [x] **Unstructured - Initial Analysis**: 
    - [x] Implement basic processing logic for unstructured text (word/line count, snippet) in `DataIngestionTool`.
- [x] **LLM Summarization (Conditional)**: 
    - [x] Agent plan includes conditional step to call `llm_tool` for summarizing unstructured text snippet from ingestion step.

## Feature: 3. Memory and Notebook System

- [ ] **Session Memory**: Store user preferences, past steps, decisions (requires defining what to store and how).
- [ ] **Notebook Backend**: Data structure/storage for notebook entries (associated with a PlanRun or session).
- [ ] **Notebook Logging**: 
    - [x] Agent logs basic execution steps (captured partially by `thinking_process` in `/status`).
    - [ ] Log raw inputs.
    - [ ] Log agent's reasoning chains (requires planner/agent introspection).
    - [ ] Log analysis outputs (text, visualizations).
    - [ ] Log generated hypotheses.
    - [ ] Log rejected ideas/paths (requires more advanced agent logic).
- [ ] **Notebook Frontend View**: 
    - [ ] Dedicated UI section to display the structured notebook content.
    - [ ] Display raw inputs.
    - [ ] Display reasoning/steps (improve on current `thinking_process` display).
    - [ ] Display analysis outputs (text, embedded visualizations).
    - [ ] Display hypotheses/rejected ideas.
- [ ] **User Notes**: Implement side canvas/input area for user annotations linked to notebook entries.
- [ ] **Export**: Functionality to export the notebook view (e.g., Markdown, PDF).

## Feature: 4. In-Depth Analysis and Exploration (Tabular Data Focus)

- [x] **AnalysisTool (Python Implementation)**:
    - [x] Define Input/Output Schemas (Input: file_path or data reference; Output: structured dictionary of results).
    - [x] Implement Pearson Correlation calculation (numerical columns).
    - [x] Implement filtering for strong correlations (e.g., > |0.7| threshold).
    - [x] Implement IQR Outlier Detection (numerical columns).
    - [x] Implement Descriptive Statistics calculation (Mean, median, std, min/max, range, skewness, kurtosis, missing %, unique counts, zero counts, constant columns, dominant categories, high cardinality).
    - [x] Structure output into a dictionary (e.g., `{"correlations": [...], "outliers": {...}, "descriptive_stats": {...}}`).
- [x] **Agent Integration (AnalysisTool)**: Update agent plan prompt to conditionally call `AnalysisTool` if data type is 'tabular', passing necessary input. Name output (e.g., `$analysis_results`).
- [ ] **LLM Summary & Suggestion Step**: 
    - [ ] Add a new step to the agent plan prompt, conditional on `AnalysisTool` completion.
    - [ ] This step uses the built-in `LLMTool`.
    - [ ] Pass the structured `$analysis_results` to the `LLMTool`.
    - [ ] Prompt the LLM to summarize findings, add insights, and suggest next analysis steps.
    - [ ] Name the output (e.g., `$llm_summary_suggestions`).
- [x] **Frontend Display (Analysis)**:
    - [x] Display structured Python analysis results (`$analysis_results`) in the 'Thinking Process' or a dedicated section.
    - [ ] Display LLM summary/insights/suggestions (`$llm_summary_suggestions`) in the final 'Results' or a dedicated section.
- [ ] **Analysis Clarifications**: Implement agent logic/UI to handle user follow-up requests based on LLM suggestions (Deferred - requires Clarification Loop).
- [ ] **Code Interpreter Integration**: Implement follow-up analysis execution using a code interpreter approach (Deferred).

## Feature: 5. Visualization (Insight-Driven)

- [ ] **VisualizationTool (Python Implementation)**:
    - [ ] Define Input/Output Schemas (Input: data reference, analysis results; Output: list of plot file paths).
    - [ ] Generate plots based on `$analysis_results` (e.g., correlation heatmap, outlier boxplots).
    - [ ] Save plots to disk.
- [ ] **Agent Integration (VisualizationTool)**: Update agent plan to conditionally call `VisualizationTool` after `AnalysisTool`.
- [ ] **Frontend Display (Plots)**: Embed generated plots in the UI (e.g., Notebook view).

## Feature: 6. External Research Integration

- [ ] **Keyword Extraction**: Logic to identify relevant keywords from data/analysis for external search.
- [ ] **External Search Tool**: 
    - [x] Integrate basic web search (e.g., via Portia's `SearchTool`).
    - [ ] (Future) Integrate with specific academic APIs (Semantic Scholar, PubMed, etc.).
- [ ] **Information Gathering**: Agent processes search results to gather relevant datasets/references.
- [ ] **Hypothesis Generation**: 
    - [ ] Tool/Logic using LLM to generate hypotheses based on internal analysis *and* external research findings.
- [ ] **Refinement Q&A**: Agent asks follow-up questions to refine research scope based on external findings.

## Feature: 7. Interaction & User Experience

- [x] **Basic UI**: File upload, prompt input, status display, basic results/thinking display.
- [x] **Clarification Framework (Backend)**: Portia SDK supports clarifications.
- [ ] **Clarification Framework (Frontend)**: Implement UI and JS logic for submitting clarification responses.
- [x] **Transparency**: Display thinking process/steps (basic implementation exists).
    - [ ] Enhance thinking process display (more detail, better formatting).
- [ ] **Error Handling**: Improve display of errors on the frontend.

## Cross-Cutting Concerns & Future Enhancements

- [x] **Basic Configuration**: Load API keys from `.env`.
- [x] **Persistence**: Use Disk storage for runs/plans.
- [ ] **LLM Configuration**: Allow user selection/configuration of LLMs.
- [ ] **Vector Storage**: Implement for context management/memory.
- [ ] **Testing**: Add unit/integration tests.
- [ ] **Deployment**: Dockerize.

**Step 5: Frontend - Basic Run Lifecycle (JS)** `[Complete]`
  - [X] Call `/upload` on form submission.
  - [X] Store returned `run_id`.
  - [X] Call `/resume/{run_id}` after successful upload.
  - [X] Implement `startPolling(run_id)` called after successful `/resume`.
  - [X] Periodically poll `/status/{run_id}` within `checkStatus`.
  - [X] Display status, hide/show relevant areas (status, results, thinking, clarification).
  - [X] Implement clarification form submission (`/clarify/{run_id}`).
  - [X] Call `/resume/{run_id}` after successful clarification.
- **Step 6: Modify `/status/{run_id}` Endpoint** `[Incomplete]`
  - [ ] Ensure it accurately reflects current `PlanRun` state (`NOT_STARTED`, `IN_PROGRESS`, `NEED_CLARIFICATION`, `COMPLETE`, `FAILED`).
  - [ ] Include the full `thinking_process` (list of `StepRun` objects or similar representation) in the response. 