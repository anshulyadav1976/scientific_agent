# Scientific Workflow Agent - Revised Roadmap (Interactive Flow Focus)

This roadmap outlines an iterative development plan prioritizing the stateful, asynchronous, and interactive user experience defined in `user_flowand_features.md`. We will establish the core UI/backend communication loop early and then incrementally add the scientific workflow capabilities defined in `newworkflow.md`.

## Milestone 1: Stateful Backend & Basic Polling UI

*(Goal: Establish the core asynchronous workflow: user starts a run, backend creates it, frontend resumes it and polls for basic status updates.)*

-   [ ] **Agent - Plan & Create Run (`scientific_workflow/agents/research_agent.py`)**:
    -   [ ] **Modify `__init__`**: Ensure default Portia config and a minimal `ToolRegistry` (e.g., just `LLMTool` for now) are loaded.
    -   [ ] **Modify `start_analysis`**:
        -   Takes `user_prompt` and optional `file_path`.
        -   Implements a *basic* plan prompt (e.g., just Step 1 from `newworkflow.md`: Ingest + LLM Planning, *or even simpler* just an LLM rephrase step for initial testing).
        -   Uses `self.portia.plan()` to generate the plan.
        -   Uses `self.portia.create_plan_run(plan)` to create the run **without executing it**.
        -   Returns the `plan_run.id` string.
-   [ ] **Backend - Enable Core Endpoints (`scientific_workflow/main.py`)**:
    -   [ ] **Modify `/upload`**:
        -   Calls the updated `agent.start_analysis`.
        -   Returns `JSONResponse(content={"run_id": plan_run_id})`.
    -   [ ] **Re-enable `/resume/{run_id}`**:
        -   Calls `agent.portia.resume(plan_run_id=run_id)` (this will block until completion or clarification in the simple case, which is fine for now).
        -   Returns a simple success/status message.
    -   [ ] **Re-enable `/status/{run_id}`**:
        -   Calls `agent.portia.storage.get_plan_run(run_id)`.
        -   Returns basic run info: `JSONResponse(content={"run_id": ..., "status": ..., "final_output": ..., "error": ...})`. (No thinking process or clarifications needed *yet*).
    -   [ ] Comment out `/clarify` for now.
-   [ ] **Frontend - Basic Async Flow (`static/js/main.js`)**:
    -   [ ] **Modify Form Submit**:
        -   Call `/upload` to get `run_id`.
        -   On success, call `/resume/{run_id}`.
        -   On success from `/resume`, *start* polling `/status/{run_id}` using `setInterval`.
    -   [ ] **Implement Basic `checkStatus`**:
        -   Fetch `/status/{run_id}`.
        -   Display the raw `data.status` in the status area.
        -   If status is `COMPLETE` or `FAILED`, display `data.final_output` or `data.error` in the main chat area (using `addChatMessage`) and *stop* polling (`clearInterval`).
    -   [ ] Remove logic for clarifications and thinking process for now.
-   [ ] **Testing**: Verify that submitting a prompt: creates a run, resumes it, the UI polls for status, displays "IN_PROGRESS", and finally displays the simple LLM response (or error) in the chat when the run completes.

## Milestone 2: Thinking Process Panel & Data Ingestion

*(Goal: Display the planned steps and their status in the side panel and integrate the first real tool: Data Ingestion.)*

-   [ ] **Backend - Enhance `/status` (`main.py`)**:
    -   [ ] Fetch the `Plan` associated with the `PlanRun` using `agent.portia.storage.get_plan()`.
    -   [ ] Add `thinking_process` data to the response (iterate through plan steps, show description, status based on `plan_run.current_step_index` and `plan_run.state`). Include basic step output if available in `plan_run.outputs.step_outputs`.
-   [ ] **Frontend - Implement Thinking Panel (`static/js/main.js`, `templates/index.html`, `static/css/style.css`)**:
    -   [ ] Add the collapsible side panel element to `index.html`.
    -   [ ] Implement `displayThinkingProcess` function in JS to render the `thinking_process` data from the `/status` response into the panel.
    -   [ ] Call `displayThinkingProcess` within `checkStatus`.
-   [ ] **Tool - Data Ingestion (`scientific_workflow/tools/data_tools.py`, `tools/__init__.py`)**:
    -   [ ] Implement/Refine `DataIngestionTool` as per the plan.
    -   [ ] Ensure it's registered in the `ToolRegistry` used by `ResearchAgent`.
-   [ ] **Agent - Update Plan (`agents/research_agent.py`)**:
    -   [ ] Modify the plan prompt in `start_analysis` to *actually* include Step 1 from `newworkflow.md` (Ingest Data using `data_ingestion_tool`, then LLM planning step using `llm_tool` to generate `$initial_plan`).
-   [ ] **Frontend - File Upload (`static/js/main.js`, `templates/index.html`)**:
    -   [ ] Ensure file upload UI works and sends the file to `/upload`.
-   [ ] **Testing**:
    *   Verify the Thinking Process panel appears and shows the planned steps (Ingest, LLM Plan).
    *   Upload a file and verify the `DataIngestionTool` runs (check logs/status panel output) and the subsequent LLM planning step uses its output.
    *   Check that the final output (the `$initial_plan` JSON) appears in the main chat when the run completes.

## Milestone 3: Displaying Intermediate Outputs & Search Integration

*(Goal: Show outputs from individual steps in the main chat as they complete, and add the search functionality.)*

-   [ ] **Frontend - Display Step Outputs (`static/js/main.js`)**:
    -   [ ] Modify `checkStatus` to track which step outputs have already been displayed.
    -   [ ] When polling, check `plan_run.outputs.step_outputs`. If a *new* output is found for a completed step, display it in the main chat area using `addChatMessage("Agent", output_value)`.
    -   [ ] Ensure the final output is still displayed correctly upon completion.
-   [ ] **Tool - Scientific Search (`scientific_workflow/tools/search_tools.py`, etc.)**:
    -   [ ] Implement `ScientificSearchTool` class.
    -   [ ] Handle API keys/setup.
    -   [ ] Register the tool.
-   [ ] **Agent - Update Plan (`agents/research_agent.py`)**:
    -   [ ] Extend the plan prompt to include Step 3 (LLM generates search queries) and Step 4 (execute `scientific_search_tool`).
-   [ ] **Testing**:
    *   Verify that the output of the LLM planning step (the `$initial_plan` JSON) appears in the chat *before* the search step runs.
    *   Verify that the output of the `scientific_search_tool` (the `$search_findings` list) appears in the chat after it completes.
    *   Check the Thinking Panel updates correctly.

## Milestone 4: Clarification Handling

*(Goal: Implement the ability for the agent to pause and ask for user input.)*

-   [ ] **Backend - Enhance `/status` (`main.py`)**:
    -   [ ] If `plan_run.state` is `NEED_CLARIFICATION`, extract the *first* outstanding clarification details (ID, prompt) and add them to the response.
-   [ ] **Backend - Re-enable `/clarify/{run_id}` (`main.py`)**:
    -   [ ] Implement the logic to find the correct clarification object.
    -   [ ] Call `agent.portia.resolve_clarification()`.
    -   [ ] Return the updated run status.
-   [ ] **Frontend - Clarification UI (`static/js/main.js`, `templates/index.html`)**:
    -   [ ] Implement `displayClarification` function to show the clarification prompt and input form.
    -   [ ] Modify `checkStatus`: if clarification data is present, call `displayClarification` and stop polling.
    -   [ ] Re-enable the clarification form submit listener:
        -   Send the response to `/clarify/{run_id}`.
        -   On success, call `/resume/{run_id}` and restart polling.
-   [ ] **Agent - Add Clarification Point (Example)** (`agents/research_agent.py`):
    -   [ ] Modify the plan prompt *or* a tool's logic to intentionally raise a simple clarification (e.g., ask the user to confirm before running a potentially long search). This is needed for testing.
-   [ ] **Testing**: Trigger the test clarification point. Verify the UI pauses, displays the prompt, allows user input, sends the response, resumes the run, and restarts polling.

## Milestone 5 onwards: Add Remaining Tools & Features

*(Goal: Incrementally add the Synthesis, Code Execution, Citation, Data Analysis tools and final report generation, building on the established interactive framework.)*

-   [ ] **Implement & Integrate Tools**: Follow Milestones 4, 5, 6 from the previous roadmap, ensuring intermediate outputs are displayed and clarifications handled where appropriate.
    -   [ ] Synthesis (LLM Step 5) -> Display summary in chat.
    -   [ ] Code Generation (LLM Step 6) & Execution (`CodeExecutionTool` Step 7) -> Display stdout/stderr and generated plot images in chat.
    -   [ ] Citation Formatting (`CitationTool` Step 8).
    -   [ ] Data Analysis (`AnalysisTool` Step 2 - conditional).
    -   [ ] Final Report Generation (LLM Step 9) -> Display final report in chat.
-   [ ] **Refine UI**: Improve the chat display, thinking panel, error handling.
-   [ ] **Comprehensive Testing**.
-   [ ] **Iteration & Advanced Features** (Prompt engineering, optional evaluation step, etc.).