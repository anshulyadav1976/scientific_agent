# Scientific Workflow Agent - Development Roadmap (Based on user_flowand_features.md)

This roadmap tracks the implementation of the AI Agent for Scientific Workflows.

## Feature: 1. Data Input and Classification

- [x] **Upload**: Support for uploading tabular data (CSV). (`/upload` endpoint, `index.html` form)
- [ ] **Upload**: Support for unstructured text input (e.g., pasting text, uploading .txt).
- [ ] **Classification**: Automatic data type detection (tabular vs. unstructured).
- [ ] **Classification**: Allow user override/confirmation of data type.

## Feature: 2. Context-Aware Processing

- [x] **Context Input**: Prompt user for initial research context via text area. (`index.html` form)
- [x] **Tabular - Initial Analysis**: Perform basic local Python-based analysis (current `DataIngestionTool` does this). 
    - [x] Read data (pandas).
    - [x] Basic summary (shape, columns, head, info, describe).
- [ ] **Unstructured - Initial Analysis**: 
    - [ ] Implement processing logic for unstructured text (e.g., tokenization, basic stats, keyword extraction).
- [ ] **LLM Summarization (Optional/Conditional)**: 
    - [ ] Tool/Logic to summarize initial findings (from Python analysis) using an LLM.
    - [ ] Mechanism to enable/disable LLM usage.

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

## Feature: 4. In-Depth Analysis and Exploration (Enhances former 'Pattern Detection')

- [ ] **Analysis Tool(s)**:
    - [ ] Calculate correlations (numerical data).
    - [ ] Detect outliers/anomalies.
    - [ ] Identify statistical patterns/trends.
    - [ ] (Future) Support analysis specific to unstructured text (topic modeling, sentiment, etc.).
- [ ] **Suggestion Engine**: Agent logic to suggest areas of focus based on initial analysis.
- [ ] **Clarification Integration**: 
    - [ ] Agent explicitly asks clarifying questions when analysis results are ambiguous or multiple paths exist (e.g., "Found correlations A & B, which to explore?").
    - [ ] Backend/Frontend implementation for handling these analysis-specific clarifications (current mechanism is generic).

## Feature: 5. Visualization (Insight-Driven)

- [ ] **Visualization Tool(s)**: 
    - [ ] Generate relevant plots based on analysis findings (e.g., correlation heatmaps, anomaly plots, trend lines), not just raw data.
    - [ ] Save plots as image files.
- [ ] **Agent Integration**: Trigger visualization tool based on analysis results.
- [ ] **Frontend Display**: Embed generated plots within the Notebook view.

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