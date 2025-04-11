AI Agent for Scientific Workflows
Feature Overview & User Flow

Core Features
1. Data Input and Classification
Upload support for tabular data, unstructured text.

Automatic or user-defined data type classification.

2. Context-Aware Processing
For unstructured or tabular data:

Prompt user for research context.

Perform local Python-based exploratory analysis.

Summarize insights using LLMs (if enabled).


3. Memory and Notebook System
Session memory for user preferences, past steps, and decisions.

Automatic generation of a research notebook view with:

Raw input logs

Reasoning chains

Analysis outputs

Hypotheses and rejected ideas

User-editable notes in  a side canvas like interface

Export capability (e.g., PDF, markdown)



5. In-Depth Analysis and Exploration
Based on data type and context, the agent suggests areas of focus.

Asks clarifying questions when necessary.

Performs advanced analysis (correlations, outliers, patterns).

Recommends relevant subtopics or areas for deeper investigation.
6. Visualization
Automatically generates graphs, correlation heatmaps, and anomaly indicators.

Visualization driven by analytical insights, not raw data alone.

7. External Research Integration
Agent performs external searches using APIs (e.g., Semantic Scholar).

Gathers relevant datasets and references.

Generates hypotheses and outlines experimental or research directions.

Asks follow-up questions to refine scope and relevance.




User Flow
Start a New Session

Upload dataset.



Select Data Type

Unstructured

Tabular


Context Input & Initial Analysis

Prompt for research context.

Process data accordingly:

Local Python-based queries.

LLM summaries (if enabled).


Notebook Logging

Agent logs each step into the notebook.

User can review, edit, or annotate logs at any time.

Visualizations

Auto-generated insights visualized for the user.

Analysis Suggestions

Agent recommends deeper areas to explore.

User provides additional context or selects from suggestions.

Advanced Research Phase

Based on goals and context, agent performs:

Online research

Hypothesis generation

Literature review

Dataset augmentation



Summarize findings.

Provide downloadable notebook or research report.


