"""Main Agent logic for the Scientific Workflow"""

from portia import (
Portia,
 PlanRun)
from config.agent_config import create_agent_config
from tools import create_tool_registry
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initializes the Research Agent with Portia configuration and tools."""
        logger.info("Initializing ResearchAgent...")
        self.config = create_agent_config()
        self.tools = create_tool_registry()
        self.portia = Portia(config=self.config, tools=self.tools)
        logger.info("ResearchAgent initialized.")

    async def start_analysis(self, file_path: str, user_prompt: str, data_type_hint: Optional[str] = None) -> PlanRun:
        """Generates a plan and creates the initial PlanRun, but does not execute it."""
        logger.info(f"Planning analysis for file: {file_path}, hint: {data_type_hint}, prompt: {user_prompt}")

        # Define the high-level plan prompt for Portia
        # Include the hint in the prompt for the planner
        hint_text = f" The user provided a hint that the data type is '{data_type_hint}'." if data_type_hint else ""
        plan_prompt = f"""
        Analyze the dataset located at '{file_path}'.{hint_text}
        User context/request: '{user_prompt}'

        Follow these steps:
        1. Ingest the data from the file path using the data_ingestion_tool, passing the provided data type hint ('{data_type_hint}' if data_type_hint else 'None') if available. The tool will output a dictionary containing a summary and the detected_type ('tabular' or 'unstructured'). Name the output of this step $ingestion_output.
        2. **Conditionally Summarize**: If the 'detected_type' in $ingestion_output is 'unstructured', use the llm_tool to generate a concise summary of the 'snippet' found in $ingestion_output.summary. Ask the llm_tool: "Provide a brief summary (2-3 sentences) of the following text snippet: [snippet content]". Name the output of this step $llm_summary. If the detected_type is 'tabular', skip this step.
        3. **Conditionally Analyze**: If the 'detected_type' in $ingestion_output is 'tabular', use the analysis_tool. The required 'file_path' argument for this tool should be taken directly from the 'file_path' field within the $ingestion_output variable. Name the output of this step $analysis_results.
        # Add step 4 for LLM summary of analysis results later
        """
        logger.debug(f"Generated plan prompt:\n{plan_prompt}")

        try:
            # 1. Generate the plan
            logger.info("Generating plan...")
            plan = self.portia.plan(plan_prompt) # This is synchronous
            logger.info(f"Plan generated: {plan.id}")
            # --- DEBUG LOG: Check planned args for analysis_tool --- 
            try:
                analysis_step_index = -1
                for i, step in enumerate(plan.steps):
                    if step.tool_id == "analysis_tool":
                        analysis_step_index = i
                        break
                if analysis_step_index != -1:
                     analysis_step = plan.steps[analysis_step_index]
                     logger.info(f"[DEBUG] Planned args for analysis_tool (Step {analysis_step_index}): {getattr(analysis_step, 'tool_args', 'Not Set')}")
                else:
                     logger.info("[DEBUG] analysis_tool step not found in the generated plan.")
            except Exception as log_ex:
                 logger.error(f"[DEBUG] Error logging planned args: {log_ex}")
            # --- END DEBUG LOG ---
            
            # 2. Create the PlanRun
            logger.info(f"Creating PlanRun for plan {plan.id}...")
            plan_run = self.portia.create_plan_run(plan)
            logger.info(f"PlanRun created: {plan_run.id}, State: {plan_run.state}")

            # 3. Return the initial PlanRun (DO NOT run or resume here)
            return plan_run
        except Exception as e:
            logger.error(f"Error during planning or PlanRun creation: {e}", exc_info=True)
            raise # Re-raise the exception to be handled by the caller (FastAPI endpoint)

    async def get_run_status(self, run_id: str) -> PlanRun | None:
        """Retrieves the status and results of a specific plan run."""
        logger.debug(f"Getting status for run ID: {run_id}")
        try:
            # Access storage directly to get the run state
            run = self.portia.storage.get_plan_run(run_id)
            return run
        except Exception as e:
            # Handle cases where the run ID might not be found
            logger.error(f"Error retrieving run status for {run_id}: {e}", exc_info=True)
            return None

    async def resolve_clarification(self, clarification_id: str, response: str) -> PlanRun:
        """Submits a user response to resolve an agent clarification."""
        logger.info(f"Resolving clarification {clarification_id} with response: '{response}'")
        try:
            plan_run = await self.portia.resolve_clarification(clarification_id, response)
            logger.info(f"Clarification resolved. Run ID: {plan_run.id}, New Status: {plan_run.state}")
            return plan_run
        except Exception as e:
            logger.error(f"Error resolving clarification {clarification_id}: {e}", exc_info=True)
            raise 