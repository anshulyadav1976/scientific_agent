"""Main Agent logic for the Scientific Workflow"""

import logging
from typing import Optional

from portia import (
    Portia,
    PlanRun,
    PlanRunState,
    Plan,
    default_config,
    LLMTool,
    # DiskFileStorage is in the storage submodule
    StorageClass,
)
# Import DiskFileStorage from its submodule
from portia.storage import DiskFileStorage
# from config.agent_config import create_agent_config # Not needed for M1
from tools import create_tool_registry # Import locally to avoid circular dependency if tools need config
# Import PlanBuilder for easier plan creation if needed, or construct manually
# from portia import PlanBuilder

logger = logging.getLogger(__name__)

# --- Define the plan prompt for Milestone 2 --- #
# This now includes Step 1a (Ingest) and Step 1b (LLM Plan Generation)
PLAN_PROMPT_STEP_1 = """
Follow these steps to analyze the user request:
1.  **Ingest Data (Conditional):** If a file path '{file_path}' was provided, run the `data_ingestion_tool` with the `file_path` argument set to '{file_path}'. Name the output of this step `$ingestion_output`. If no file path was provided, skip this step and set `$ingestion_output` to None.
2.  **Generate Initial Plan:** Analyze the user prompt '{user_prompt}' and the `$ingestion_output` (if available). Define the main research `goal`. Determine if analysis of the provided data is required ('`analysis_required`: true/false'). Outline key `research_questions` and list initial `search_topics` for literature/web search. Output ONLY a JSON object containing these keys: {{"goal": "...", "analysis_required": true/false, "research_questions": [...], "search_topics": [...]}}. Name the output of this step `$initial_plan`.
"""


class ResearchAgent:
    def __init__(self):
        """Initializes the Research Agent with Portia configuration and tools."""
        # --- DEBUG LOGGING --- #
        logger.info(f"Initializing ResearchAgent instance: ID={id(self)}")
        # --- END DEBUG LOGGING --- #

        # --- Configure for Disk Storage --- #
        self.config = default_config()
        self.config.storage_class = StorageClass.DISK
        if not self.config.storage_dir:
             self.config.storage_dir = "./portia_storage"
             logger.info(f"Defaulting storage_dir to: {self.config.storage_dir}")
        # --- END Configure for Disk Storage --- #

        # --- Load Tools --- #
        # Use the create_tool_registry function which includes DataIngestionTool, LLMTool etc.
        self.tools = create_tool_registry()
        # --- END Load Tools --- #

        # Initialize Portia with config and the combined tool registry
        self.portia = Portia(config=self.config, tools=self.tools)

        # --- DEBUG LOGGING --- #
        logger.info(f"Agent {id(self)} using storage: ID={id(self.portia.storage)}, Type={type(self.portia.storage).__name__}")
        # --- END DEBUG LOGGING --- #
        logger.info("ResearchAgent initialized with default config and LLMTool.")

    # Return type is now str (the run_id)
    async def start_analysis(self, user_prompt: str, file_path: Optional[str] = None) -> str:
        """Generates a plan for the user prompt & optional file, and creates the initial PlanRun."""
        logger.info(f"Planning analysis for prompt: {user_prompt}")
        file_path_str = file_path if file_path else "None" # Pass "None" string if no path
        if file_path:
            logger.info(f"File path {file_path} provided.")

        # Generate the plan prompt using the template for Step 1
        plan_prompt = PLAN_PROMPT_STEP_1.format(user_prompt=user_prompt, file_path=file_path_str)
        logger.debug(f"Generating plan with prompt:\n{plan_prompt}")

        try:
            # 1. Generate the plan
            plan: Plan = self.portia.plan(plan_prompt)
            logger.info(f"Plan generated: {plan.id}")

            # 2. Create the PlanRun (does not start execution)
            plan_run: PlanRun = self.portia.create_plan_run(plan)
            logger.info(f"PlanRun created: {plan_run.id}, State: {plan_run.state}")

            # --- DEBUG LOGGING --- #
            logger.info(f"[start_analysis] Agent {id(self)} returning run_id {plan_run.id} from storage {id(self.portia.storage)}")
            # --- END DEBUG LOGGING --- #

            # 3. Return the run_id
            return str(plan_run.id)

        except Exception as e:
            logger.error(f"Error during planning or PlanRun creation: {e}", exc_info=True)
            # Re-raise the exception to be handled by the FastAPI endpoint
            raise

    # --- Methods needed for stateful execution --- #

    # Method to get status - needed by /status endpoint
    async def get_run_status(self, run_id: str) -> PlanRun:
        """Retrieves the status and results of a specific plan run."""
        logger.debug(f"Getting status for run ID: {run_id}")
        # Access storage directly. Raises PlanRunNotFoundError if not found.
        # Assuming storage access might be async or okay to await from async context
        run = self.portia.storage.get_plan_run(run_id)
        return run

    # Method to get the plan - needed by /status endpoint later for thinking process
    async def get_plan(self, plan_id: str) -> Plan:
        """Retrieves the plan associated with a run."""
        logger.debug(f"Getting plan for Plan ID: {plan_id}")
        # Access storage directly. Raises PlanNotFoundError if not found.
        # Assuming storage access might be async or okay to await from async context
        plan = self.portia.storage.get_plan(plan_id)
        return plan

    # Method to resume a run - needed by /resume endpoint
    # This *is* the main execution logic trigger in Portia
    async def resume_run(self, run_id: str) -> PlanRun:
        """Resumes a PlanRun execution."""
        logger.info(f"Attempting to resume run: {run_id}")
        # --- DEBUG LOGGING --- #
        logger.info(f"[resume_run] Agent {id(self)} using storage {id(self.portia.storage)} to resume run {run_id}")
        # --- END DEBUG LOGGING --- #
        # Portia's resume handles running steps until completion or clarification
        # Raises PlanRunNotFoundError, InvalidPlanRunStateError etc.
        # Assuming portia.resume is awaitable or handles sync execution appropriately
        resumed_run = self.portia.resume(plan_run_id=run_id)
        logger.info(f"Resume completed for {run_id}. Final state: {resumed_run.state}")
        return resumed_run

    # Method to resolve clarification - needed by /clarify endpoint later
    async def resolve_clarification(self, run_id: str, clarification_id: str, response: str) -> PlanRun:
        """Submits a user response to resolve an agent clarification."""
        logger.info(f"Resolving clarification {clarification_id} for run {run_id} with response: '{response}'")
        # Need to fetch the run first to find the clarification object
        run = await self.get_run_status(run_id)
        clarification_to_resolve = None
        for clr in run.get_outstanding_clarifications():
            if str(clr.id) == clarification_id:
                clarification_to_resolve = clr
                break

        if not clarification_to_resolve:
            error_msg = f"Outstanding clarification ID {clarification_id} not found for run {run_id}"
            logger.error(error_msg)
            # Depending on Portia version, InvalidState might be better
            raise ValueError(error_msg)

        # Use Portia's method to resolve
        # Assuming portia.resolve_clarification is awaitable or handles sync execution
        updated_run = self.portia.resolve_clarification(
            clarification=clarification_to_resolve,
            response=response,
            plan_run=run
        )
        logger.info(f"Clarification {clarification_id} resolved for run {run_id}. New Status: {updated_run.state}")
        return updated_run

    # --- Methods below are not needed for Milestone 1 --- #

    # async def get_run_status(self, run_id: str) -> PlanRun | None:
    #     """Retrieves the status and results of a specific plan run."""
    #     logger.debug(f"Getting status for run ID: {run_id}")
    #     try:
    #         # Access storage directly to get the run state
    #         run = self.portia.storage.get_plan_run(run_id)
    #         return run
    #     except Exception as e:
    #         # Handle cases where the run ID might not be found
    #         logger.error(f"Error retrieving run status for {run_id}: {e}", exc_info=True)
    #         return None

    # async def resolve_clarification(self, clarification_id: str, response: str) -> PlanRun:
    #     """Submits a user response to resolve an agent clarification."""
    #     logger.info(f"Resolving clarification {clarification_id} with response: '{response}'")
    #     try:
    #         plan_run = await self.portia.resolve_clarification(clarification_id, response)
    #         logger.info(f"Clarification resolved. Run ID: {plan_run.id}, New Status: {plan_run.state}")
    #         return plan_run
    #     except Exception as e:
    #         logger.error(f"Error resolving clarification {clarification_id}: {e}", exc_info=True)
    #         raise 