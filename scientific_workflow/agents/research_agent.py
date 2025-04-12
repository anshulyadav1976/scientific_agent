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
# from tools import create_tool_registry # Not needed for M1
# Import PlanBuilder for easier plan creation if needed, or construct manually
# from portia import PlanBuilder

logger = logging.getLogger(__name__)

# --- Define the basic plan prompt for Milestone 1 --- #
# We can make this more complex later. For now, just rephrase.
# Alternatively, we could use the Step 1 prompt from newworkflow.md if DataIngestionTool is ready.
# Let's stick to the simplest LLM-only plan for initial async testing.
SIMPLE_PLAN_PROMPT_TEMPLATE = "Briefly acknowledge and rephrase the user's request: '{user_prompt}'"


class ResearchAgent:
    def __init__(self):
        """Initializes the Research Agent with Portia configuration and tools."""
        # --- DEBUG LOGGING --- #
        logger.info(f"Initializing ResearchAgent instance: ID={id(self)}")
        # --- END DEBUG LOGGING --- #

        # --- Configure for Disk Storage --- #
        self.config = default_config()
        # Explicitly set storage class to DISK
        self.config.storage_class = StorageClass.DISK
        # Ensure storage_dir is set if using DISK (Portia might default, but explicit is safer)
        # Default is often ./portia_storage in the current working directory
        if not self.config.storage_dir:
             self.config.storage_dir = "./portia_storage"
             logger.info(f"Defaulting storage_dir to: {self.config.storage_dir}")
        # --- END Configure for Disk Storage --- #

        # LLMTool is included by default in Portia's DefaultToolRegistry
        # self.llm_tool = LLMTool()
        # No need to manually create registry when using config
        # self.tools = InMemoryToolRegistry.from_local_tools([self.llm_tool])

        # Initialize Portia with the config. It will create DiskFileStorage
        # and DefaultToolRegistry (which includes LLMTool) automatically.
        self.portia = Portia(config=self.config)

        # --- DEBUG LOGGING --- #
        logger.info(f"Agent {id(self)} using storage: ID={id(self.portia.storage)}, Type={type(self.portia.storage).__name__}")
        # --- END DEBUG LOGGING --- #
        logger.info("ResearchAgent initialized with default config and LLMTool.")

    # Return type is now str (the run_id)
    async def start_analysis(self, user_prompt: str, file_path: Optional[str] = None) -> str:
        """Generates a plan for the user prompt and creates the initial PlanRun."""
        logger.info(f"Planning analysis for prompt: {user_prompt}")
        if file_path:
            logger.info(f"File path {file_path} provided, but ignoring for initial plan generation.")
            # TODO: Incorporate file_path into plan generation in later milestones

        # Generate the simple plan prompt
        plan_prompt = SIMPLE_PLAN_PROMPT_TEMPLATE.format(user_prompt=user_prompt)
        logger.debug(f"Generating plan with prompt: {plan_prompt}")

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