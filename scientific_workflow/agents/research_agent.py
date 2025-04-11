"""Main Agent logic for the Scientific Workflow"""

from portia import (
Portia,
 PlanRun)
from config.agent_config import create_agent_config
from tools import create_tool_registry
import logging

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initializes the Research Agent with Portia configuration and tools."""
        logger.info("Initializing ResearchAgent...")
        self.config = create_agent_config()
        self.tools = create_tool_registry()
        self.portia = Portia(config=self.config, tools=self.tools)
        logger.info("ResearchAgent initialized.")

    async def start_analysis(self, file_path: str, user_prompt: str) -> PlanRun:
        """Starts the analysis workflow for a given dataset and user prompt."""
        logger.info(f"Starting analysis for file: {file_path}, prompt: {user_prompt}")

        # Define the high-level plan prompt for Portia
        # This will evolve as we add more tools in later phases
        plan_prompt = f"""
        Analyze the dataset located at '{file_path}'.
        User context/request: '{user_prompt}'

        Follow these steps:
        1. Ingest the data from the file path and provide a basic summary.
        # Add steps for analysis, visualization, search, question generation in later phases
        """

        try:
            # Use portia.run() which handles planning and execution
            plan_run = self.portia.run(plan_prompt)
            logger.info(f"Portia run initiated. Run ID: {plan_run.id}, Status: {plan_run.state}")
            return plan_run
        except Exception as e:
            logger.error(f"Error starting Portia analysis: {e}", exc_info=True)
            # In a real app, you might want to return a specific error state
            raise

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