"""Main Agent logic for the Scientific Workflow"""

import logging
from typing import Optional
from pathlib import Path

from portia import (
    Portia,
    PlanRun,
    PlanRunState,
    Plan,
    default_config,
    StorageClass,
    # LLMTool no longer needed if OllamaTool handles everything
    # LLMTool,
    Clarification, InputClarification # Need Clarification types
)
# Import DiskFileStorage from its submodule
from portia.storage import DiskFileStorage
# from config.agent_config import create_agent_config # Not needed for M1
from tools import create_tool_registry # Import locally to avoid circular dependency if tools need config
# Import PlanBuilder for easier plan creation if needed, or construct manually
# from portia import PlanBuilder

logger = logging.getLogger(__name__)

# --- Prompt for Stage 1: Question Generation --- #
QUESTION_GEN_PROMPT_TEMPLATE = """
Analyze the following user request and context (if provided).
Your goal is *only* to identify and list the specific clarifying questions you need to ask the user to fulfill their request comprehensively. Do not answer the request itself yet.

User Request: {user_prompt}

Context from uploaded file (if any):
--- START CONTEXT ---
{file_context}
--- END CONTEXT ---

Based on the request and context, list the essential questions you need answered by the user before proceeding with the main task. Output *only* the questions, preferably as a numbered or bulleted list.
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

    # Accept ollama_url
    async def start_analysis(self, user_prompt: str, ollama_url: str, file_path: Optional[str] = None) -> str:
        """Generates the Stage 1 plan (question generation) and creates the initial PlanRun."""
        logger.info(f"Planning Stage 1 (Question Gen) for prompt: {user_prompt}, Ollama: {ollama_url}")

        # --- Basic Text Extraction Placeholder --- #
        # TODO: Replace with robust extraction in Milestone 2
        file_context = "No file provided." 
        if file_path:
            logger.info(f"File path {file_path} provided.")
            try:
                file_context = Path(file_path).read_text(encoding='utf-8', errors='ignore')
                logger.info(f"Read {len(file_context)} chars from file.")
            except Exception as e:
                logger.error(f"Error reading file {file_path} for context: {e}")
                file_context = "Error reading file content."
        # --- End Placeholder ---

        # Generate the question generation prompt
        question_gen_prompt = QUESTION_GEN_PROMPT_TEMPLATE.format(
            user_prompt=user_prompt,
            file_context=file_context
        )
        logger.debug(f"Generating Stage 1 plan with prompt:\n{question_gen_prompt[:500]}...")

        # Construct the plan manually for the single step
        # We need to pass ollama_url and the prompt to the tool
        plan = Plan.model_validate({
            "plan_context": {"query": f"Stage 1 Question Gen: {user_prompt[:100]}", "tool_ids": ["ollama_tool"]},
            "steps": [
                {
                    "task": "Generate clarifying questions using Ollama.",
                    "tool_id": "ollama_tool",
                    "output": "$clarifying_questions",
                    "prompt": question_gen_prompt,
                    "ollama_url": ollama_url,
                }
            ]
        })
        # Save the plan explicitly as portia.plan() was bypassed
        self.portia.storage.save_plan(plan)

        try:
            # Create the PlanRun (does not start execution)
            plan_run: PlanRun = self.portia.create_plan_run(plan)
            logger.info(f"PlanRun created for Stage 1: {plan_run.id}, State: {plan_run.state}")

            # Return the run_id
            return str(plan_run.id)

        except Exception as e:
            logger.error(f"Error during PlanRun creation: {e}", exc_info=True)
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
        """Resumes a PlanRun execution. Handles transition from Stage 1 to clarification."""
        logger.info(f"Attempting to resume run: {run_id}")
        current_run = await self.get_run_status(run_id)

        if current_run.state == PlanRunState.NOT_STARTED:
            # --- Execute Stage 1: Question Generation --- #
            logger.info(f"Executing Stage 1 (Question Generation) for run {run_id}")
            try:
                # Run the plan which calls OllamaTool to get questions
                # portia.resume should handle the single step execution
                executed_run = self.portia.resume(plan_run_id=run_id)

                # Check if the step completed successfully
                if executed_run.state == PlanRunState.COMPLETE and "$clarifying_questions" in executed_run.outputs.step_outputs:
                    questions_output = executed_run.outputs.step_outputs["$clarifying_questions"]
                    if hasattr(questions_output, 'value'):
                        questions_text = str(questions_output.value)
                        logger.info(f"Stage 1 completed. Questions received: {questions_text[:200]}...")

                        # Manually create clarification and update run state
                        clarification = InputClarification(
                            run_id=executed_run.id,
                            step=0, # Assuming step 0 generated questions
                            prompt=f"Please answer the following questions to proceed:\n\n{questions_text}"
                        )
                        # Add clarification to the run's outputs
                        executed_run.outputs.clarifications.append(clarification)
                        # Set state to NEED_CLARIFICATION
                        executed_run.state = PlanRunState.NEED_CLARIFICATION
                        # Save the updated run state back to storage
                        self.portia.storage.save_plan_run(executed_run)
                        logger.info(f"Run {run_id} state set to NEED_CLARIFICATION.")
                        return executed_run
                    else:
                        raise ToolHardError("OllamaTool output for questions missing expected value.")
                else:
                    # Handle cases where the single step failed
                    logger.error(f"Stage 1 OllamaTool step failed for run {run_id}. State: {executed_run.state}")
                    executed_run.state = PlanRunState.FAILED # Ensure it's marked as failed
                    self.portia.storage.save_plan_run(executed_run)
                    return executed_run # Return the failed run object

            except Exception as e:
                logger.error(f"Error during Stage 1 execution for {run_id}: {e}", exc_info=True)
                # Mark run as failed if exception occurs during resume
                try:
                    failed_run = await self.get_run_status(run_id)
                    failed_run.state = PlanRunState.FAILED
                    # TODO: Store error details in the run?
                    self.portia.storage.save_plan_run(failed_run)
                    return failed_run
                except Exception as e_save:
                    logger.error(f"Failed even to save failed state for run {run_id}: {e_save}")
                    raise e # Re-raise original error

        elif current_run.state == PlanRunState.READY_TO_RESUME:
             # --- Execute Stage 2: Research --- #
             logger.info(f"Executing Stage 2 (Research) for run {run_id}")
             # TODO: Implement Stage 2 logic in Milestone 2 of new roadmap
             # 1. Retrieve original prompt, context, user_answers from storage/run
             # 2. Define RESEARCH_PROMPT_TEMPLATE
             # 3. Format prompt
             # 4. Create Plan: Step 1: Run OllamaTool with research prompt
             # 5. Run plan (e.g., using portia.run or resume)
             # 6. Update original run with final output and COMPLETE state
             logger.warning(f"Stage 2 execution not yet implemented for run {run_id}.")
             # For now, just mark as complete after clarification for testing flow
             current_run.state = PlanRunState.COMPLETE
             current_run.outputs.final_output = LocalOutput(value="(Stage 2 research not implemented yet)")
             self.portia.storage.save_plan_run(current_run)
             return current_run
        else:
             # Handle unexpected states (e.g., already COMPLETE, FAILED, IN_PROGRESS)
             logger.warning(f"Resume called on run {run_id} with unexpected state: {current_run.state}. Returning current state.")
             return current_run

    # Method to resolve clarification - needed by /clarify endpoint later
    async def resolve_clarification(self, run_id: str, clarification_id: str, response: str) -> PlanRun:
        """Stores user response and sets run state to READY_TO_RESUME."""
        logger.info(f"Storing clarification response for run {run_id}, clarification {clarification_id}")
        run = await self.get_run_status(run_id)

        if run.state != PlanRunState.NEED_CLARIFICATION:
            raise InvalidPlanRunStateError(f"Run {run_id} is not in NEED_CLARIFICATION state.")

        # Find the specific clarification to mark resolved (optional but good practice)
        clarification_resolved = False
        for clr in run.outputs.clarifications:
            if str(clr.id) == clarification_id and not clr.resolved:
                clr.resolved = True # Mark it resolved
                clr.response = response # Store response directly on clarification object
                clarification_resolved = True
                break

        if not clarification_resolved:
            logger.warning(f"Clarification ID {clarification_id} not found or already resolved for run {run_id}. Storing answers anyway.")

        # Store user answers separately in outputs for Stage 2 use
        run.outputs.step_outputs["user_answers"] = LocalOutput(value=response)

        # Update state
        run.state = PlanRunState.READY_TO_RESUME
        self.portia.storage.save_plan_run(run)
        logger.info(f"Run {run_id} state set to READY_TO_RESUME.")
        return run

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