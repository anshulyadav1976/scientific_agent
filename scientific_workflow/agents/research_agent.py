"""Main Agent logic for the Scientific Workflow"""

import logging
from typing import Optional

from portia import (
    Portia,
    PlanRun,
    PlanRunState,
    Plan,
    default_config,
    # LLMTool no longer needed if OllamaTool handles it
    # LLMTool,
    StorageClass,
    Clarification, InputClarification # Import clarification types
)
# Import DiskFileStorage from its submodule
from portia.storage import DiskFileStorage
# from config.agent_config import create_agent_config # Not needed for M1
from tools import create_tool_registry # Import locally to avoid circular dependency if tools need config
# Import PlanBuilder for easier plan creation if needed, or construct manually
# from portia import PlanBuilder
import json # Import json for potential parsing if needed

logger = logging.getLogger(__name__)

# --- Define the Stage 1 Prompt --- #
QUESTION_GEN_PROMPT_TEMPLATE = """
Analyze the following user request and extracted text content (if any).
Your goal is ONLY to identify and list the specific clarifying questions you need to ask the user to fully understand the research task and perform it effectively later.
Do NOT perform the research task itself yet.

User Request: {user_prompt}

Extracted Text Content:
```
{extracted_text}
```

Based *only* on the above, what clarifying questions do you have for the user? List them clearly.
If no clarification is needed, state "No clarification needed.".
Output ONLY the questions or the statement "No clarification needed.".
"""

# Define Stage 2 Prompt Template (for later use)
RESEARCH_PROMPT_TEMPLATE = """Perform the research task based on the original request, the provided text content, and the user's answers to clarifying questions.

Original User Request: {user_prompt}

Extracted Text Content:
```
{extracted_text}
```

User Answers to Clarifying Questions:
```
{user_answers}
```

Now, perform the full research task based on all the information above. Provide a comprehensive final report.
{tool_instructions} # Placeholder for how to call tools if needed later
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

    async def start_analysis(self, user_prompt: str, ollama_url: str, extracted_text: Optional[str] = None) -> str:
        """Generates the Stage 1 plan (question generation) and creates the PlanRun."""
        logger.info(f"Planning Stage 1 (Question Gen) for prompt: {user_prompt} using {ollama_url}")
        text_content = extracted_text if extracted_text else "No text content provided."

        # Format the Stage 1 prompt
        stage1_prompt = QUESTION_GEN_PROMPT_TEMPLATE.format(
            user_prompt=user_prompt,
            extracted_text=text_content
        )
        logger.debug(f"Generating Stage 1 plan with prompt:\n{stage1_prompt}")

        try:
            # Manually create Plan object for more control
            plan = Plan(
                plan_context={ # Use dict directly for context
                    "query": user_prompt, # Store original query
                    "ollama_url": ollama_url, # Store ollama url
                     # Add other relevant context if needed
                 },
                 steps=[
                     {
                         "task": "Generate clarifying questions using Ollama.",
                         "tool_id": "ollama_tool",
                         # Pass arguments needed by OllamaTool.run
                         "tool_args": {
                             "prompt": stage1_prompt,
                             "ollama_url": ollama_url,
                             # "model_name": "your_preferred_model" # Optional: override default
                         },
                         "output": "$clarifying_questions", # Name the output
                     }
                 ]
            )
            # Save the plan explicitly if needed (Portia might handle this in create_plan_run too)
            self.portia.storage.save_plan(plan)
            logger.info(f"Stage 1 Plan generated and saved: {plan.id}")

            # Create the PlanRun
            plan_run = self.portia.create_plan_run(plan)
            logger.info(f"PlanRun created: {plan_run.id}, State: {plan_run.state}")
            return str(plan_run.id)

        except Exception as e:
            logger.error(f"Error during Stage 1 planning or PlanRun creation: {e}", exc_info=True)
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
        """Resumes a PlanRun. Handles transition from Stage 1 (Question Gen) to Clarification."""
        logger.info(f"Attempting to resume run: {run_id}")
        current_run = await self.get_run_status(run_id)

        if current_run.state == PlanRunState.NOT_STARTED:
            # --- Execute Stage 1: Question Generation --- #
            logger.info(f"Executing Stage 1 (Question Gen) for run {run_id}")
            try:
                # Use Portia's resume to run the single OllamaTool step
                completed_stage1_run = self.portia.resume(plan_run_id=run_id)

                # Check if OllamaTool step completed successfully
                if completed_stage1_run.state == PlanRunState.COMPLETE:
                    # Get the output (the questions string)
                    questions_output = completed_stage1_run.outputs.step_outputs.get("$clarifying_questions")
                    if questions_output and hasattr(questions_output, 'value'):
                        questions_text = str(questions_output.value).strip()
                        logger.info(f"Ollama generated questions: {questions_text[:200]}...")

                        if "no clarification needed" in questions_text.lower():
                            # TODO: Skip clarification & proceed directly to Stage 2?
                            # For now, treat as complete, requiring manual resume for stage 2
                            logger.info("Ollama indicated no clarification needed. Run complete (manual resume needed for Stage 2).")
                            # State is already COMPLETE from portia.resume
                            return completed_stage1_run
                        else:
                            # Manually create and save clarification
                            clarification = InputClarification(prompt=questions_text)
                            # Add clarification to the *original* run object we fetched
                            current_run.outputs.clarifications.append(clarification)
                            current_run.state = PlanRunState.NEED_CLARIFICATION
                            self.portia.storage.save_plan_run(current_run)
                            logger.info(f"Run {run_id} state set to NEED_CLARIFICATION.")
                            return current_run # Return the modified run object
                    else:
                        logger.error(f"Could not find or read $clarifying_questions output for run {run_id}")
                        completed_stage1_run.state = PlanRunState.FAILED # Mark as failed
                        self.portia.storage.save_plan_run(completed_stage1_run)
                        return completed_stage1_run
                else:
                    # Stage 1 failed during portia.resume
                    logger.error(f"Stage 1 execution failed for run {run_id}. Final state: {completed_stage1_run.state}")
                    return completed_stage1_run # Return the failed run

            except Exception as e:
                logger.error(f"Exception during Stage 1 execution for run {run_id}: {e}", exc_info=True)
                # Mark run as failed if exception occurs
                current_run.state = PlanRunState.FAILED
                self.portia.storage.save_plan_run(current_run)
                return current_run

        elif current_run.state == PlanRunState.READY_TO_RESUME:
            # --- Execute Stage 2: Main Research --- #
            logger.info(f"Executing Stage 2 (Research) for run {run_id}")
            # Retrieve stored answers and original context
            user_answers = current_run.outputs.step_outputs.get("user_answers", "No answers provided.")
            # We need original prompt and text - store them in PlanRun context or re-fetch Plan?
            # Fetching plan is safer
            plan = await self.get_plan(current_run.plan_id)
            original_prompt = plan.plan_context.get("query", "Original prompt missing.")
            ollama_url = plan.plan_context.get("ollama_url", None)
            # TODO: Need extracted text too - store it in plan_context? Assume it is for now.
            extracted_text = plan.plan_context.get("extracted_text", "Extracted text missing.")
            # TODO: Get tool instructions/format if implementing tool use
            tool_instructions = "" # Placeholder

            if not ollama_url:
                 logger.error(f"Ollama URL missing in plan context for run {run_id}. Cannot execute Stage 2.")
                 current_run.state = PlanRunState.FAILED
                 self.portia.storage.save_plan_run(current_run)
                 return current_run

            stage2_prompt = RESEARCH_PROMPT_TEMPLATE.format(
                user_prompt=original_prompt,
                extracted_text=extracted_text,
                user_answers=user_answers,
                tool_instructions=tool_instructions
            )
            logger.debug(f"Stage 2 prompt for run {run_id}:\n{stage2_prompt[:500]}...")

            try:
                # Call OllamaTool directly (or create a temporary run)
                # Simplest: Direct call within the resume logic
                ollama_tool = self.tools.get_tool("ollama_tool")
                # Need to pass a dummy ToolRunContext or the real one if available?
                # Pass None for now, OllamaTool doesn't use it yet
                final_report = await ollama_tool.run(
                    ctx=None, # OllamaTool doesn't use ctx currently
                    prompt=stage2_prompt,
                    ollama_url=ollama_url
                    # model_name can be specified if needed
                )

                # Update the original run
                current_run.outputs.final_output = LocalOutput(value=final_report) # Wrap in Output object
                current_run.state = PlanRunState.COMPLETE
                self.portia.storage.save_plan_run(current_run)
                logger.info(f"Stage 2 completed for run {run_id}. State set to COMPLETE.")
                return current_run

            except Exception as e:
                logger.error(f"Exception during Stage 2 execution for run {run_id}: {e}", exc_info=True)
                current_run.state = PlanRunState.FAILED
                self.portia.storage.save_plan_run(current_run)
                return current_run
        else:
            # Should not happen if called correctly after /clarify or /upload
            logger.warning(f"Resume called for run {run_id} in unexpected state: {current_run.state}. Returning current run.")
            return current_run

    # Method to resolve clarification - needed by /clarify endpoint later
    async def resolve_clarification(self, run_id: str, clarification_id: str, response: str) -> PlanRun:
        """Stores user answers and sets run state to READY_TO_RESUME."""
        logger.info(f"Storing clarification response for run {run_id}, clarification {clarification_id}")
        run = await self.get_run_status(run_id)

        if run.state != PlanRunState.NEED_CLARIFICATION:
            raise InvalidPlanRunStateError(f"Run {run_id} is not in NEED_CLARIFICATION state.")

        # Find the clarification to mark as resolved (optional but good practice)
        found = False
        for clr in run.outputs.clarifications:
            if str(clr.id) == clarification_id:
                clr.resolved = True
                clr.resolution = response # Store response with the clarification too
                found = True
                break
        if not found:
             logger.warning(f"Did not find clarification {clarification_id} to mark resolved, but proceeding.")

        # Store the answers in the step_outputs for Stage 2
        # Use LocalOutput wrapper
        from portia.execution_agents.output import LocalOutput
        run.outputs.step_outputs["user_answers"] = LocalOutput(value=response)
        run.state = PlanRunState.READY_TO_RESUME

        self.portia.storage.save_plan_run(run)
        logger.info(f"Stored answers for run {run_id}. State set to READY_TO_RESUME.")
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