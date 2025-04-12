"""Main FastAPI application for the Scientific Workflow Agent."""

import logging
import os
import uvicorn
import shutil # Added for saving file
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
# Changed RedirectResponse to JSONResponse for API-like behavior
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path # Added for path manipulation
import json
from typing import Optional # Import Optional
from pydantic import BaseModel # Import BaseModel

# Import PlanRunState to check run status
from portia import PlanRunState

# Import Portia models needed for status checking and response
from portia import PortiaBaseError # Import error base class if needed for specific checks
from portia.plan_run import PlanRun # Explicitly import PlanRun if needed for type hinting
from portia import Plan # Added Plan
from portia import PlanNotFoundError # Added for PlanNotFoundError
from portia import PlanRunNotFoundError # Import PlanRunNotFoundError
from portia import InvalidPlanRunStateError # Added for InvalidPlanRunStateError

# Import the summarizer
from portia.execution_agents.utils.final_output_summarizer import FinalOutputSummarizer

from agents.research_agent import ResearchAgent

# --- Pydantic Models for Request Bodies --- #
class ClarificationPayload(BaseModel):
    clarification_id: str
    response: str

# --- Configuration & Initialization --- #

# Load environment variables (API keys, etc.)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Scientific Workflow Agent")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Initialize the Research Agent (Singleton pattern might be better for production)
# We initialize it here so it's ready when needed by endpoints
try:
    agent = ResearchAgent()
except Exception as e:
    logger.error(f"Failed to initialize ResearchAgent: {e}", exc_info=True)
    # Depending on the error, you might want to exit or run with limited functionality
    agent = None # Or handle appropriately
    # raise HTTPException(status_code=500, detail="Failed to initialize AI Agent.")

# Temporary storage for uploaded files (replace with better storage later)
UPLOAD_DIR = Path("./uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- FastAPI Endpoints --- #

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main index page."""
    logger.info("Serving index page.")
    if agent is None:
         # Render a simple error message if agent failed to load
         return templates.TemplateResponse("error.html", {"request": request, "error_message": "Agent failed to initialize. Check logs."}, status_code=500)
    return templates.TemplateResponse("index.html", {"request": request})

# --- Placeholder Endpoints (to be implemented in later steps) --- #

# Uncommented and implemented the upload endpoint
@app.post("/upload")
# Add data_type_hint to arguments, default to None
async def handle_upload(
    request: Request, 
    file: UploadFile = File(...), 
    prompt: str = Form(...),
    data_type_hint: Optional[str] = Form(None) # Get hint from form
):
    """Handles file upload and starts the analysis workflow."""
    logger.info(f"Received upload request. Filename: {file.filename}, Hint: {data_type_hint}, Prompt: '{prompt[:50]}...'")

    if agent is None:
        logger.error("Upload attempt failed: Agent not initialized.")
        raise HTTPException(status_code=503, detail="Agent is not available.")

    # Update validation to include text/plain
    allowed_content_types = ["text/csv", "application/vnd.ms-excel", "text/plain"]
    if file.content_type not in allowed_content_types:
        logger.warning(f"Upload failed: Invalid content type {file.content_type}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Please upload a CSV or TXT file. Got: {file.content_type}"
        )

    # Secure filename and define save path
    # TODO: Implement more robust filename sanitization/generation
    safe_filename = Path(file.filename).name # Basic sanitization
    file_path = UPLOAD_DIR / safe_filename

    # Save the uploaded file
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")
    finally:
        await file.close()

    # Start the analysis using the agent
    try:
        logger.info(f"Calling agent.start_analysis with path: {file_path}, hint: {data_type_hint}, and prompt.")
        # Pass the absolute path, hint, and prompt
        plan_run = await agent.start_analysis(
            file_path=str(file_path.resolve()), 
            user_prompt=prompt,
            data_type_hint=data_type_hint
        )
        logger.info(f"Agent analysis started. Run ID: {plan_run.id}, Initial Status: {plan_run.state}")
        # Return only the run ID - frontend will call /resume to start
        return JSONResponse(content={
            "message": "Plan created successfully. Ready to start analysis.",
            "run_id": str(plan_run.id)
        })
    except Exception as e:
        logger.error(f"Failed to start agent analysis for {file_path}: {e}", exc_info=True)
        # Clean up saved file if analysis start fails?
        # os.remove(file_path) # Consider implications
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {e}")

# Uncommented and implemented the status endpoint
@app.get("/status/{run_id}")
async def get_status(run_id: str):
    """Gets the status of an ongoing analysis run."""
    logger.info(f"Received status request for run_id: {run_id}")
    if agent is None:
        logger.error(f"Status request failed for {run_id}: Agent not initialized.")
        raise HTTPException(status_code=503, detail="Agent is not available.")

    try:
        plan_run = await agent.get_run_status(run_id)
        plan = None
        if plan_run:
            # Also fetch the corresponding plan to get step descriptions
            # Access storage directly, get_plan is synchronous
            try:
                plan = agent.portia.storage.get_plan(plan_run.plan_id)
            except PlanNotFoundError:
                 logger.error(f"Plan not found in storage for Plan ID {plan_run.plan_id}")
            except Exception as e:
                logger.error(f"Error retrieving plan {plan_run.plan_id}: {e}", exc_info=True)

        if plan_run is None:
            logger.warning(f"Run ID not found: {run_id}")
            raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
        if plan is None:
             # This shouldn't happen if plan_run exists, but handle defensively
             logger.error(f"Plan not found for Run ID {run_id}, Plan ID {plan_run.plan_id if plan_run else 'N/A'}")
             # Continue without thinking process if plan is missing

        # --- Prepare Thinking Process History --- # 
        thinking_process = []
        if plan:
            # Iterate through steps up to the current one (or all if completed/failed)
            max_step_index = len(plan.steps) if plan_run.state in [PlanRunState.COMPLETE, PlanRunState.FAILED] else plan_run.current_step_index + 1
            for i in range(min(max_step_index, len(plan.steps))):
                step = plan.steps[i]
                step_detail = {
                    "step_index": i,
                    "description": step.task,
                    "tool_id": step.tool_id,
                    # Use getattr for safe access to potentially missing tool_args
                    "tool_args": getattr(step, 'tool_args', {}), # Default to empty dict
                    "status": "Pending", # Default status
                    "output": None,
                    "output_name": step.output
                }

                # Determine step status and output
                if i < plan_run.current_step_index or plan_run.state in [PlanRunState.COMPLETE, PlanRunState.FAILED]:
                    step_detail["status"] = "Executed"
                    # Try to get the output for this step using step.output as the key
                    if step.output in plan_run.outputs.step_outputs:
                         step_output_obj = plan_run.outputs.step_outputs[step.output]
                         if hasattr(step_output_obj, 'value'):
                            # Serialize complex objects like dicts/lists to string for display
                            output_value = step_output_obj.value
                            if isinstance(output_value, (dict, list, tuple, set)):
                                try:
                                     step_detail["output"] = json.dumps(output_value, indent=2)
                                except TypeError:
                                     step_detail["output"] = "(Output is complex/non-serializable)"
                            else:
                                step_detail["output"] = str(output_value)
                         else:
                            step_detail["output"] = "(Output object lacks .value)"
                    else:
                         step_detail["output"] = "(Output not found in run)"
                elif i == plan_run.current_step_index and plan_run.state == PlanRunState.IN_PROGRESS:
                    step_detail["status"] = "Executing"
                elif plan_run.state == PlanRunState.NEED_CLARIFICATION and i == plan_run.current_step_index:
                     step_detail["status"] = "Paused (Needs Clarification)"
                     # Optional: Add clarification details here if needed
                
                thinking_process.append(step_detail)
        else:
             logger.warning(f"Could not retrieve plan for run {run_id}, thinking process unavailable.")

        # --- Prepare the main response content --- #
        response_content = {
            "run_id": str(plan_run.id),
            "status": plan_run.state.value,
            "raw_output": None, # Store raw final output here
            "formatted_summary": None, # Store human-readable summary here
            "error": None,
            "clarification": None,
            "thinking_process": thinking_process
        }

        # --- Handle FINAL Output/Error/Clarification based on Run State --- #

        if plan_run.state == PlanRunState.COMPLETE:
            logger.debug(f"Run {run_id} completed. Accessing final outputs and generating summary.")
            extracted_raw_output = None
            final_output_extracted = False
            formatted_summary = "(Summary generation failed)" # Default summary on error
            try:
                # --- Try to get final_output.value FIRST --- 
                final_output_obj = plan_run.outputs.final_output
                if final_output_obj and hasattr(final_output_obj, 'value'):
                    extracted_raw_output = final_output_obj.value # Store the raw value
                    final_output_extracted = True
                    logger.info(f"Successfully extracted final_output.value for run {run_id}")
                
                # --- Fallback to first step_output ONLY IF final_output failed ---
                if not final_output_extracted and plan_run.outputs.step_outputs:
                    logger.warning(f"final_output not found or lacked .value for {run_id}, falling back to step outputs.")
                    try:
                        first_step_output_key = next(iter(plan_run.outputs.step_outputs))
                        first_step_output_object = plan_run.outputs.step_outputs[first_step_output_key]
                        if hasattr(first_step_output_object, 'value'):
                            extracted_raw_output = first_step_output_object.value
                            logger.debug(f"Using value from first step output '{first_step_output_key}' for run {run_id}")
                        else:
                             logger.warning(f"Fallback step output for {run_id} lacks .value attribute.")
                    except StopIteration:
                         logger.warning(f"Fallback step_outputs dict for {run_id} is empty.")
                elif not final_output_extracted:
                     logger.warning(f"Completed run {run_id} has no usable final_output and empty step_outputs.")

                response_content["raw_output"] = extracted_raw_output

                # --- Generate Formatted Summary --- 
                if plan: # Need the plan object for summarization
                     try:
                         summarizer = FinalOutputSummarizer(agent.config)
                         formatted_summary = summarizer.create_summary(plan=plan, plan_run=plan_run)
                         if formatted_summary:
                             logger.info(f"Generated formatted summary for run {run_id}")
                         else:
                             logger.warning(f"Summary generation returned None for run {run_id}")
                             formatted_summary = "(Summary generation returned empty)" # Use a different default
                     except Exception as summ_ex:
                         logger.error(f"Error generating formatted summary for {run_id}: {summ_ex}", exc_info=True)
                         # Keep default error summary
                else:
                    logger.warning(f"Cannot generate formatted summary for {run_id} because plan object is missing.")
                    formatted_summary = "(Could not generate summary: Plan data missing)"

                response_content["formatted_summary"] = formatted_summary

            except Exception as ex:
                logger.error(f"Error processing completed run {run_id} outputs: {ex}", exc_info=True)
                response_content["error"] = f"Error processing outputs: {ex}"
                # Use default summary if an error occurred during processing
                response_content["formatted_summary"] = formatted_summary 
        
        elif plan_run.state == PlanRunState.FAILED:
            logger.warning(f"Run {run_id} failed. Extracting error details.")
            # Extract error details from the PlanRun object
            error_message = "Unknown error" # Default
            if plan_run.error:
                if isinstance(plan_run.error, dict) and 'message' in plan_run.error:
                    error_message = plan_run.error['message']
                elif isinstance(plan_run.error, str):
                    error_message = plan_run.error
                else:
                    try:
                         error_message = str(plan_run.error)
                    except Exception:
                         error_message = "(Could not serialize error details)"
            response_content["error"] = error_message
            logger.debug(f"Setting error for failed run {run_id}: {error_message}")
        
        elif plan_run.state == PlanRunState.NEED_CLARIFICATION:
            logger.info(f"Run {run_id} needs clarification. Extracting details.")
            # Extract the first pending clarification
            if plan_run.outputs.clarifications:
                # Assuming we only show the first one based on JS logic
                first_clarification = plan_run.outputs.clarifications[0]
                response_content["clarification"] = {
                    "id": str(first_clarification.id), # Ensure ID is a string for JSON
                    "prompt": first_clarification.prompt
                }
                logger.debug(f"Added clarification details for {run_id}: ID {first_clarification.id}")
            else:
                # This case should ideally not happen if state is NEED_CLARIFICATION
                logger.error(f"Run {run_id} is in NEED_CLARIFICATION state but has no clarifications listed.")
                response_content["error"] = "Internal inconsistency: Run needs clarification but none found."
                # Optionally, change status? Or let frontend handle missing clarification data.

        # Return the composed status response
        return JSONResponse(content=response_content)

    except PlanRunNotFoundError:
        logger.warning(f"Run ID not found during status attempt: {run_id}")
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
    except PortiaBaseError as pe:
        logger.error(f"Portia error retrieving status for {run_id}: {pe}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Portia-related error retrieving status: {pe}")
    except Exception as e:
        logger.error(f"Error retrieving status for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve status for run {run_id}: {e}")

# --- New Endpoint: Resume Run --- #
@app.post("/resume/{run_id}")
async def resume_run(run_id: str):
    """Resumes a plan run, executing steps until completion, failure, or next clarification."""
    logger.info(f"Received resume request for run_id: {run_id}")
    if agent is None:
        logger.error(f"Resume request failed for {run_id}: Agent not initialized.")
        raise HTTPException(status_code=503, detail="Agent is not available.")

    try:
        # Fetch the current run to ensure it exists and is in a resumable state
        # (get_run_status also uses storage.get_plan_run which is synchronous)
        current_plan_run = agent.portia.storage.get_plan_run(run_id)
        if not current_plan_run:
             # Should be caught by get_plan_run raising PlanRunNotFoundError, but belt-and-suspenders
             raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
        
        logger.info(f"Run {run_id} current state: {current_plan_run.state}. Attempting resume...")
        
        # Check if resumable (Portia's resume also checks this, but good to be explicit)
        if current_plan_run.state not in [
            PlanRunState.NOT_STARTED,
            PlanRunState.READY_TO_RESUME, # Typically resume from this state after clarification
            # Allow resuming IN_PROGRESS? Portia might handle this.
            PlanRunState.IN_PROGRESS, 
            PlanRunState.NEED_CLARIFICATION # Resume will likely just return immediately if clarification not resolved
        ]:
             logger.warning(f"Run {run_id} is in state {current_plan_run.state}, cannot resume.")
             raise HTTPException(status_code=400, detail=f"Run {run_id} is in state {current_plan_run.state} and cannot be resumed now.")

        # Call the synchronous portia.resume method
        # FastAPI handles running sync functions in a threadpool for async endpoints
        updated_plan_run = agent.portia.resume(plan_run_id=run_id)
        
        logger.info(f"Resume call completed for {run_id}. New state: {updated_plan_run.state}")
        
        # Return the state after resuming - the frontend will poll /status for details
        return JSONResponse(content={
            "message": f"Resume initiated for run {run_id}.",
            "run_id": str(updated_plan_run.id),
            "status_after_resume": updated_plan_run.state.value
        })

    except PlanRunNotFoundError:
         logger.warning(f"Run ID not found during resume attempt: {run_id}")
         raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
    except InvalidPlanRunStateError as ipse:
        logger.warning(f"Invalid state for resume for run {run_id}: {ipse}")
        # Fetch the current state again to return it accurately
        failed_run = agent.portia.storage.get_plan_run(run_id)
        raise HTTPException(status_code=400, detail=f"Run {run_id} is in state {failed_run.state} and cannot be resumed now. {ipse}")
    except PortiaBaseError as pe:
        logger.error(f"Portia error resuming run {run_id}: {pe}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Portia-related error resuming run: {pe}")
    except Exception as e:
        logger.error(f"Unexpected error resuming run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to resume run {run_id}: {e}")

# --- New Endpoint: Resolve Clarification --- #
@app.post("/clarify/{run_id}")
async def handle_clarification(run_id: str, payload: ClarificationPayload):
    """Resolves a specific clarification for a given run."""
    logger.info(f"Received clarification resolution for run_id: {run_id}, clarification_id: {payload.clarification_id}")
    if agent is None:
        logger.error(f"Clarification request failed for {run_id}: Agent not initialized.")
        raise HTTPException(status_code=503, detail="Agent is not available.")

    try:
        # We need the PlanRun object to find the specific clarification
        plan_run = agent.portia.storage.get_plan_run(run_id)
        if not plan_run:
            raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")

        # Find the specific clarification object by ID within the run's clarifications
        clarification_to_resolve = next(
            (c for c in plan_run.outputs.clarifications if str(c.id) == payload.clarification_id),
            None,
        )

        if not clarification_to_resolve:
             logger.error(f"Clarification ID {payload.clarification_id} not found within run {run_id}")
             raise HTTPException(status_code=404, detail=f"Clarification ID {payload.clarification_id} not found for run {run_id}.")
        
        # Call the synchronous portia.resolve_clarification method
        # It updates the run state internally (e.g., to READY_TO_RESUME if all resolved)
        # We pass the actual Clarification object found
        agent.portia.resolve_clarification(
            clarification=clarification_to_resolve,
            response=payload.response,
            plan_run=plan_run # Pass the run object to avoid re-fetching
        )
        
        logger.info(f"Clarification {payload.clarification_id} resolved successfully for run {run_id}.")
        
        # Return simple success - frontend should call /resume next
        return JSONResponse(content={
            "message": f"Clarification {payload.clarification_id} resolved successfully.",
            "run_id": run_id
        })

    except PlanRunNotFoundError:
         logger.warning(f"Run ID {run_id} not found during clarification attempt.")
         raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
    # Add specific exception for clarification not found?
    # Portia's resolve_clarification might raise InvalidPlanRunStateError if clarification doesn't match
    except InvalidPlanRunStateError as ipse:
         logger.warning(f"Error resolving clarification for run {run_id}: {ipse}")
         raise HTTPException(status_code=400, detail=f"Error resolving clarification: {ipse}")
    except PortiaBaseError as pe:
        logger.error(f"Portia error resolving clarification for run {run_id}: {pe}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Portia-related error resolving clarification: {pe}")
    except Exception as e:
        logger.error(f"Unexpected error resolving clarification for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to resolve clarification for run {run_id}: {e}")

# @app.post("/clarify") # Remove or comment out old placeholder if exists

# --- Run Application --- #

if __name__ == "__main__":
    logger.info("Starting Scientific Workflow Agent server...")
    # Ensure necessary API keys are set (add more checks as needed)
    # Simplified check - assumes at least one key is needed
    api_keys = [os.getenv(k) for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRALAI_API_KEY", "GOOGLE_API_KEY"]]
    if not any(api_keys):
        logger.warning("Warning: No primary LLM API key found in environment variables.")
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("Warning: TAVILY_API_KEY not found. Search tool might not function.")

    uvicorn.run(app, host="0.0.0.0", port=8000) 