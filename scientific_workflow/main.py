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

# Import PlanRunState to check run status
from portia import PlanRunState

# Import Portia models needed for status checking and response
from portia import PortiaBaseError # Import error base class if needed for specific checks
from portia.plan_run import PlanRun # Explicitly import PlanRun if needed for type hinting
from portia import Plan # Added Plan
from portia import PlanNotFoundError # Added for PlanNotFoundError

from agents.research_agent import ResearchAgent

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
async def handle_upload(request: Request, file: UploadFile = File(...), prompt: str = Form(...)):
    """Handles file upload and starts the analysis workflow."""
    logger.info(f"Received upload request. Filename: {file.filename}, Prompt: '{prompt[:50]}...'")

    if agent is None:
        logger.error("Upload attempt failed: Agent not initialized.")
        raise HTTPException(status_code=503, detail="Agent is not available.")

    # Basic validation (allow common CSV types)
    allowed_content_types = ["text/csv", "application/vnd.ms-excel"]
    if file.content_type not in allowed_content_types:
        logger.warning(f"Upload failed: Invalid content type {file.content_type}")
        raise HTTPException(status_code=400, detail=f"Invalid file type. Please upload a CSV file. Got: {file.content_type}")

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
        logger.info(f"Calling agent.start_analysis with path: {file_path} and prompt.")
        # Pass the absolute path or path relative to where the script runs
        plan_run = await agent.start_analysis(str(file_path.resolve()), prompt)
        logger.info(f"Agent analysis started. Run ID: {plan_run.id}, Initial Status: {plan_run.state}")
        # Return the run ID and initial status
        return JSONResponse(content={
            "message": "Analysis started successfully!",
            # Convert PlanRunUUID to string for JSON serialization
            "run_id": str(plan_run.id),
            "status": plan_run.state.value # Use .value for enum serialization
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
            "output": None, # Final output (set below)
            "error": None,  # Final error (set below)
            "clarification": None, # Current clarification (set below)
            "thinking_process": thinking_process # Add the history
        }

        # --- Handle FINAL Output/Error/Clarification based on Run State --- #

        if plan_run.state == PlanRunState.COMPLETE:
            logger.debug(f"Run {run_id} completed. Accessing final outputs.")
            extracted_output_data = None
            final_output_extracted = False # Flag to track if we got final_output
            try:
                # --- Try to get final_output.value FIRST --- 
                final_output_obj = plan_run.outputs.final_output
                if final_output_obj and hasattr(final_output_obj, 'value'):
                    # Prioritize this value, convert to string for display
                    extracted_output_data = str(final_output_obj.value)
                    final_output_extracted = True
                    logger.info(f"Successfully extracted final_output.value for run {run_id}")
                
                # --- Fallback to first step_output ONLY IF final_output failed ---
                if not final_output_extracted and plan_run.outputs.step_outputs:
                    logger.warning(f"final_output not found or lacked .value for {run_id}, falling back to step outputs.")
                    try:
                        first_step_output_key = next(iter(plan_run.outputs.step_outputs))
                        first_step_output_object = plan_run.outputs.step_outputs[first_step_output_key]
                        if hasattr(first_step_output_object, 'value'):
                            extracted_output_data = first_step_output_object.value
                            # No need to stringify here, JS handles the object display
                            logger.debug(f"Using value from first step output '{first_step_output_key}' for run {run_id}")
                        else:
                             logger.warning(f"Fallback step output for {run_id} lacks .value attribute.")
                    except StopIteration:
                         logger.warning(f"Fallback step_outputs dict for {run_id} is empty.")
                elif not final_output_extracted:
                     logger.warning(f"Completed run {run_id} has no usable final_output and empty step_outputs.")

                response_content["output"] = extracted_output_data
            except Exception as ex:
                 logger.error(f"Error extracting FINAL output data for completed run {run_id}: {ex}", exc_info=True)
                 response_content["output"] = None # Ensure output is None on error
                 response_content["error"] = "Error extracting final output data after completion."

        elif plan_run.state == PlanRunState.FAILED:
            # (Existing logic)
            logger.warning(f"Run {run_id} failed.")
            response_content["error"] = "The analysis run failed. Check server logs for details."

        elif plan_run.state == PlanRunState.NEED_CLARIFICATION:
            # (Existing logic)
             logger.info(f"Run {run_id} needs clarification.")
             try:
                 outstanding_clarifications = plan_run.get_outstanding_clarifications()
                 if outstanding_clarifications:
                     clarification_to_send = outstanding_clarifications[0]
                     response_content["clarification"] = clarification_to_send.model_dump()
                 else:
                    response_content["error"] = "Agent needs clarification, but details are unavailable."
             except Exception as ex:
                 logger.error(f"Error extracting clarification details for run {run_id}: {ex}", exc_info=True)
                 response_content["error"] = "Error retrieving clarification details."

        # Intermediate states already handled by default None values

        return JSONResponse(content=response_content)

    except PortiaBaseError as pe:
        # Catch specific Portia errors if needed, e.g., PlanRunNotFoundError if storage handled it that way
        logger.error(f"Portia error retrieving status for {run_id}: {pe}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Portia-related error retrieving status: {pe}")
    except Exception as e:
        logger.error(f"Error retrieving status for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve status for run {run_id}: {e}")

# @app.post("/clarify")
# async def handle_clarification(request: Request):
#     """Handles user responses to agent clarifications."""
#     # Implementation in Phase 4
#     pass

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