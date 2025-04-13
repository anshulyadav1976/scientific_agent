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
import pandas as pd # Import pandas

# Import PlanRunState to check run status
# Remove unused state import for M1 - Re-enable for stateful flow
from portia import PlanRunState

# Import Portia models needed for status checking and response
# Remove unused imports for M1
# from portia import PortiaBaseError # Import error base class if needed for specific checks
# from portia.plan_run import PlanRun # Explicitly import PlanRun if needed for type hinting
# from portia import Plan # Added Plan
from portia import PlanNotFoundError # Re-enable PlanNotFoundError
from portia import PlanRunNotFoundError # Import PlanRunNotFoundError
from portia import InvalidPlanRunStateError # Added for InvalidPlanRunStateError

# Import the summarizer - Not needed for M1
# from portia.execution_agents.utils.final_output_summarizer import FinalOutputSummarizer

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
# Make file optional for Milestone 1
async def handle_upload(
    request: Request, 
    prompt: str = Form(...),
    ollama_url: str = Form(...), # Add ollama_url form field
    file: Optional[UploadFile] = File(None), # Make file optional
    data_type_hint: Optional[str] = Form(None) # Get hint from form
):
    """Handles prompt submission, Ollama URL, and optional file upload."""
    logger.info(f"Received upload request. Prompt: '{prompt[:50]}...', Ollama: {ollama_url}")

    if agent is None:
        logger.error("Upload attempt failed: Agent not initialized.")
        raise HTTPException(status_code=503, detail="Agent is not available.")

    file_path_str: Optional[str] = None

    # --- File Handling Block (Keep for future, but bypass if no file) ---
    if file:
        logger.info(f"File received: {file.filename}, Hint: {data_type_hint}")
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
            file_path_str = str(file_path.resolve())
        except Exception as e:
            logger.error(f"Failed to save uploaded file {file_path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to save uploaded file.")
        finally:
            # Ensure file is closed even if saving fails
            await file.close()
    else:
        logger.info("No file provided with the request.")
    # --- End File Handling Block ---

    # Start the analysis using the agent
    try:
        logger.info(f"Calling agent.start_analysis with prompt and Ollama URL.")
        # Pass prompt, ollama_url, and optional file path
        run_id = await agent.start_analysis(
            user_prompt=prompt,
            ollama_url=ollama_url, # Pass the URL
            file_path=file_path_str
        )
        logger.info(f"Agent Stage 1 analysis planned. Run ID: {run_id}")

        # Return the run ID to the frontend
        return JSONResponse(content={
            "run_id": run_id
        })
    except Exception as e:
        logger.error(f"Failed to start agent analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {e}")

# --- Remove or comment out endpoints not needed for M1 --- #

# @app.get("/status/{run_id}")
# async def get_status(run_id: str):
#     """Gets the status of an ongoing analysis run."""
#     logger.info(f"Received status request for run_id: {run_id}")
#     if agent is None:
#         logger.error(f"Status request failed for {run_id}: Agent not initialized.")
#         raise HTTPException(status_code=503, detail="Agent is not available.")
#
#     try:
#         plan_run = await agent.get_run_status(run_id)
#         plan = None
#         if plan_run:
#             # Also fetch the corresponding plan to get step descriptions
#             # Access storage directly, get_plan is synchronous
#             try:
#                 plan = agent.portia.storage.get_plan(plan_run.plan_id)
#             except PlanNotFoundError:
#                  logger.error(f"Plan not found in storage for Plan ID {plan_run.plan_id}")
#             except Exception as e:
#                 logger.error(f"Error retrieving plan {plan_run.plan_id}: {e}", exc_info=True)
#
#         if plan_run is None:
#             logger.warning(f"Run ID not found: {run_id}")
#             raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
#         if plan is None:
#              # This shouldn't happen if plan_run exists, but handle defensively
#              logger.error(f"Plan not found for Run ID {run_id}, Plan ID {plan_run.plan_id if plan_run else 'N/A'}")
#              # Continue without thinking process if plan is missing
#
#         # --- Prepare Thinking Process History --- #
#         thinking_process = []
#         if plan:
#             # Iterate through steps up to the current one (or all if completed/failed)
#             max_step_index = len(plan.steps) if plan_run.state in [PlanRunState.COMPLETE, PlanRunState.FAILED] else plan_run.current_step_index + 1
#             for i in range(min(max_step_index, len(plan.steps))):
#                 step = plan.steps[i]
#                 step_detail = {
#                     "step_index": i,
#                     "description": step.task,
#                     "tool_id": step.tool_id,
#                     # Use getattr for safe access to potentially missing tool_args
#                     "tool_args": getattr(step, 'tool_args', {}), # Default to empty dict
#                     "status": "Pending", # Default status
#                     "output": None,
#                     "output_name": step.output
#                 }
#
#                 # Determine step status and output
#                 if i < plan_run.current_step_index or plan_run.state in [PlanRunState.COMPLETE, PlanRunState.FAILED]:
#                     step_detail["status"] = "Executed"
#                     # Try to get the output for this step using step.output as the key
#                     if step.output in plan_run.outputs.step_outputs:
#                          step_output_obj = plan_run.outputs.step_outputs[step.output]
#                          if hasattr(step_output_obj, 'value'):
#                             # Serialize complex objects like dicts/lists to string for display
#                             output_value = step_output_obj.value
#                             if isinstance(output_value, (dict, list, tuple, set)):
#                                 try:
#                                      step_detail["output"] = json.dumps(output_value, indent=2)
#                                 except TypeError:
#                                      step_detail["output"] = "(Output is complex/non-serializable)"
#                             else:
#                                 step_detail["output"] = str(output_value)
#                          else:
#                             step_detail["output"] = "(Output object lacks .value)"
#                     else:
#                          step_detail["output"] = "(Output not found in run)"
#                 elif i == plan_run.current_step_index and plan_run.state == PlanRunState.IN_PROGRESS:
#                     step_detail["status"] = "Executing"
#                 elif plan_run.state == PlanRunState.NEED_CLARIFICATION and i == plan_run.current_step_index:
#                      step_detail["status"] = "Paused (Needs Clarification)"
#                      # Optional: Add clarification details here if needed
#
#                 thinking_process.append(step_detail)
#         else:
#              logger.warning(f"Could not retrieve plan for run {run_id}, thinking process unavailable.")
#
#         # --- Prepare the main response content --- #
#         response_content = {
#             "run_id": str(plan_run.id),
#             "status": plan_run.state.value,
#             "raw_output": None, # Store raw final output here
#             "formatted_summary": None, # Store human-readable summary here
#             "error": None,
#             "clarification": None,
#             "thinking_process": thinking_process
#         }
#
#         # --- Handle FINAL Output/Error/Clarification based on Run State --- #
#
#         if plan_run.state == PlanRunState.COMPLETE:
#             logger.debug(f"Run {run_id} completed. Accessing final outputs and generating summary.")
#             extracted_raw_output = None
#             final_output_extracted = False
#             formatted_summary = "(Summary generation failed)" # Default summary on error
#             try:
#                 # --- Try to get final_output.value FIRST --- #
#                 final_output_obj = plan_run.outputs.final_output
#                 if final_output_obj and hasattr(final_output_obj, 'value'):
#                     extracted_raw_output = final_output_obj.value # Store the raw value
#                     final_output_extracted = True
#                     logger.info(f"Successfully extracted final_output.value for run {run_id}")
#
#                 # --- Fallback: If no final_output.value, check last step's output --- #
#                 elif not final_output_extracted and plan and plan_run.outputs.step_outputs:
#                     last_step_output_key = plan.steps[-1].output
#                     if last_step_output_key in plan_run.outputs.step_outputs:
#                          last_step_output_obj = plan_run.outputs.step_outputs[last_step_output_key]
#                          if hasattr(last_step_output_obj, 'value'):
#                             extracted_raw_output = last_step_output_obj.value # Store raw value
#                             final_output_extracted = True
#                             logger.info(f"Extracted output from last step '{last_step_output_key}' for run {run_id}")
#                         else:
#                             logger.warning(f"Last step output object for {run_id} lacks .value attribute")
#                     else:
#                          logger.warning(f"Output key '{last_step_output_key}' for last step not found in run {run_id}")
#                 else:
#                     logger.warning(f"Could not extract final_output or last step output for completed run {run_id}")
#
#                 # --- Store Raw Output --- #
#                 response_content["raw_output"] = extracted_raw_output
#                 # Serialize complex objects if needed for display (though ideally summary is used)
#                 # if isinstance(extracted_raw_output, (dict, list, tuple, set)):
#                 #    try: response_content["raw_output"] = json.dumps(extracted_raw_output)
#                 #    except TypeError: response_content["raw_output"] = "(Non-JSON serializable object)"
#                 # else:
#                 #    response_content["raw_output"] = str(extracted_raw_output)
#
#                 # --- Generate Formatted Summary --- #
#                 # Use the summarizer utility if raw output exists
#                 if final_output_extracted:
#                     summarizer = FinalOutputSummarizer(config=agent.config)
#                     try:
#                         formatted_summary = summarizer.summarize(str(extracted_raw_output))
#                         logger.info(f"Generated summary for run {run_id}: {formatted_summary[:100]}...")
#                     except Exception as summ_ex:
#                         logger.error(f"Error during summary generation for run {run_id}: {summ_ex}", exc_info=True)
#                         formatted_summary = "(Summary generation failed due to error)"
#                 else:
#                     formatted_summary = "(Could not generate summary: No final output extracted)"
#
#             except Exception as final_output_ex:
#                  logger.error(f"Error processing final output/summary for run {run_id}: {final_output_ex}", exc_info=True)
#                  response_content["error"] = f"Error processing final output: {final_output_ex}"
#
#             response_content["formatted_summary"] = formatted_summary
#
#         elif plan_run.state == PlanRunState.FAILED:
#             logger.warning(f"Run {run_id} failed. Reporting error.")
#             # Attempt to find error information in the run outputs or context (if available)
#             # This structure might change based on how Portia reports errors
#             error_info = plan_run.outputs.get("error_details", "Unknown error") # Example key
#             response_content["error"] = f"Run failed: {error_info}"
#
#         elif plan_run.state == PlanRunState.NEED_CLARIFICATION:
#             logger.info(f"Run {run_id} needs clarification.")
#             # Extract the *first* unresolved clarification
#             outstanding_clarifications = plan_run.get_outstanding_clarifications()
#             if outstanding_clarifications:
#                 clarification = outstanding_clarifications[0]
#                 response_content["clarification"] = {
#                     "id": str(clarification.id),
#                     "prompt": clarification.prompt
#                     # Add other clarification details if needed by the frontend
#                 }
#             else:
#                 # This case shouldn't normally happen if state is NEED_CLARIFICATION
#                 logger.error(f"Run {run_id} is in NEED_CLARIFICATION state but no outstanding clarifications found.")
#                 response_content["error"] = "Internal state error: Needs clarification but none found."
#
#         return JSONResponse(content=response_content)
#
#     except PlanRunNotFoundError:
#         logger.warning(f"Run ID not found during status check: {run_id}")
#         raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
#     except Exception as e:
#         logger.error(f"Unexpected error getting status for {run_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error checking status: {e}")
#
#
# @app.post("/resume/{run_id}")
# async def resume_run(run_id: str):
#     """Resumes a paused or newly created plan run."""
#     logger.info(f"Received resume request for run_id: {run_id}")
#     if agent is None:
#         logger.error(f"Resume request failed for {run_id}: Agent not initialized.")
#         raise HTTPException(status_code=503, detail="Agent is not available.")
#
#     try:
#         # Retrieve the current run to check its state before resuming
#         current_run = await agent.get_run_status(run_id)
#         if not current_run:
#              raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found to resume.")
#
#         # Allow resuming if NOT_STARTED or READY_TO_RESUME
#         allowed_states = [PlanRunState.NOT_STARTED, PlanRunState.READY_TO_RESUME]
#         if current_run.state not in allowed_states:
#             logger.warning(f"Cannot resume run {run_id} from state {current_run.state}. Allowed: {allowed_states}")
#             raise HTTPException(status_code=400, detail=f"Cannot resume run from state: {current_run.state}. Run must be NOT_STARTED or READY_TO_RESUME.")
#
#         # Portia's resume handles the logic
#         resumed_run = await agent.resume_run(plan_run_id=run_id) # This is synchronous
#         logger.info(f"Run {run_id} resumed. New state: {resumed_run.state}")
#         # Return the status immediately after resuming
#         # The frontend will start polling based on this
#         return JSONResponse(content={
#             "message": f"Run {run_id} resumed successfully.",
#             "run_id": str(resumed_run.id),
#             "status": resumed_run.state.value
#         })
#     except (PlanRunNotFoundError, InvalidPlanRunStateError) as pe:
#         logger.warning(f"Error resuming run {run_id}: {pe}")
#         raise HTTPException(status_code=400, detail=str(pe))
#     except Exception as e:
#         logger.error(f"Unexpected error resuming run {run_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error resuming run: {e}")
#
# @app.post("/clarify/{run_id}")
# async def handle_clarification(run_id: str, payload: ClarificationPayload):
#     """Handles user responses to agent clarifications."""
#     logger.info(f"Received clarification response for run_id: {run_id}, clarification_id: {payload.clarification_id}")
#     if agent is None:
#         logger.error(f"Clarification attempt failed for {run_id}: Agent not initialized.")
#         raise HTTPException(status_code=503, detail="Agent is not available.")
#
#     try:
#         # Use the agent's method to resolve clarification
#         # Note: The agent's resolve_clarification might need adjustment if it expects
#         # the PlanRun object, or if it needs to call portia.resume afterwards.
#         # Assuming agent.resolve_clarification handles finding the run and calling Portia.
#         # If agent.resolve_clarification is async as before:
#         # updated_run = await agent.resolve_clarification(payload.clarification_id, payload.response)

#         # Let's assume Portia's resolve_clarification is synchronous and just updates the state
#         # Find the clarification first (might be complex depending on storage)
#         run = await agent.get_run_status(run_id)
#         if not run:
#              raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
#         
#         clarification_found = False
#         for clarification in run.outputs.clarifications:
#             if str(clarification.id) == payload.clarification_id and not clarification.resolved:
#                 # Use Portia directly to resolve - assuming resolve_clarification is sync
#                 updated_run = await agent.resolve_clarification(
#                     clarification=clarification, 
#                     response=payload.response, 
#                     plan_run=run
#                 )
#                 clarification_found = True
#                 logger.info(f"Clarification {payload.clarification_id} resolved for run {run_id}. New state: {updated_run.state}")
#                 # Return status immediately, frontend should call /resume if state is READY_TO_RESUME
#                 return JSONResponse(content={
#                     "message": f"Clarification resolved. Run state is now {updated_run.state.value}",
#                     "run_id": str(updated_run.id),
#                     "status": updated_run.state.value
#                 })
#                 break # Exit loop once found and resolved
#
#         if not clarification_found:
#             logger.warning(f"Outstanding clarification ID {payload.clarification_id} not found for run {run_id}")
#             raise HTTPException(status_code=404, detail=f"Clarification ID {payload.clarification_id} not found or already resolved.")
#
#     except (PlanRunNotFoundError, InvalidPlanRunStateError) as pe:
#         logger.warning(f"Error handling clarification for run {run_id}: {pe}")
#         raise HTTPException(status_code=400, detail=str(pe))
#     except Exception as e:
#         logger.error(f"Unexpected error handling clarification for {run_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error handling clarification: {e}")

# --- Re-enable /status and /resume for Milestone 1 (Stateful Flow) --- #

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
            try:
                plan = await agent.get_plan(plan_run.plan_id)
            except PlanNotFoundError:
                 logger.error(f"Plan not found in storage for Plan ID {plan_run.plan_id}")
            except Exception as e:
                logger.error(f"Error retrieving plan {plan_run.plan_id}: {e}", exc_info=True)

        # --- Get Plan for Thinking Process --- #
        plan = None
        # Correct indentation for try block
        try:
            # Use the async agent method
            plan = await agent.get_plan(plan_run.plan_id)
        except PlanNotFoundError:
            logger.error(f"Plan not found in storage for Plan ID {plan_run.plan_id}")
            # Continue without thinking process if plan is missing, but log error
        except Exception as e:
            logger.error(f"Error retrieving plan {plan_run.plan_id}: {e}", exc_info=True)
            # Continue without thinking process

        # --- Prepare Thinking Process History --- #
        thinking_process = []
        if plan:
            # Iterate through steps up to the current one (or all if completed/failed)
            max_step_index = len(plan.steps) if plan_run.state in [PlanRunState.COMPLETE, PlanRunState.FAILED] else plan_run.current_step_index + 1
            for i in range(min(max_step_index, len(plan.steps))):
                step = plan.steps[i]
                # Correct indentation for step_detail dict
                step_detail = {
                    "step_index": i,
                    "description": step.task,
                    "tool_id": step.tool_id,
                    "status": "Pending", # Default status
                    "output": "(Not yet generated)", # Default output message
                    "output_name": step.output
                }

                # Determine step status and output
                # Correct indentation for if/elif block
                if i < plan_run.current_step_index or plan_run.state in [PlanRunState.COMPLETE, PlanRunState.FAILED]:
                    step_detail["status"] = "Executed"
                    # Try to get the output for this step using step.output as the key
                    if step.output in plan_run.outputs.step_outputs:
                        step_output_obj = plan_run.outputs.step_outputs[step.output]
                        output_value_str = "(Error reading output value)" # Default in case of error
                        if hasattr(step_output_obj, 'value'):
                            output_value = step_output_obj.value
                            # Serialize complex objects like dicts/lists to string for display
                            if isinstance(output_value, (dict, list, tuple, set)):
                                try:
                                    # Use json.dumps for consistent serialization
                                    output_value_str = json.dumps(output_value, indent=2)
                                except TypeError:
                                    output_value_str = "(Output is complex/non-JSON-serializable)"
                            else:
                                output_value_str = str(output_value)
                        else:
                            output_value_str = "(Output object lacks .value)"
                        step_detail["output"] = output_value_str
                    else:
                        step_detail["output"] = "(Output not found in run)"
                elif i == plan_run.current_step_index and plan_run.state == PlanRunState.IN_PROGRESS:
                    step_detail["status"] = "Executing"
                elif plan_run.state == PlanRunState.NEED_CLARIFICATION and i == plan_run.current_step_index:
                    step_detail["status"] = "Paused (Needs Clarification)"
                    step_detail["output"] = "(Awaiting user input)"
                
                thinking_process.append(step_detail)
        else:
            logger.warning(f"Could not retrieve plan for run {run_id}, thinking process unavailable.")

        # --- Prepare the basic response content --- #
        # Process step outputs to make them JSON serializable
        serializable_step_outputs = {}
        if plan_run.outputs.step_outputs:
            for name, output_obj in plan_run.outputs.step_outputs.items():
                if hasattr(output_obj, 'value'):
                    val = output_obj.value
                    # Attempt to serialize complex types, fallback to string
                    if isinstance(val, (dict, list, tuple, set)):
                        try:
                            serializable_step_outputs[name] = json.dumps(val) # Store as JSON string
                        except TypeError:
                             serializable_step_outputs[name] = str(val) # Fallback
                    else:
                         serializable_step_outputs[name] = str(val)
                else:
                     serializable_step_outputs[name] = "(Output object lacks .value)"

        serializable_final_output = None
        # Process final output as well
        if plan_run.state == PlanRunState.COMPLETE:
            final_output_obj = plan_run.outputs.final_output
            if final_output_obj and hasattr(final_output_obj, 'value'):
                val = final_output_obj.value
                if isinstance(val, (dict, list, tuple, set)):
                     try:
                         serializable_final_output = json.dumps(val) # Store as JSON string
                     except TypeError:
                         serializable_final_output = str(val) # Fallback
                else:
                     serializable_final_output = str(val)
            else:
                 logger.warning(f"Run {run_id} complete but no final_output.value found.")
                 serializable_final_output = "(Completed, but no final output available)"

        # Add clarification extraction
        clarification_info = None
        if plan_run.state == PlanRunState.NEED_CLARIFICATION:
            logger.info(f"Run {run_id} needs clarification.")
            outstanding = plan_run.get_outstanding_clarifications() # Corrected variable name
            if outstanding:
                # Get the first outstanding clarification
                clarification = outstanding[0]
                clarification_info = {
                    "id": str(clarification.id),
                    "prompt": clarification.prompt
                }
            else:
                 logger.error(f"Run {run_id} is NEED_CLARIFICATION but no outstanding clarifications found.")
                 # Optionally set an error state here?

        response_content = {
            "run_id": str(plan_run.id),
            "status": plan_run.state.value,
            "final_output": serializable_final_output, # Use processed value
            "error": None,
            "thinking_process": thinking_process,
            # Add step_outputs - Use the processed dictionary
            "step_outputs": serializable_step_outputs,
            "clarification": clarification_info, # Add clarification details
        }

        # --- Handle FINAL Output/Error based on Run State --- #
        if plan_run.state == PlanRunState.FAILED:
            logger.warning(f"Run {run_id} failed. Reporting error.")
            # TODO: Extract more specific error from plan_run if possible in future Portia versions
            response_content["error"] = f"Run failed. Check agent logs for details."

        return JSONResponse(content=response_content)

    except PlanRunNotFoundError:
        logger.warning(f"Run ID not found during status check: {run_id}")
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
    except Exception as e:
        logger.error(f"Unexpected error getting status for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error checking status: {e}")


@app.post("/resume/{run_id}")
async def resume_run(run_id: str):
    """Resumes a paused or newly created plan run."""
    logger.info(f"Received resume request for run_id: {run_id}")
    if agent is None:
        logger.error(f"Resume request failed for {run_id}: Agent not initialized.")
        raise HTTPException(status_code=503, detail="Agent is not available.")

    try:
        # Retrieve the current run to check its state before resuming (optional but good practice)
        current_run = await agent.get_run_status(run_id)
        # Allow resuming if NOT_STARTED or READY_TO_RESUME
        allowed_states = [PlanRunState.NOT_STARTED, PlanRunState.READY_TO_RESUME]
        if current_run.state not in allowed_states:
            logger.warning(f"Cannot resume run {run_id} from state {current_run.state}. Allowed: {allowed_states}")
            # Return a specific error or maybe just the current status?
            # Let's return current status, frontend polling will handle it.
            # Or raise HTTP 400:
            raise HTTPException(status_code=400, detail=f"Cannot resume run from state: {current_run.state}. Run must be NOT_STARTED or READY_TO_RESUME.")

        # Use the synchronous agent method - Now async
        resumed_run = await agent.resume_run(run_id)
        logger.info(f"Run {run_id} resume attempt completed. New state: {resumed_run.state}")

        # Return the status immediately after resuming
        # The frontend will start polling based on this
        return JSONResponse(content={
            "message": f"Run {run_id} resumed successfully.",
            "run_id": str(resumed_run.id),
            "status": resumed_run.state.value
        })
    except (PlanRunNotFoundError, InvalidPlanRunStateError) as pe:
        logger.warning(f"Error resuming run {run_id}: {pe}")
        # Use 404 for Not Found, 400 for Invalid State
        status_code = 404 if isinstance(pe, PlanRunNotFoundError) else 400
        raise HTTPException(status_code=status_code, detail=str(pe))
    except Exception as e:
        logger.error(f"Unexpected error resuming run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error resuming run: {e}")

# --- Re-enable /clarify endpoint --- #
@app.post("/clarify/{run_id}")
async def handle_clarification(run_id: str, payload: ClarificationPayload):
    """Handles user responses to agent clarifications."""
    logger.info(f"Received clarification response for run_id: {run_id}, clarification_id: {payload.clarification_id}")
    if agent is None:
        logger.error(f"Clarification attempt failed for {run_id}: Agent not initialized.")
        raise HTTPException(status_code=503, detail="Agent is not available.")

    try:
        # Use the agent's method to store response and update state
        updated_run = await agent.resolve_clarification(
            run_id=run_id,
            clarification_id=payload.clarification_id,
            response=payload.response
        )
        logger.info(f"Clarification response stored for run {run_id}. New state: {updated_run.state}")
        # Return status immediately, frontend should call /resume if state is READY_TO_RESUME
        return JSONResponse(content={
            "message": f"Clarification response received. Run state is now {updated_run.state.value}",
            "run_id": str(updated_run.id),
            "status": updated_run.state.value
        })

    except (PlanRunNotFoundError, InvalidPlanRunStateError, ValueError) as pe:
        # ValueError added for case where clarification ID not found in agent method
        logger.warning(f"Error handling clarification for run {run_id}: {pe}")
        status_code = 404 if isinstance(pe, PlanRunNotFoundError) or isinstance(pe, ValueError) else 400
        raise HTTPException(status_code=status_code, detail=str(pe))
    except Exception as e:
        logger.error(f"Unexpected error handling clarification for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error handling clarification: {e}")

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