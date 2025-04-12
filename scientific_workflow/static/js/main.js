// Frontend JavaScript logic for Scientific Workflow Agent
console.log("Scientific Workflow Agent JS loaded.");

const uploadForm = document.getElementById('upload-form');
const statusArea = document.getElementById('status-area');
const statusContent = document.getElementById('status-content');
const resultsArea = document.getElementById('results-area');
const resultsContent = document.getElementById('results-content');
const clarificationArea = document.getElementById('clarification-area');
const clarificationForm = document.getElementById('clarification-form');
const clarificationQuestion = document.getElementById('clarification-question');
const clarificationIdInput = document.getElementById('clarification-id');
const clarificationResponse = document.getElementById('clarification-response');
const thinkingArea = document.getElementById('thinking-area');
const thinkingContent = document.getElementById('thinking-content');

let currentRunId = null;
let pollingInterval = null;
const POLLING_INTERVAL_MS = 3000; // Poll every 3 seconds

// --- Helper Functions --- 

function updateStatus(message) {
    statusContent.textContent = message;
}

function displayResults(data) {
    console.log('displayResults called. Data received:', data);
    console.log('Type of data.formatted_summary:', typeof data.formatted_summary);
    console.log('Value of data.formatted_summary:', data.formatted_summary);
    // console.log('Type of data.raw_output:', typeof data.raw_output);
    // console.log('Value of data.raw_output:', data.raw_output);

    resultsContent.textContent = ''; // Clear previous results
    // Use formatted_summary as the primary display
    let displayContent = `Run ID: ${data.run_id}\nStatus: ${data.status}\n\n`;

    if (data.formatted_summary) {
         displayContent += "--- Analysis Summary ---\n";
         displayContent += data.formatted_summary;
    } 
    // Fallback if summary is missing but raw output exists (less likely now)
    else if (data.raw_output !== null && data.raw_output !== undefined) {
        displayContent += "--- Raw Output Data ---\n";
        if (typeof data.raw_output === 'object') {
             try { displayContent += JSON.stringify(data.raw_output, null, 2); }
             catch (e) { displayContent += "(Could not display raw object)"; }
        } else {
             displayContent += String(data.raw_output);
        }
    } else if (data.error) {
        displayContent += `--- Error ---\n${data.error}`;
    } else {
        // Handle case where run is COMPLETE but both summary and raw output are missing
        if (data.status === "COMPLETE") {
             displayContent += "(Run completed, but no output data or summary was found)";
        } else {
             displayContent += "(No output, summary or error information received)";
        }
    }
    console.log('Final displayContent for Results:', displayContent);

    resultsContent.textContent = displayContent; // Set the text
    // Ensure resultsArea is shown and others hidden (might be redundant but safe)
    if (resultsArea) resultsArea.style.display = 'block';        
    if (statusArea) statusArea.style.display = 'none';         
    if (clarificationArea) clarificationArea.style.display = 'none';   
}

function displayClarification(clarification) {
    clarificationQuestion.textContent = clarification.prompt;
    clarificationIdInput.value = clarification.id;
    clarificationArea.style.display = 'block';
    statusArea.style.display = 'none';
    resultsArea.style.display = 'none';
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
        console.log("Polling stopped.");
    }
}

// --- New Function: Display Thinking Process --- 
function displayThinkingProcess(steps) {
    if (!thinkingArea || !thinkingContent) return; // Exit if elements not found

    if (!steps || steps.length === 0) {
        thinkingContent.textContent = "(No thinking process steps available yet)";
        thinkingArea.style.display = 'block'; // Show the area even if empty
        return;
    }

    let formattedThinking = "";
    steps.forEach(step => {
        formattedThinking += `--- Step ${step.step_index} [${step.status}] ---\n`;
        formattedThinking += `Description: ${step.description}\n`;
        if (step.status !== 'Pending') { // Only show tool details if not pending
             formattedThinking += `Tool: ${step.tool_id}\n`;
             formattedThinking += `Args: ${JSON.stringify(step.tool_args)}\n`;
             if (step.output !== null && step.output !== undefined) {
                 // Display the raw output from the step
                 let outputStr = String(step.output); // Start with string conversion
                 try {
                     // Attempt to pretty-print if it's JSON 
                     const outputObj = JSON.parse(step.output);
                     // Convert back to string with indentation for display
                     outputStr = JSON.stringify(outputObj, null, 2); 
                 } catch (e) {
                     // If it wasn't valid JSON, stick with the simple string conversion
                     // (already assigned to outputStr)
                 }

                 // Display the output, potentially prefixed if it's the LLM summary
                 if (step.output_name === '$llm_summary') {
                    formattedThinking += `LLM Summary (${step.output_name}):\n${outputStr}\n`;
                 } else {
                    formattedThinking += `Output (${step.output_name}):\n${outputStr}\n`;
                 }
             } else {
                 formattedThinking += `Output (${step.output_name}): (Not available or not generated yet)\n`;
             }
        }
        formattedThinking += `\n`; // Add space between steps
    });

    thinkingContent.textContent = formattedThinking;
    thinkingArea.style.display = 'block'; // Ensure area is visible
}

// --- Status Polling (Modified) --- 

async function checkStatus() {
    if (!currentRunId) return;

    console.log(`Polling status for run_id: ${currentRunId}`);
    try {
        const response = await fetch(`/status/${currentRunId}`);
        if (!response.ok) {
            const errorData = await response.json();
            updateStatus(`Error checking status: ${errorData.detail || response.statusText}`);
            thinkingArea.style.display = 'none'; // Hide thinking on error
            stopPolling();
            return;
        }

        const data = await response.json();
        console.log('Received status data:', data);
        updateStatus(`Current status: ${data.status}`);

        // **** Update Thinking Process on every poll ****
        if (data.thinking_process) {
            displayThinkingProcess(data.thinking_process);
        } else {
            // Handle case where thinking process might be missing
             if (thinkingArea) thinkingArea.style.display = 'none';
        }

        // Terminal states
        const terminalStates = ["COMPLETE", "FAILED", "CANCELLED"];
        if (terminalStates.includes(data.status)) {
            console.log(`Run ${currentRunId} reached terminal state: ${data.status}`);
            stopPolling();
            console.log('Calling displayResults with data:', data);
            // Display final results (which hides status and clarification)
            displayResults(data);
             // Keep thinking area visible
             if (thinkingArea) thinkingArea.style.display = 'block';
        } else if (data.status === "NEED_CLARIFICATION") {
            console.log(`Run ${currentRunId} needs clarification.`);
            stopPolling();
            // Display clarification (which hides status and results)
            if (data.clarification) {
                displayClarification(data.clarification);
            } else {
                 updateStatus("Clarification needed, but no details provided.");
            }
             // Keep thinking area visible
             if (thinkingArea) thinkingArea.style.display = 'block';
        } else {
             // For IN_PROGRESS, NOT_STARTED, etc.
             // Keep status visible, hide results/clarification
             if (resultsArea) resultsArea.style.display = 'none';
             if (clarificationArea) clarificationArea.style.display = 'none';
             if (statusArea) statusArea.style.display = 'block';
        }

    } catch (error) {
        console.error("Error during status polling:", error);
        updateStatus(`Polling error: ${error.message}`);
        // Decide whether to stop polling on error or keep trying
        // stopPolling(); 
    }
}

function startPolling(runId) {
    stopPolling(); // Clear any previous interval
    currentRunId = runId;
    console.log(`Starting polling for run_id: ${currentRunId}`);
    // Initial check immediately, wrapped in try/catch
    try {
        checkStatus();
    } catch(initialCheckError) {
        console.error("Error during initial status check:", initialCheckError);
        updateStatus(`Error during initial check: ${initialCheckError.message}`);
    }
    // Set interval for subsequent checks
    pollingInterval = setInterval(checkStatus, POLLING_INTERVAL_MS);
}

// --- New Function: Resume Run ---
async function resumeRun(runId) {
    console.log(`Attempting to resume run_id: ${runId}`);
    updateStatus(`Attempting to resume run ${runId}...`);
    thinkingArea.style.display = 'block'; // Show thinking area during resume/run
    clarificationArea.style.display = 'none'; // Hide clarification form
    resultsArea.style.display = 'none'; // Hide results area
    statusArea.style.display = 'block'; // Ensure status is visible

    try {
        const response = await fetch(`/resume/${runId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json' // Although no body, good practice
            }
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error(`Failed to resume run ${runId}:`, errorData);
            updateStatus(`Error resuming run ${runId}: ${errorData.detail || response.statusText}`);
            stopPolling(); // Stop polling if resume fails
            return; // Exit the function
        }

        const data = await response.json(); // Get response from resume (might include initial status)
        console.log(`Resume request successful for ${runId}. Server response:`, data);
        updateStatus(`Run ${runId} resuming... Polling for status.`); // Update status
        startPolling(runId); // Start polling *after* successful resume call

    } catch (error) {
        console.error(`Network or other error resuming run ${runId}:`, error);
        updateStatus(`Error resuming run ${runId}: ${error.message}`);
        stopPolling(); // Stop polling on critical errors
    }
}

// --- Event Listeners --- 

if (uploadForm) {
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission
        stopPolling(); // Stop any previous polling

        const formData = new FormData(uploadForm);
        updateStatus("Uploading file and initializing analysis plan...");
        resultsArea.style.display = 'none';
        clarificationArea.style.display = 'none';
        thinkingArea.style.display = 'none'; // Hide thinking initially
        statusArea.style.display = 'block';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            // Store runId globally immediately after getting it
            currentRunId = data.run_id; 
            console.log(`Upload successful. Received run_id: ${currentRunId}`);
            updateStatus(`Plan created for Run ID: ${currentRunId}. Initiating execution...`);
            
            // *** Call resumeRun instead of startPolling ***
            await resumeRun(currentRunId); 

        } catch (error) {
            console.error("Upload error:", error);
            updateStatus(`Upload failed: ${error.message}`);
            currentRunId = null; // Reset run ID on upload failure
        }
    });
}

// Add clarification form submission handler
if (clarificationForm) {
    clarificationForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        if (!currentRunId) {
            updateStatus("Error: No active run ID found for clarification.");
            return;
        }

        const clarificationId = clarificationIdInput.value;
        const userResponse = clarificationResponse.value; // Get value from the response input

        if (!clarificationId || !userResponse) {
            updateStatus("Error: Clarification ID or response is missing.");
            // Optionally add visual feedback to the form
            return;
        }

        console.log(`Submitting clarification for run ${currentRunId}, clarification ${clarificationId}`);
        updateStatus("Submitting clarification...");
        
        try {
             const response = await fetch(`/clarify/${currentRunId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    clarification_id: clarificationId, 
                    response: userResponse 
                }),
            });

            if (!response.ok) {
                 const errorData = await response.json();
                 throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            console.log("Clarification submitted successfully:", data);
            updateStatus(`Clarification submitted for ${currentRunId}. Resuming run...`);
            clarificationArea.style.display = 'none'; // Hide form on success
            clarificationResponse.value = ''; // Clear the input field
            
            // *** Call resumeRun to continue processing ***
            await resumeRun(currentRunId);

        } catch (error) {
             console.error("Clarification submission error:", error);
             updateStatus(`Clarification failed: ${error.message}`);
             // Keep the clarification form visible so the user can retry or see the error
        }
    });
} 