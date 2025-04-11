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
    console.log('Type of data.output:', typeof data.output);
    console.log('Value of data.output:', data.output);

    resultsContent.textContent = ''; // Clear previous results
    let formattedOutput = `Run ID: ${data.run_id}\nStatus: ${data.status}\n\n`;

    if (data.output !== null && data.output !== undefined) { // Check explicitly for non-null/undefined
        // Check if the output is a simple string (likely the final summary)
        if (typeof data.output === 'string') {
            formattedOutput += "--- Analysis Summary ---\n";
            formattedOutput += data.output; // Display the summary directly
        }
        // Check if it's the object from DataIngestionTool (fallback, check for 'value' containing expected keys)
        else if (typeof data.output === 'object' && data.output !== null && data.output.hasOwnProperty('shape') && data.output.hasOwnProperty('columns')) {
            formattedOutput += "--- Raw Data Summary (from step output) ---\n";
            formattedOutput += `Shape: ${data.output.shape || 'N/A'}\n`;
            formattedOutput += `Columns: ${(data.output.columns || []).join(', ')}\n\n`;
            formattedOutput += `Head:\n${data.output.head || 'N/A'}\n\n`;
            formattedOutput += `Info:\n${data.output.info || 'N/A'}\n\n`;
            formattedOutput += `Describe:\n${data.output.describe || 'N/A'}\n`;
        }
        // Handle other potential object outputs generically
        else if (typeof data.output === 'object') {
             formattedOutput += "--- Result Data (Object) ---\n";
             try {
                formattedOutput += JSON.stringify(data.output, null, 2); // Pretty print JSON
             } catch (e) {
                formattedOutput += "(Could not display complex object)";
             }
        } else {
             // Catch any other non-null/undefined types
             formattedOutput += "--- Result Data ---\n";
             formattedOutput += String(data.output);
        }
    } else if (data.error) {
        formattedOutput += `--- Error ---\n${data.error}`;
    } else {
        // Handle case where run is COMPLETE but output is null/undefined
        if (data.status === "COMPLETE") {
             formattedOutput += "(Run completed, but no output data was found)";
        } else {
             formattedOutput += "(No output or error information received)"; // Should ideally not be hit for terminal states without error
        }
    }
    console.log('Final formattedOutput:', formattedOutput);

    resultsContent.textContent = formattedOutput; // Set the text
    resultsArea.style.display = 'block';        // Show results div
    statusArea.style.display = 'none';          // Hide status div
    clarificationArea.style.display = 'none';   // Hide clarification div
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
                 // Truncate long outputs for display
                 let outputStr = String(step.output);
                 const maxLength = 500; // Max length for step output display
                 if (outputStr.length > maxLength) {
                     outputStr = outputStr.substring(0, maxLength) + '... (truncated)';
                 }
                 formattedThinking += `Output (${step.output_name}):\n${outputStr}\n`;
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

// --- Event Listeners --- 

if (uploadForm) {
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission
        stopPolling(); // Stop any previous polling

        const formData = new FormData(uploadForm);
        updateStatus("Uploading file and starting analysis...");
        resultsArea.style.display = 'none';
        clarificationArea.style.display = 'none';
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
            updateStatus(`Analysis started! Run ID: ${data.run_id}. Status: ${data.status}. Polling for updates...`);
            startPolling(data.run_id);

        } catch (error) {
            console.error("Upload error:", error);
            updateStatus(`Upload failed: ${error.message}`);
        }
    });
}

// Add clarification form submission handler later in Phase 4
if (clarificationForm) {
    clarificationForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        // TODO: Implement clarification submission logic in Phase 4
        console.log("Clarification form submitted - logic TBD");
        // Fetch call to POST /clarify
        // Handle response, potentially restart polling
    });
} 