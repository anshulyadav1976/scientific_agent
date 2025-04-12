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
// Add a variable to track displayed step outputs for the current run
let displayedStepOutputs = {}; // Key: runId, Value: Set of output names displayed

// --- Helper Functions --- 

// Display status messages
function updateStatus(message) {
    if (!statusArea || !statusContent) return;
    console.log("Status Update:", message);
    statusContent.textContent = message;
    statusArea.style.display = 'block'; // Make sure status is visible
}

// Add messages to the main chat/results area
function addChatMessage(sender, message) {
    if (!resultsArea || !resultsContent) return;
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender.toLowerCase());
    const senderStrong = document.createElement('strong');
    senderStrong.textContent = sender + ": ";
    messageElement.appendChild(senderStrong);
    // Handle potential objects in message, display as string
    if (typeof message === 'object' && message !== null) {
        try { message = JSON.stringify(message, null, 2); } catch (e) { message = String(message); }
    }
    messageElement.appendChild(document.createTextNode(String(message))); // Ensure it's a string
    resultsContent.appendChild(messageElement);
    resultsArea.style.display = 'block';
    resultsContent.scrollTop = resultsContent.scrollHeight;
}

// Hide areas not used initially or when run finishes
function hideAuxiliaryAreas(){
    if (clarificationArea) clarificationArea.style.display = 'none';
    // Don't hide thinkingArea by default, let displayThinkingProcess manage it
    // if (thinkingArea) thinkingArea.style.display = 'none';
}

// Stop the polling interval
function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
        console.log("Polling stopped.");
    }
}

// --- New Function: Display Thinking Process --- 
function displayThinkingProcess(steps) {
    if (!thinkingArea || !thinkingContent) {
        console.warn("Thinking process display elements not found.");
        return;
    }

    if (!steps || !Array.isArray(steps)) {
        thinkingContent.innerHTML = '<p>(Thinking process data unavailable)</p>';
        thinkingArea.style.display = 'block'; // Show area even if empty/error
        return;
    }
    if (steps.length === 0) {
        thinkingContent.innerHTML = '<p>(No plan steps generated yet)</p>';
        thinkingArea.style.display = 'block'; // Show area even if empty
        return;
    }

    let htmlContent = '';
    steps.forEach(step => {
        // Determine status class for styling
        let statusClass = 'pending';
        if (step.status === 'Executed') statusClass = 'executed';
        else if (step.status === 'Executing') statusClass = 'executing';
        else if (step.status === 'Paused (Needs Clarification)') statusClass = 'paused';
        else if (step.status === 'Failed') statusClass = 'failed'; // Assuming FAILED is possible

        htmlContent += `<div class="step-item ${statusClass}">`;
        htmlContent += `  <div class="step-header">Step ${step.step_index}: [${step.status}]</div>`;
        htmlContent += `  <div class="step-body">`;
        htmlContent += `    <p><strong>Task:</strong> ${escapeHtml(step.description)}</p>`;
        if (step.tool_id) {
             htmlContent += `    <p><strong>Tool:</strong> ${escapeHtml(step.tool_id)}</p>`;
        }
        // Display output if not pending/executing (and exists)
        if (step.status !== 'Pending' && step.status !== 'Executing' && step.output) {
            // Use <pre> for preformatted text, good for JSON or code output
            htmlContent += `    <p><strong>Output (${escapeHtml(step.output_name || '')}):</strong></p>`;
            htmlContent += `    <pre>${escapeHtml(step.output)}</pre>`;
        }
        htmlContent += `  </div>`; // end step-body
        htmlContent += `</div>`; // end step-item
    });

    thinkingContent.innerHTML = htmlContent;
    thinkingArea.style.display = 'block'; // Ensure area is visible when there's content
}

// Simple HTML escaping function
function escapeHtml(unsafe) {
    if (unsafe === null || unsafe === undefined) return '';
    return String(unsafe)
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
 }

// Function to reset displayed outputs when a new run starts
function resetDisplayedOutputs(runId) {
    displayedStepOutputs[runId] = new Set();
}

// --- Status Polling (Updated for M3 - Intermediate Outputs) ---

async function checkStatus() {
    if (!currentRunId) return;

    // Ensure we have a tracker for the current run
    if (!displayedStepOutputs[currentRunId]) {
        resetDisplayedOutputs(currentRunId);
    }

    console.log(`Polling status for run_id: ${currentRunId}`);
    try {
        const response = await fetch(`/status/${currentRunId}`);
        if (!response.ok) {
            const errorData = await response.json();
            updateStatus(`Error checking status: ${errorData.detail || response.statusText}`);
            addChatMessage("System", `Error checking status: ${errorData.detail || response.statusText}`);
            stopPolling();
            return;
        }

        const data = await response.json();
        console.log('Received status data:', data);
        updateStatus(`Run ${data.run_id} Status: ${data.status}`);

        // Update Thinking Process display
        if (data.thinking_process) {
            displayThinkingProcess(data.thinking_process);
        } else {
            if (thinkingArea) thinkingArea.style.display = 'none';
        }

        // **** Display NEW Intermediate Step Outputs ****
        if (data.step_outputs && typeof data.step_outputs === 'object') {
             const currentDisplayed = displayedStepOutputs[currentRunId];
             // Loop through step outputs received from backend
             // Use thinking_process to get the order if needed, but simple loop for now
             for (const outputName in data.step_outputs) {
                 // Check if this output hasn't been displayed yet for this run
                 if (!currentDisplayed.has(outputName)) {
                     const outputData = data.step_outputs[outputName];
                     // Check if the output object has a 'value' property
                     if (outputData && outputData.hasOwnProperty('value')) {
                          console.log(`Displaying intermediate output: ${outputName}`);
                          // Format the output value before adding
                          let displayValue = outputData.value;
                          if (typeof displayValue === 'object' && displayValue !== null) {
                              try { displayValue = JSON.stringify(displayValue, null, 2); } catch (e) { displayValue = String(displayValue); }
                          }
                          addChatMessage("Agent", `Output [${escapeHtml(outputName)}]:\n${displayValue}`);
                          currentDisplayed.add(outputName); // Mark as displayed
                     } else {
                          // Log if format is unexpected, but don't display
                          console.warn(`Step output '${outputName}' has unexpected format:`, outputData);
                     }
                 }
             }
        }

        // Terminal states
        const terminalStates = ["COMPLETE", "FAILED"];
        if (terminalStates.includes(data.status)) {
            console.log(`Run ${currentRunId} reached terminal state: ${data.status}`);
            stopPolling();

            // Display final output/error (only if different from last step output?)
            const finalOutputName = "$final_report"; // TODO: Update this if final output name changes
            if (data.status === "COMPLETE") {
                 // Check if the final output has already been displayed as a step output
                 const finalOutputAlreadyDisplayed = displayedStepOutputs[currentRunId]?.has(finalOutputName);
                 if (!finalOutputAlreadyDisplayed && data.final_output !== null) {
                     // Format the final output value before adding
                     let displayValue = data.final_output;
                     if (typeof displayValue === 'object' && displayValue !== null) {
                        try { displayValue = JSON.stringify(displayValue, null, 2); } catch (e) { displayValue = String(displayValue); }
                     }
                     addChatMessage("Agent", `Final Output:\n${displayValue}`);
                 } else if (data.final_output === null && displayedStepOutputs[currentRunId]?.size === 0) {
                     // Handle cases where it completes with no output at all
                      addChatMessage("Agent", "(Run completed without any output)");
                 }
            } else { // FAILED
                addChatMessage("System", data.error || "Run failed with unknown error.");
            }
            updateStatus(`Run ${currentRunId} finished with status: ${data.status}`);
            if (thinkingArea && data.thinking_process) thinkingArea.style.display = 'block';
             // Optional: Clear the tracker for the finished run
             // delete displayedStepOutputs[currentRunId];
        }
        // **** Clarification handling will be added in M4 ****
        // else if (data.status === "NEED_CLARIFICATION") { ... }

    } catch (error) {
        console.error("Error during status polling:", error);
        updateStatus(`Polling error: ${error.message}`);
        addChatMessage("System", `Polling error: ${error.message}`);
        stopPolling();
        if (thinkingArea) thinkingArea.style.display = 'none';
    }
}

// Start the polling loop
function startPolling(runId) {
    stopPolling(); // Clear any previous interval
    currentRunId = runId;
    console.log(`Starting polling for run_id: ${currentRunId}`);
    checkStatus(); // Initial check immediately
    pollingInterval = setInterval(checkStatus, POLLING_INTERVAL_MS);
}

// --- Resume Run --- (Called after getting run_id)
async function resumeRun(runId) {
    console.log(`Attempting to resume run_id: ${runId}`);
    updateStatus(`Attempting to execute run ${runId}...`);
    // Hide other areas, ensure status is visible
    hideAuxiliaryAreas();
    if (resultsArea) resultsArea.style.display = 'none'; // Hide results until polling shows output
    if (statusArea) statusArea.style.display = 'block';

    try {
        const response = await fetch(`/resume/${runId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error(`Failed to resume run ${runId}:`, errorData);
            updateStatus(`Error starting run ${runId}: ${errorData.detail || response.statusText}`);
            addChatMessage("System", `Error starting run: ${errorData.detail || response.statusText}`);
            stopPolling(); // Stop polling if resume fails
            return false; // Indicate failure
        }

        const data = await response.json(); // Get response from resume
        console.log(`Resume request successful for ${runId}. Server response:`, data);
        updateStatus(`Run ${runId} started (${data.status}). Polling for updates...`); // Update status
        return true; // Indicate success

    } catch (error) {
        console.error(`Network or other error resuming run ${runId}:`, error);
        updateStatus(`Error starting run ${runId}: ${error.message}`);
        addChatMessage("System", `Error starting run: ${error.message}`);
        stopPolling(); // Stop polling on critical errors
        return false; // Indicate failure
    }
}

// --- Event Listeners --- 

if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log("Form submitted.");
        stopPolling(); // Stop any previous run polling
        currentRunId = null; // Reset current run ID
        // Reset the tracker for displayed outputs
        displayedStepOutputs = {};

        // Clear previous results and hide areas
        if (resultsContent) resultsContent.innerHTML = '';
        hideAuxiliaryAreas();
        if (statusArea) statusArea.style.display = 'block'; // Show status area
        if (resultsArea) resultsArea.style.display = 'none'; // Hide results initially

        const formData = new FormData(uploadForm);
        const userPrompt = formData.get('prompt');

        // Display user prompt immediately
        addChatMessage("User", userPrompt);
        updateStatus("Sending request to agent to create plan...");

        try {
            // Step 1: Call /upload to get run_id
            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                console.error("Plan creation failed:", errorData);
                addChatMessage("System", `Error creating plan: ${errorData.detail || uploadResponse.statusText}`);
                updateStatus("Plan creation failed.");
                return;
            }

            const uploadData = await uploadResponse.json();
            const runId = uploadData.run_id;
            currentRunId = runId; // Set the new currentRunId
            resetDisplayedOutputs(currentRunId); // Initialize tracker for this run

            console.log("Received run_id:", runId);
            updateStatus(`Plan created (Run ID: ${runId}). Starting execution...`);

            // Step 2: Call /resume to start the run
            const resumeSuccess = await resumeRun(runId);

            // Step 3: If resume was successful, start polling
            if (resumeSuccess) {
                startPolling(runId);
            } else {
                // resumeRun function already updated status and logged error
                updateStatus(`Failed to start execution for Run ID: ${runId}.`);
            }

        } catch (error) {
            console.error("Error during form submission process:", error);
            addChatMessage("System", `Error: ${error.message}`);
            updateStatus("Request process error.");
        }
    });
} else {
    console.error("Upload form not found!");
}

// --- Remove clarification form listener for M1 --- //
/*
if (clarificationForm) {
    clarificationForm.addEventListener('submit', async (e) => {
       ...
    });
} else {
    console.error("Clarification form not found!");
}
*/

// Initial state setup on load
hideAuxiliaryAreas();
if(resultsContent) resultsContent.innerHTML = ''; // Clear results on load
if(statusContent) statusContent.textContent = 'Enter a prompt and optional file to start.'; 