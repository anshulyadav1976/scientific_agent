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

// --- Status Polling (Updated for M2) --- 

async function checkStatus() {
    if (!currentRunId) return;

    console.log(`Polling status for run_id: ${currentRunId}`);
    try {
        const response = await fetch(`/status/${currentRunId}`);
        if (!response.ok) {
            const errorData = await response.json();
            updateStatus(`Error checking status: ${errorData.detail || response.statusText}`);
            // Display error in chat as well?
            addChatMessage("System", `Error checking status: ${errorData.detail || response.statusText}`);
            stopPolling();
            return;
        }

        const data = await response.json();
        console.log('Received status data:', data);
        updateStatus(`Run ${data.run_id} Status: ${data.status}`);

        // **** Update Thinking Process ****
        if (data.thinking_process) {
            displayThinkingProcess(data.thinking_process);
        } else {
            // Optionally hide or show default message if no thinking process
            if (thinkingArea) thinkingArea.style.display = 'none'; 
        }

        // **** Display Intermediate Outputs will be added in M3 ****

        // Terminal states
        const terminalStates = ["COMPLETE", "FAILED"];
        if (terminalStates.includes(data.status)) {
            console.log(`Run ${currentRunId} reached terminal state: ${data.status}`);
            stopPolling();
            // Display final output or error
            if (data.status === "COMPLETE") {
                addChatMessage("Agent", data.final_output !== null ? data.final_output : "(Run completed without final output)");
            } else { // FAILED
                addChatMessage("System", data.error || "Run failed with unknown error.");
            }
            updateStatus(`Run ${currentRunId} finished with status: ${data.status}`);
            // Keep thinking panel visible at the end
            if (thinkingArea && data.thinking_process) thinkingArea.style.display = 'block'; 
        }
        // **** Clarification handling will be added in M4 ****
        // else if (data.status === "NEED_CLARIFICATION") { ... }

    } catch (error) {
        console.error("Error during status polling:", error);
        updateStatus(`Polling error: ${error.message}`);
        addChatMessage("System", `Polling error: ${error.message}`);
        stopPolling(); // Stop polling on fetch errors
        if (thinkingArea) thinkingArea.style.display = 'none'; // Hide thinking on error
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