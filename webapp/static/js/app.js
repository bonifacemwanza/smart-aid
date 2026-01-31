/**
 * Main application logic for Smart Aid Navigator
 * - API interactions
 * - UI updates
 * - Initialization
 */

// ===== API FUNCTIONS =====

/**
 * Perform a quick safety scan.
 */
async function quickScan() {
    document.getElementById('safety-text').textContent = 'Scanning...';

    try {
        const response = await fetch('/api/scan');
        const data = await response.json();
        const text = data.response;

        let level = 'safe';
        if (text.includes('STOP') || text.includes('!')) {
            level = 'danger';
        } else if (text.includes('Caution') || text.includes('Warning')) {
            level = 'warning';
        }

        updateSafety(text, level);
    } catch (error) {
        console.error('Scan error:', error);
        updateSafety('Scan failed', 'danger');
    }
}

/**
 * Ask a question or process a command.
 */
async function askQuestion(question) {
    const query = question || document.getElementById('query-input').value.trim();
    if (!query) return;

    const responseEl = document.getElementById('response-text');
    responseEl.textContent = 'Processing...';
    responseEl.classList.add('loading');

    try {
        const response = await fetch('/api/navigate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        const data = await response.json();
        setResponse(data.response);
    } catch (error) {
        console.error('Navigate error:', error);
        setResponse('Error: Could not process request.');
    }
}

/**
 * Check system status.
 */
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        const statusEl = document.getElementById('system-status');
        const textEl = document.getElementById('status-text');

        if (data.ready) {
            textEl.textContent = 'System ready: ' + data.model;
            statusEl.className = 'system-status ready';
        } else {
            textEl.textContent = 'Loading models...';
            statusEl.className = 'system-status';
        }
    } catch (e) {
        console.error('Status check error:', e);
        document.getElementById('status-text').textContent = 'Connection error';
        document.getElementById('system-status').className = 'system-status error';
    }
}

// ===== UI UPDATE FUNCTIONS =====

/**
 * Update safety panel with result.
 */
function updateSafety(text, level) {
    const box = document.getElementById('safety-box');
    const textEl = document.getElementById('safety-text');
    textEl.textContent = text;
    box.className = 'safety-panel ' + level;
    speak(text);
}

/**
 * Set response text and speak it.
 */
function setResponse(text) {
    const el = document.getElementById('response-text');
    el.textContent = text;
    el.classList.remove('loading');
    speak(text);
}

// ===== INITIALIZATION =====

document.addEventListener('DOMContentLoaded', function() {
    // Check system status
    checkStatus();

    // Set initial safety text
    document.getElementById('safety-text').textContent = 'Say "Hey Manso" or press SCAN';

    // Load saved calibration
    loadCalibration();
    const userVariants = getUserVariants();

    // Initialize wake word system
    if (initWakeWordRecognition()) {
        // Need user interaction to start audio - use first click
        document.body.addEventListener('click', function initAudio(e) {
            // Don't trigger if clicking calibration modal
            if (e.target.closest('#calibration-modal') || e.target.closest('#wake-status')) return;

            // Initialize TTS (required for Chrome/Safari)
            initTTS();

            startWakeWordListening();

            // Small delay to let TTS initialize
            setTimeout(() => {
                if (userVariants.length > 0) {
                    speak('Ready. Say Hey Manso anytime.');
                } else {
                    speak('Ready. Click the microphone icon to calibrate your voice, or say Hey Manso.');
                }
            }, 300);

            document.body.removeEventListener('click', initAudio);
        }, { once: false });

        if (userVariants.length > 0) {
            updateWakeStatus('Click to start', 'idle');
        } else {
            updateWakeStatus('Click to calibrate', 'idle');
        }
    }

    // Poll status periodically
    setInterval(checkStatus, 10000);
});
