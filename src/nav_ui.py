"""Navigation-focused web UI for blind users.

This UI is designed for actual navigation assistance:
1. Quick safety scan (always visible)
2. Voice commands for navigation
3. Step-by-step guidance using local LLM
"""

import sys
import threading
import time
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from src.config import Config
from src.smart_navigator import SmartNavigator

app = Flask(__name__)

# Global state
navigator: SmartNavigator | None = None
camera: cv2.VideoCapture | None = None
current_frame: np.ndarray | None = None
frame_lock = threading.Lock()

# HTML Template - Professional enterprise-grade design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Aid | Navigation Assistant</title>
    <style>
        :root {
            --primary: #00d4aa;
            --primary-dark: #00a888;
            --danger: #ef4444;
            --danger-dark: #dc2626;
            --warning: #f59e0b;
            --bg-primary: #0f0f14;
            --bg-secondary: #16161d;
            --bg-tertiary: #1e1e28;
            --bg-elevated: #252530;
            --text-primary: #f4f4f5;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --border: #27272a;
            --border-focus: #00d4aa;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }

        /* Header */
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 16px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .logo-icon svg {
            width: 18px;
            height: 18px;
            fill: var(--bg-primary);
        }

        .logo-text {
            font-size: 18px;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .logo-text span {
            color: var(--primary);
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 24px;
        }

        /* Status indicator in header */
        .voice-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 24px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .voice-indicator:hover {
            background: var(--bg-elevated);
            border-color: var(--text-muted);
        }

        .voice-indicator.listening {
            background: rgba(0, 212, 170, 0.1);
            border-color: var(--primary);
            color: var(--primary);
        }

        .voice-indicator.active {
            background: rgba(239, 68, 68, 0.1);
            border-color: var(--danger);
            color: var(--danger);
            animation: pulse-border 1.5s ease-in-out infinite;
        }

        .voice-indicator.calibrating {
            background: rgba(59, 130, 246, 0.1);
            border-color: #3b82f6;
            color: #3b82f6;
        }

        @keyframes pulse-border {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
        }

        .indicator-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
        }

        .voice-indicator.listening .indicator-dot {
            background: var(--primary);
            animation: pulse-dot 2s ease-in-out infinite;
        }

        .voice-indicator.active .indicator-dot {
            background: var(--danger);
            animation: pulse-dot 0.6s ease-in-out infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }

        /* Safety Panel */
        .safety-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .safety-panel.danger {
            border-color: var(--danger);
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), var(--bg-secondary));
        }

        .safety-panel.warning {
            border-color: var(--warning);
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), var(--bg-secondary));
        }

        .safety-panel.safe {
            border-color: var(--primary);
            background: linear-gradient(135deg, rgba(0, 212, 170, 0.08), var(--bg-secondary));
        }

        .safety-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .safety-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        .safety-text {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Video Feed */
        .video-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 20px;
        }

        .video-header {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .video-title {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .video-status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--primary);
        }

        .video-status::before {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--primary);
            border-radius: 50%;
        }

        #video-feed {
            width: 100%;
            display: block;
            max-height: 320px;
            object-fit: cover;
        }

        /* Action Buttons */
        .actions-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }

        .action-btn {
            padding: 20px 24px;
            font-size: 15px;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .action-btn svg {
            width: 20px;
            height: 20px;
        }

        .action-btn:active {
            transform: scale(0.98);
        }

        .btn-scan {
            background: var(--primary);
            color: var(--bg-primary);
        }

        .btn-scan:hover {
            background: var(--primary-dark);
        }

        .btn-voice {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }

        .btn-voice:hover {
            background: var(--bg-elevated);
            border-color: var(--text-muted);
        }

        .btn-voice.listening {
            background: var(--danger);
            color: white;
            border-color: var(--danger);
            animation: pulse-btn 1.5s ease-in-out infinite;
        }

        @keyframes pulse-btn {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            50% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        }

        /* Quick Commands */
        .quick-actions {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .quick-btn {
            padding: 10px 16px;
            font-size: 13px;
            font-weight: 500;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .quick-btn:hover {
            background: var(--bg-elevated);
            color: var(--text-primary);
            border-color: var(--text-muted);
        }

        /* Input Area */
        .input-panel {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .input-panel input {
            flex: 1;
            padding: 14px 18px;
            font-size: 15px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-primary);
            transition: all 0.15s ease;
        }

        .input-panel input::placeholder {
            color: var(--text-muted);
        }

        .input-panel input:focus {
            outline: none;
            border-color: var(--border-focus);
            box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.1);
        }

        .input-panel button {
            padding: 14px 24px;
            font-size: 14px;
            font-weight: 600;
            background: var(--primary);
            color: var(--bg-primary);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .input-panel button:hover {
            background: var(--primary-dark);
        }

        /* Response Panel */
        .response-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }

        .response-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }

        .response-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        .response-text {
            font-size: 18px;
            line-height: 1.6;
            color: var(--text-primary);
        }

        .response-text.loading {
            color: var(--warning);
        }

        /* System Status */
        .system-status {
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .system-status.ready {
            color: var(--primary);
        }

        .system-status.error {
            color: var(--danger);
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
        }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(4px);
            z-index: 200;
            justify-content: center;
            align-items: center;
        }

        .modal-overlay.show {
            display: flex;
        }

        .modal {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            max-width: 420px;
            width: 90%;
        }

        .modal-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }

        .modal-subtitle {
            font-size: 14px;
            color: var(--text-secondary);
            margin-bottom: 24px;
        }

        .modal p {
            font-size: 14px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            line-height: 1.6;
        }

        .modal strong {
            color: var(--primary);
        }

        .calibration-progress {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin: 24px 0;
        }

        .calibration-dot {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: var(--bg-elevated);
            border: 2px solid var(--border);
            transition: all 0.2s ease;
        }

        .calibration-dot.done {
            background: var(--primary);
            border-color: var(--primary);
        }

        .calibration-dot.current {
            border-color: #3b82f6;
            animation: pulse-cal 0.8s ease-in-out infinite;
        }

        @keyframes pulse-cal {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
            50% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(59, 130, 246, 0); }
        }

        .modal-actions {
            display: flex;
            gap: 10px;
            margin-top: 24px;
        }

        .modal-btn {
            flex: 1;
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .modal-btn.primary {
            background: var(--primary);
            color: var(--bg-primary);
        }

        .modal-btn.primary:hover {
            background: var(--primary-dark);
        }

        .modal-btn.secondary {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border: 1px solid var(--border);
        }

        .modal-btn.secondary:hover {
            background: var(--bg-elevated);
            color: var(--text-primary);
        }

        .modal-btn.text {
            background: transparent;
            color: var(--text-muted);
            font-size: 12px;
        }

        .modal-btn.text:hover {
            color: var(--text-secondary);
        }

        .heard-display {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            padding: 12px 16px;
            border-radius: 8px;
            margin: 16px 0;
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 14px;
            color: var(--warning);
        }

        .saved-list {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            margin: 16px 0;
            max-height: 140px;
            overflow-y: auto;
        }

        .saved-list div {
            padding: 6px 0;
            border-bottom: 1px solid var(--border);
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 13px;
            color: var(--text-secondary);
        }

        .saved-list div:last-child {
            border-bottom: none;
        }

        .instruction-text {
            font-size: 14px;
            color: var(--text-secondary);
            text-align: center;
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="logo">
            <div class="logo-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 6v6l4 2"/>
                </svg>
            </div>
            <div class="logo-text">Smart<span>Aid</span></div>
        </div>
        <div class="voice-indicator" id="wake-status" onclick="openCalibration()" title="Voice activation status - Click to calibrate">
            <span class="indicator-dot"></span>
            <span id="wake-text">Say "Hey Manso"</span>
        </div>
    </header>

    <!-- Calibration Modal -->
    <div class="modal-overlay" id="calibration-modal">
        <div class="modal">
            <div id="calibration-step-1">
                <div class="modal-title">Voice Calibration</div>
                <div class="modal-subtitle">Train the system to recognize your voice</div>
                <p>Say <strong>"Hey Manso"</strong> three times to help the system learn your voice pattern.</p>
                <div id="saved-phrases-container" style="display:none;">
                    <p style="font-size:12px; color:var(--text-muted);">Currently trained phrases:</p>
                    <div class="saved-list" id="saved-phrases"></div>
                </div>
                <div class="modal-actions">
                    <button class="modal-btn primary" onclick="startCalibration()">Start Calibration</button>
                    <button class="modal-btn secondary" onclick="closeCalibration()">Cancel</button>
                </div>
                <div style="text-align:center; margin-top:16px;">
                    <button class="modal-btn text" onclick="clearCalibration()">Reset to defaults</button>
                </div>
            </div>
            <div id="calibration-step-2" style="display:none;">
                <div class="modal-title">Recording</div>
                <p class="instruction-text">Say <strong>"Hey Manso"</strong> now</p>
                <div class="calibration-progress">
                    <div class="calibration-dot" id="cal-dot-1"></div>
                    <div class="calibration-dot" id="cal-dot-2"></div>
                    <div class="calibration-dot" id="cal-dot-3"></div>
                </div>
                <p id="calibration-instruction" class="instruction-text">Recording 1 of 3</p>
                <div class="heard-display" id="heard-text">Listening...</div>
                <div class="modal-actions">
                    <button class="modal-btn secondary" onclick="cancelCalibration()">Cancel</button>
                </div>
            </div>
            <div id="calibration-step-3" style="display:none;">
                <div class="modal-title">Calibration Complete</div>
                <p>The system has learned these variations of your wake word:</p>
                <div class="saved-list" id="calibration-results"></div>
                <div class="modal-actions">
                    <button class="modal-btn primary" onclick="finishCalibration()">Done</button>
                    <button class="modal-btn secondary" onclick="startCalibration()">Redo</button>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- Safety Panel -->
        <div class="safety-panel" id="safety-box">
            <div class="safety-header">
                <span class="safety-label">Safety Status</span>
            </div>
            <div class="safety-text" id="safety-text">Initializing...</div>
        </div>

        <!-- Video Feed -->
        <div class="video-panel">
            <div class="video-header">
                <span class="video-title">Camera Feed</span>
                <span class="video-status">Live</span>
            </div>
            <img id="video-feed" src="/video_feed" alt="Camera feed">
        </div>

        <!-- Main Actions -->
        <div class="actions-grid">
            <button class="action-btn btn-scan" onclick="quickScan()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="M21 21l-4.35-4.35"/>
                </svg>
                SCAN
            </button>
            <button class="action-btn btn-voice" id="btn-voice" onclick="toggleVoice()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
                <span id="btn-voice-text">SPEAK</span>
            </button>
        </div>

        <!-- Quick Commands -->
        <div class="quick-actions">
            <button class="quick-btn" onclick="askQuestion('Is it safe to walk forward?')">Safe ahead?</button>
            <button class="quick-btn" onclick="askQuestion('Where is the door?')">Find door</button>
            <button class="quick-btn" onclick="askQuestion('What obstacles are near me?')">Obstacles</button>
            <button class="quick-btn" onclick="askQuestion('Describe what you see')">Describe scene</button>
        </div>

        <!-- Input Area -->
        <div class="input-panel">
            <input type="text" id="query-input" placeholder="Ask a question..."
                   onkeypress="if(event.key==='Enter')askQuestion()">
            <button onclick="askQuestion()">Ask</button>
        </div>

        <!-- Response Panel -->
        <div class="response-panel">
            <div class="response-header">
                <span class="response-label">Response</span>
            </div>
            <div class="response-text" id="response-text">Ready. Press SCAN or say "Hey Manso" to begin.</div>
        </div>

        <!-- System Status -->
        <div class="system-status" id="system-status">
            <span class="status-dot"></span>
            <span id="status-text">Checking system...</span>
        </div>
    </div>

    <script>
        // ===== WAKE WORD DETECTION SYSTEM =====
        const WAKE_WORD = "hey manso";
        // Default variants + user calibrated ones
        const DEFAULT_VARIANTS = ["hey manso", "hey monso", "a manso", "hey manzo", "manso", "hey months", "hey months oh"];
        let WAKE_WORD_VARIANTS = [...DEFAULT_VARIANTS];

        let wakeWordRecognition = null;  // Continuous listening for wake word
        let commandRecognition = null;   // One-shot listening for command
        let calibrationRecognition = null; // For calibration
        let isWakeWordListening = false;
        let isCommandListening = false;
        let isCalibrating = false;
        let wakeWordEnabled = true;
        let calibrationCount = 0;
        let calibrationSamples = [];

        // Load saved calibration from localStorage
        function loadCalibration() {
            try {
                const saved = localStorage.getItem('wakeWordVariants');
                if (saved) {
                    const userVariants = JSON.parse(saved);
                    // Combine default + user variants
                    WAKE_WORD_VARIANTS = [...new Set([...DEFAULT_VARIANTS, ...userVariants])];
                    console.log('Loaded calibration:', WAKE_WORD_VARIANTS);
                }
            } catch (e) {
                console.log('Could not load calibration');
            }
        }

        // Save calibration to localStorage
        function saveCalibration(samples) {
            try {
                // Get existing user variants
                let existing = [];
                try {
                    const saved = localStorage.getItem('wakeWordVariants');
                    if (saved) existing = JSON.parse(saved);
                } catch (e) {}

                // Add new samples
                const allUserVariants = [...new Set([...existing, ...samples])];
                localStorage.setItem('wakeWordVariants', JSON.stringify(allUserVariants));

                // Update runtime variants
                WAKE_WORD_VARIANTS = [...new Set([...DEFAULT_VARIANTS, ...allUserVariants])];
                console.log('Saved calibration:', WAKE_WORD_VARIANTS);
            } catch (e) {
                console.log('Could not save calibration');
            }
        }

        function getUserVariants() {
            try {
                const saved = localStorage.getItem('wakeWordVariants');
                return saved ? JSON.parse(saved) : [];
            } catch (e) {
                return [];
            }
        }

        // Initialize wake word recognition (continuous)
        function initWakeWordRecognition() {
            if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
                console.log('Speech recognition not supported');
                updateWakeStatus('Not supported', 'idle');
                return false;
            }

            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

            // Wake word listener (continuous)
            wakeWordRecognition = new SpeechRecognition();
            wakeWordRecognition.continuous = true;
            wakeWordRecognition.interimResults = true;
            wakeWordRecognition.lang = 'en-US';

            wakeWordRecognition.onresult = function(event) {
                // Check all results for wake word
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript.toLowerCase().trim();
                    console.log('Heard:', transcript);

                    // Check if any wake word variant is detected
                    for (const variant of WAKE_WORD_VARIANTS) {
                        if (transcript.includes(variant)) {
                            console.log('Wake word detected!');
                            onWakeWordDetected();
                            return;
                        }
                    }
                }
            };

            wakeWordRecognition.onerror = function(event) {
                console.log('Wake word recognition error:', event.error);
                if (event.error === 'no-speech' || event.error === 'aborted') {
                    // Restart listening after brief pause
                    if (wakeWordEnabled && !isCommandListening) {
                        setTimeout(startWakeWordListening, 500);
                    }
                }
            };

            wakeWordRecognition.onend = function() {
                isWakeWordListening = false;
                // Auto-restart if enabled and not listening for command
                if (wakeWordEnabled && !isCommandListening) {
                    setTimeout(startWakeWordListening, 500);
                }
            };

            // Command listener (one-shot)
            commandRecognition = new SpeechRecognition();
            commandRecognition.continuous = false;
            commandRecognition.interimResults = false;
            commandRecognition.lang = 'en-US';

            commandRecognition.onresult = function(event) {
                const command = event.results[0][0].transcript;
                console.log('Command received:', command);
                onCommandReceived(command);
            };

            commandRecognition.onerror = function(event) {
                console.log('Command recognition error:', event.error);
                speak("Sorry, I didn't catch that. Say Hey Manso to try again.");
                returnToWakeWordListening();
            };

            commandRecognition.onend = function() {
                isCommandListening = false;
                // If no command was processed, return to wake word listening
                setTimeout(returnToWakeWordListening, 1000);
            };

            return true;
        }

        function startWakeWordListening() {
            if (isWakeWordListening || isCommandListening || !wakeWordRecognition) return;

            try {
                wakeWordRecognition.start();
                isWakeWordListening = true;
                updateWakeStatus('Say "Hey Manso"', 'listening');
                console.log('Wake word listening started');
            } catch (e) {
                console.log('Could not start wake word listening:', e);
                // Try again after delay
                setTimeout(startWakeWordListening, 1000);
            }
        }

        function stopWakeWordListening() {
            if (wakeWordRecognition && isWakeWordListening) {
                wakeWordRecognition.stop();
                isWakeWordListening = false;
            }
        }

        function onWakeWordDetected() {
            // Stop wake word listening
            stopWakeWordListening();

            // Audio feedback
            playBeep();
            speak("Yes?");

            // Update UI
            updateWakeStatus('Listening...', 'active');
            document.getElementById('btn-voice').classList.add('listening');
            document.getElementById('btn-voice-text').textContent = 'LISTENING...';

            // Start command recognition after TTS finishes
            setTimeout(startCommandListening, 800);
        }

        function startCommandListening() {
            if (isCommandListening || !commandRecognition) return;

            try {
                commandRecognition.start();
                isCommandListening = true;
                console.log('Command listening started');
            } catch (e) {
                console.log('Could not start command listening:', e);
                returnToWakeWordListening();
            }
        }

        function onCommandReceived(command) {
            isCommandListening = false;

            // Update UI
            document.getElementById('query-input').value = command;
            document.getElementById('btn-voice').classList.remove('listening');
            document.getElementById('btn-voice-text').textContent = 'SPEAK';
            updateWakeStatus('Processing...', 'idle');

            // Process the command
            askQuestion(command);

            // Return to wake word listening after response
            setTimeout(returnToWakeWordListening, 3000);
        }

        function returnToWakeWordListening() {
            if (!isCommandListening && wakeWordEnabled) {
                document.getElementById('btn-voice').classList.remove('listening');
                document.getElementById('btn-voice-text').textContent = 'SPEAK';
                startWakeWordListening();
            }
        }

        function updateWakeStatus(text, state) {
            const el = document.getElementById('wake-status');
            document.getElementById('wake-text').textContent = text;
            el.className = 'voice-indicator ' + state;
        }

        function playBeep() {
            // Simple beep using Web Audio API
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = ctx.createOscillator();
                const gain = ctx.createGain();
                oscillator.connect(gain);
                gain.connect(ctx.destination);
                oscillator.frequency.value = 800;
                oscillator.type = 'sine';
                gain.gain.value = 0.3;
                oscillator.start();
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.2);
                oscillator.stop(ctx.currentTime + 0.2);
            } catch (e) {
                console.log('Could not play beep');
            }
        }

        // ===== MANUAL VOICE BUTTON =====
        function toggleVoice() {
            if (!commandRecognition) {
                speak('Voice not supported. Please type your question.');
                return;
            }

            if (isCommandListening) {
                commandRecognition.stop();
                returnToWakeWordListening();
            } else {
                // Stop wake word listening and start command
                stopWakeWordListening();
                playBeep();
                updateWakeStatus('Listening...', 'active');
                document.getElementById('btn-voice').classList.add('listening');
                document.getElementById('btn-voice-text').textContent = 'LISTENING...';
                setTimeout(startCommandListening, 300);
            }
        }

        // ===== TOGGLE WAKE WORD =====
        function toggleWakeWord() {
            wakeWordEnabled = !wakeWordEnabled;
            if (wakeWordEnabled) {
                startWakeWordListening();
                speak('Wake word enabled. Say Hey Manso to activate.');
            } else {
                stopWakeWordListening();
                updateWakeStatus('Wake word off', 'idle');
                speak('Wake word disabled.');
            }
        }

        // ===== CALIBRATION FUNCTIONS =====
        function openCalibration() {
            // Stop wake word listening during calibration
            stopWakeWordListening();
            wakeWordEnabled = false;

            // Show saved phrases if any
            const userVariants = getUserVariants();
            if (userVariants.length > 0) {
                document.getElementById('saved-phrases-container').style.display = 'block';
                document.getElementById('saved-phrases').innerHTML = userVariants.map(v => `<div>"${v}"</div>`).join('');
            } else {
                document.getElementById('saved-phrases-container').style.display = 'none';
            }

            // Show modal
            document.getElementById('calibration-step-1').style.display = 'block';
            document.getElementById('calibration-step-2').style.display = 'none';
            document.getElementById('calibration-step-3').style.display = 'none';
            document.getElementById('calibration-modal').classList.add('show');
        }

        function closeCalibration() {
            document.getElementById('calibration-modal').classList.remove('show');
            wakeWordEnabled = true;
            startWakeWordListening();
        }

        function startCalibration() {
            calibrationCount = 0;
            calibrationSamples = [];

            // Reset dots
            for (let i = 1; i <= 3; i++) {
                document.getElementById(`cal-dot-${i}`).className = 'calibration-dot';
            }

            // Show calibration step
            document.getElementById('calibration-step-1').style.display = 'none';
            document.getElementById('calibration-step-2').style.display = 'block';
            document.getElementById('calibration-step-3').style.display = 'none';

            // Start listening
            isCalibrating = true;
            startCalibrationListening();
        }

        function startCalibrationListening() {
            if (!calibrationRecognition) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                calibrationRecognition = new SpeechRecognition();
                calibrationRecognition.continuous = false;
                calibrationRecognition.interimResults = false;
                calibrationRecognition.lang = 'en-US';

                calibrationRecognition.onresult = function(event) {
                    const transcript = event.results[0][0].transcript.toLowerCase().trim();
                    onCalibrationResult(transcript);
                };

                calibrationRecognition.onerror = function(event) {
                    console.log('Calibration error:', event.error);
                    if (isCalibrating && event.error !== 'aborted') {
                        document.getElementById('heard-text').textContent = 'Did not hear. Try again...';
                        setTimeout(startCalibrationListening, 1000);
                    }
                };

                calibrationRecognition.onend = function() {
                    // If still calibrating and not enough samples, restart
                    if (isCalibrating && calibrationCount < 3) {
                        setTimeout(startCalibrationListening, 500);
                    }
                };
            }

            updateWakeStatus('Calibrating...', 'calibrating');
            document.getElementById('cal-dot-' + (calibrationCount + 1)).className = 'calibration-dot current';
            document.getElementById('calibration-instruction').textContent = `Recording ${calibrationCount + 1} of 3...`;
            document.getElementById('heard-text').textContent = 'Listening...';

            playBeep();

            try {
                calibrationRecognition.start();
            } catch (e) {
                console.log('Could not start calibration:', e);
            }
        }

        function onCalibrationResult(transcript) {
            console.log('Calibration heard:', transcript);
            document.getElementById('heard-text').textContent = `Heard: "${transcript}"`;

            // Save this sample
            calibrationSamples.push(transcript);
            calibrationCount++;

            // Update dot
            document.getElementById(`cal-dot-${calibrationCount}`).className = 'calibration-dot done';

            if (calibrationCount < 3) {
                // Continue calibration
                document.getElementById('calibration-instruction').textContent = `Good! Recording ${calibrationCount + 1} of 3...`;
                playBeep();
            } else {
                // Calibration complete
                finishCalibrationRecording();
            }
        }

        function finishCalibrationRecording() {
            isCalibrating = false;
            if (calibrationRecognition) {
                try { calibrationRecognition.stop(); } catch(e) {}
            }

            // Save the samples
            saveCalibration(calibrationSamples);

            // Show results
            document.getElementById('calibration-step-2').style.display = 'none';
            document.getElementById('calibration-step-3').style.display = 'block';
            document.getElementById('calibration-results').innerHTML =
                calibrationSamples.map(s => `<div>"${s}"</div>`).join('');

            speak('Calibration complete. You can now say Hey Manso to activate.');
        }

        function cancelCalibration() {
            isCalibrating = false;
            if (calibrationRecognition) {
                try { calibrationRecognition.stop(); } catch(e) {}
            }
            closeCalibration();
        }

        function finishCalibration() {
            closeCalibration();
            speak('Ready. Say Hey Manso anytime.');
        }

        function clearCalibration() {
            localStorage.removeItem('wakeWordVariants');
            WAKE_WORD_VARIANTS = [...DEFAULT_VARIANTS];
            speak('Calibration cleared. Using default wake word.');
            closeCalibration();
        }

        // ===== CORE FUNCTIONS =====
        function speak(text) {
            if (!('speechSynthesis' in window)) {
                console.warn('Speech synthesis not supported');
                return;
            }

            // Cancel any ongoing speech
            speechSynthesis.cancel();

            // Wait for cancellation to complete before speaking new text
            // This fixes the race condition where cancel() is async
            setTimeout(() => {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.1;
                utterance.pitch = 1.0;

                // Error handling for debugging
                utterance.onerror = (event) => {
                    console.error('Speech synthesis error:', event.error);
                };

                speechSynthesis.speak(utterance);
            }, 100);
        }

        function updateSafety(text, level) {
            const box = document.getElementById('safety-box');
            const textEl = document.getElementById('safety-text');
            textEl.textContent = text;
            box.className = 'safety-panel ' + level;
            speak(text);
        }

        function setResponse(text) {
            const el = document.getElementById('response-text');
            el.textContent = text;
            el.classList.remove('loading');
            speak(text);
        }

        async function quickScan() {
            document.getElementById('safety-text').textContent = 'Scanning...';
            try {
                const response = await fetch('/api/scan');
                const data = await response.json();
                const text = data.response;

                let level = 'safe';
                if (text.includes('STOP') || text.includes('!')) level = 'danger';
                else if (text.includes('Caution') || text.includes('Warning')) level = 'warning';

                updateSafety(text, level);
            } catch (error) {
                updateSafety('Scan failed', 'danger');
            }
        }

        async function askQuestion(question) {
            const query = question || document.getElementById('query-input').value.trim();
            if (!query) return;

            const responseEl = document.getElementById('response-text');
            responseEl.textContent = 'Processing...';
            responseEl.classList.add('loading');

            try {
                const response = await fetch('/api/navigate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query})
                });
                const data = await response.json();
                setResponse(data.response);
            } catch (error) {
                setResponse('Error: Could not process request.');
            }
        }

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
                document.getElementById('status-text').textContent = 'Connection error';
                document.getElementById('system-status').className = 'system-status error';
            }
        }

        // ===== INITIALIZATION =====
        document.addEventListener('DOMContentLoaded', function() {
            checkStatus();
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

                    startWakeWordListening();
                    if (userVariants.length > 0) {
                        speak('Ready. Say Hey Manso anytime.');
                    } else {
                        speak('Ready. Click the microphone icon to calibrate your voice, or say Hey Manso.');
                    }
                    document.body.removeEventListener('click', initAudio);
                }, { once: false });

                if (userVariants.length > 0) {
                    updateWakeStatus('Click to start', 'idle');
                } else {
                    updateWakeStatus('Click to calibrate', 'idle');
                }
            }
        });
    </script>
</body>
</html>
"""


def generate_frames():
    """Generate video frames for streaming."""
    global current_frame
    while True:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue
        with frame_lock:
            current_frame = frame.copy()
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/scan')
def api_scan():
    """Quick safety scan."""
    global current_frame, navigator
    if current_frame is None:
        return jsonify({'response': 'No camera.'})
    if navigator is None:
        return jsonify({'response': 'System loading...'})
    with frame_lock:
        frame = current_frame.copy()
    response = navigator.quick_scan(frame)
    return jsonify({'response': response})


@app.route('/api/navigate', methods=['POST'])
def api_navigate():
    """Process navigation query."""
    global current_frame, navigator
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'response': 'Please ask a question.'})
    if current_frame is None:
        return jsonify({'response': 'No camera feed.'})
    if navigator is None:
        return jsonify({'response': 'System loading...'})
    with frame_lock:
        frame = current_frame.copy()
    response = navigator.navigate(query, frame)
    return jsonify({'response': response})


@app.route('/api/status')
def api_status():
    """Check system status."""
    return jsonify({
        'ready': navigator is not None and navigator._loaded,
        'model': 'Florence-2 + YOLO + Depth',
    })


def init_navigator():
    """Initialize the smart navigator."""
    global navigator
    config = Config()
    config.pipeline.use_florence = True
    config.pipeline.use_yolo_world = True
    config.pipeline.use_depth = True
    navigator = SmartNavigator(config)
    print("Loading models...")
    navigator.load()
    print("Navigator ready!")


def init_camera(source):
    """Initialize camera."""
    global camera
    print(f"Opening camera: {source}")
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        print(f"Failed to open camera: {source}")
        return False
    print("Camera ready!")
    return True


def run_server(source=0, host="0.0.0.0", port=8080):
    """Run the navigation server."""
    print("=" * 50)
    print("Smart Aid Navigator")
    print("=" * 50)

    init_navigator()
    init_camera(source)

    print(f"\n→ Open http://localhost:{port} in your browser")
    print("=" * 50)

    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smart Aid Navigator")
    parser.add_argument("--source", default="0", help="Video source")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_server(source=source, port=args.port)
