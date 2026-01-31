/**
 * Voice handling for Smart Aid Navigator
 * - Wake word detection ("Hey Manso")
 * - Voice calibration
 * - Text-to-speech (TTS)
 */

// ===== CONFIGURATION =====
const WAKE_WORD = "hey manso";
const DEFAULT_VARIANTS = [
    "hey manso", "hey monso", "a manso", "hey manzo",
    "manso", "hey months", "hey months oh"
];
let WAKE_WORD_VARIANTS = [...DEFAULT_VARIANTS];

// ===== STATE =====
let wakeWordRecognition = null;
let commandRecognition = null;
let calibrationRecognition = null;
let isWakeWordListening = false;
let isCommandListening = false;
let isCalibrating = false;
let wakeWordEnabled = true;
let calibrationCount = 0;
let calibrationSamples = [];

// ===== TEXT-TO-SPEECH (FIXED) =====

// Track if TTS has been initialized by user interaction
let ttsInitialized = false;

/**
 * Initialize TTS - must be called from user interaction (click/tap).
 */
function initTTS() {
    if (ttsInitialized) return;

    // Trigger voice loading
    speechSynthesis.getVoices();

    // Speak empty string to "wake up" the synthesis
    const warmup = new SpeechSynthesisUtterance('');
    speechSynthesis.speak(warmup);
    speechSynthesis.cancel();

    ttsInitialized = true;
    console.log('TTS initialized');
}

/**
 * Speak text using Web Speech API.
 * Fixed: Handle Chrome/Safari speechSynthesis bugs.
 */
function speak(text) {
    if (!('speechSynthesis' in window)) {
        console.warn('Speech synthesis not supported');
        return;
    }

    if (!text || text.trim() === '') {
        return;
    }

    console.log('Speaking:', text.substring(0, 50) + '...');

    // Cancel any ongoing speech
    speechSynthesis.cancel();

    // Chrome bug: speechSynthesis can get stuck, need to resume
    if (speechSynthesis.paused) {
        speechSynthesis.resume();
    }

    // Wait for cancellation to complete before speaking new text
    setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        // Try to get a good voice
        const voices = speechSynthesis.getVoices();
        const englishVoice = voices.find(v => v.lang.startsWith('en') && v.localService);
        if (englishVoice) {
            utterance.voice = englishVoice;
        }

        // Event handlers for debugging
        utterance.onstart = () => {
            console.log('Speech started');
        };

        utterance.onend = () => {
            console.log('Speech ended');
        };

        utterance.onerror = (event) => {
            console.error('Speech synthesis error:', event.error);
            // Try to recover
            speechSynthesis.cancel();
        };

        // Speak!
        speechSynthesis.speak(utterance);

        // Chrome bug workaround: keep synthesis alive
        // Without this, long texts may stop mid-sentence
        const keepAlive = setInterval(() => {
            if (!speechSynthesis.speaking) {
                clearInterval(keepAlive);
            } else {
                speechSynthesis.pause();
                speechSynthesis.resume();
            }
        }, 10000);

    }, 150);
}

// ===== AUDIO FEEDBACK =====

/**
 * Play a short beep sound for audio feedback.
 */
function playBeep() {
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

// ===== CALIBRATION STORAGE =====

/**
 * Load saved calibration from localStorage.
 */
function loadCalibration() {
    try {
        const saved = localStorage.getItem('wakeWordVariants');
        if (saved) {
            const userVariants = JSON.parse(saved);
            WAKE_WORD_VARIANTS = [...new Set([...DEFAULT_VARIANTS, ...userVariants])];
            console.log('Loaded calibration:', WAKE_WORD_VARIANTS);
        }
    } catch (e) {
        console.log('Could not load calibration');
    }
}

/**
 * Save calibration samples to localStorage.
 */
function saveCalibration(samples) {
    try {
        let existing = [];
        try {
            const saved = localStorage.getItem('wakeWordVariants');
            if (saved) existing = JSON.parse(saved);
        } catch (e) {}

        const allUserVariants = [...new Set([...existing, ...samples])];
        localStorage.setItem('wakeWordVariants', JSON.stringify(allUserVariants));
        WAKE_WORD_VARIANTS = [...new Set([...DEFAULT_VARIANTS, ...allUserVariants])];
        console.log('Saved calibration:', WAKE_WORD_VARIANTS);
    } catch (e) {
        console.log('Could not save calibration');
    }
}

/**
 * Get user-trained variants from localStorage.
 */
function getUserVariants() {
    try {
        const saved = localStorage.getItem('wakeWordVariants');
        return saved ? JSON.parse(saved) : [];
    } catch (e) {
        return [];
    }
}

// ===== WAKE WORD RECOGNITION =====

/**
 * Initialize speech recognition for wake word and commands.
 */
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
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript.toLowerCase().trim();
            console.log('Heard:', transcript);

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
            if (wakeWordEnabled && !isCommandListening) {
                setTimeout(startWakeWordListening, 500);
            }
        }
    };

    wakeWordRecognition.onend = function() {
        isWakeWordListening = false;
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
        setTimeout(returnToWakeWordListening, 1000);
    };

    return true;
}

/**
 * Start listening for wake word.
 */
function startWakeWordListening() {
    if (isWakeWordListening || isCommandListening || !wakeWordRecognition) return;

    try {
        wakeWordRecognition.start();
        isWakeWordListening = true;
        updateWakeStatus('Say "Hey Manso"', 'listening');
        console.log('Wake word listening started');
    } catch (e) {
        console.log('Could not start wake word listening:', e);
        setTimeout(startWakeWordListening, 1000);
    }
}

/**
 * Stop listening for wake word.
 */
function stopWakeWordListening() {
    if (wakeWordRecognition && isWakeWordListening) {
        wakeWordRecognition.stop();
        isWakeWordListening = false;
    }
}

/**
 * Handle wake word detection.
 */
function onWakeWordDetected() {
    stopWakeWordListening();
    playBeep();
    speak("Yes?");

    updateWakeStatus('Listening...', 'active');
    document.getElementById('btn-voice').classList.add('listening');
    document.getElementById('btn-voice-text').textContent = 'LISTENING...';

    setTimeout(startCommandListening, 800);
}

/**
 * Start listening for command after wake word.
 */
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

/**
 * Handle received command.
 */
function onCommandReceived(command) {
    isCommandListening = false;

    document.getElementById('query-input').value = command;
    document.getElementById('btn-voice').classList.remove('listening');
    document.getElementById('btn-voice-text').textContent = 'SPEAK';
    updateWakeStatus('Processing...', 'idle');

    // Process the command (defined in app.js)
    askQuestion(command);

    setTimeout(returnToWakeWordListening, 3000);
}

/**
 * Return to wake word listening after command.
 */
function returnToWakeWordListening() {
    if (!isCommandListening && wakeWordEnabled) {
        document.getElementById('btn-voice').classList.remove('listening');
        document.getElementById('btn-voice-text').textContent = 'SPEAK';
        startWakeWordListening();
    }
}

/**
 * Update wake word status indicator.
 */
function updateWakeStatus(text, state) {
    const el = document.getElementById('wake-status');
    document.getElementById('wake-text').textContent = text;
    el.className = 'voice-indicator ' + state;
}

// ===== MANUAL VOICE BUTTON =====

/**
 * Toggle voice input manually (button click).
 */
function toggleVoice() {
    if (!commandRecognition) {
        speak('Voice not supported. Please type your question.');
        return;
    }

    if (isCommandListening) {
        commandRecognition.stop();
        returnToWakeWordListening();
    } else {
        stopWakeWordListening();
        playBeep();
        updateWakeStatus('Listening...', 'active');
        document.getElementById('btn-voice').classList.add('listening');
        document.getElementById('btn-voice-text').textContent = 'LISTENING...';
        setTimeout(startCommandListening, 300);
    }
}

// ===== CALIBRATION =====

/**
 * Open calibration modal.
 */
function openCalibration() {
    stopWakeWordListening();
    wakeWordEnabled = false;

    const userVariants = getUserVariants();
    if (userVariants.length > 0) {
        document.getElementById('saved-phrases-container').style.display = 'block';
        document.getElementById('saved-phrases').innerHTML =
            userVariants.map(v => `<div>"${v}"</div>`).join('');
    } else {
        document.getElementById('saved-phrases-container').style.display = 'none';
    }

    document.getElementById('calibration-step-1').style.display = 'block';
    document.getElementById('calibration-step-2').style.display = 'none';
    document.getElementById('calibration-step-3').style.display = 'none';
    document.getElementById('calibration-modal').classList.add('show');
}

/**
 * Close calibration modal.
 */
function closeCalibration() {
    document.getElementById('calibration-modal').classList.remove('show');
    wakeWordEnabled = true;
    startWakeWordListening();
}

/**
 * Start calibration process.
 */
function startCalibration() {
    calibrationCount = 0;
    calibrationSamples = [];

    for (let i = 1; i <= 3; i++) {
        document.getElementById(`cal-dot-${i}`).className = 'calibration-dot';
    }

    document.getElementById('calibration-step-1').style.display = 'none';
    document.getElementById('calibration-step-2').style.display = 'block';
    document.getElementById('calibration-step-3').style.display = 'none';

    isCalibrating = true;
    startCalibrationListening();
}

/**
 * Start listening for calibration sample.
 */
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
            if (isCalibrating && calibrationCount < 3) {
                setTimeout(startCalibrationListening, 500);
            }
        };
    }

    updateWakeStatus('Calibrating...', 'calibrating');
    document.getElementById(`cal-dot-${calibrationCount + 1}`).className = 'calibration-dot current';
    document.getElementById('calibration-instruction').textContent = `Recording ${calibrationCount + 1} of 3...`;
    document.getElementById('heard-text').textContent = 'Listening...';

    playBeep();

    try {
        calibrationRecognition.start();
    } catch (e) {
        console.log('Could not start calibration:', e);
    }
}

/**
 * Handle calibration result.
 */
function onCalibrationResult(transcript) {
    console.log('Calibration heard:', transcript);
    document.getElementById('heard-text').textContent = `Heard: "${transcript}"`;

    calibrationSamples.push(transcript);
    calibrationCount++;

    document.getElementById(`cal-dot-${calibrationCount}`).className = 'calibration-dot done';

    if (calibrationCount < 3) {
        document.getElementById('calibration-instruction').textContent = `Good! Recording ${calibrationCount + 1} of 3...`;
        playBeep();
    } else {
        finishCalibrationRecording();
    }
}

/**
 * Finish calibration recording.
 */
function finishCalibrationRecording() {
    isCalibrating = false;
    if (calibrationRecognition) {
        try { calibrationRecognition.stop(); } catch(e) {}
    }

    saveCalibration(calibrationSamples);

    document.getElementById('calibration-step-2').style.display = 'none';
    document.getElementById('calibration-step-3').style.display = 'block';
    document.getElementById('calibration-results').innerHTML =
        calibrationSamples.map(s => `<div>"${s}"</div>`).join('');

    speak('Calibration complete. You can now say Hey Manso to activate.');
}

/**
 * Cancel calibration.
 */
function cancelCalibration() {
    isCalibrating = false;
    if (calibrationRecognition) {
        try { calibrationRecognition.stop(); } catch(e) {}
    }
    closeCalibration();
}

/**
 * Finish calibration and close modal.
 */
function finishCalibration() {
    closeCalibration();
    speak('Ready. Say Hey Manso anytime.');
}

/**
 * Clear all calibration data.
 */
function clearCalibration() {
    localStorage.removeItem('wakeWordVariants');
    WAKE_WORD_VARIANTS = [...DEFAULT_VARIANTS];
    speak('Calibration cleared. Using default wake word.');
    closeCalibration();
}
