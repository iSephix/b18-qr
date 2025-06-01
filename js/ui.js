// js/ui.js
/** @fileoverview Handles UI interactions, event listeners, and DOM updates for the Custom Visual Code tool. */

// Ensure this code runs after main.js has defined window.mainJsDOMContentLoadedCallbacks
if (window.mainJsDOMContentLoadedCallbacks) {
    window.mainJsDOMContentLoadedCallbacks.push(setupUIEventListeners);
} else {
    // Fallback if main.js didn't load or define the queue first (less ideal, but robust)
    console.warn("ui.js: mainJsDOMContentLoadedCallbacks not found at script parse time. Using direct DOMContentLoaded listener as fallback.");
    document.addEventListener('DOMContentLoaded', setupUIEventListeners);
}

/**
 * Sets up all UI event listeners for the application.
 * This function is intended to be called once the DOM is fully loaded.
 */
function setupUIEventListeners() {
    console.info("ui.js: Setting up UI event listeners.");
    const decodeFileInput = document.getElementById('decode-file-input');
    const uploadedImagePreview = document.getElementById('uploaded-image-preview');
    const decodeButton = document.getElementById('decode-button');
    const encodeButton = document.getElementById('encode-button');
    const encodeDataInput = document.getElementById('encode-data-input');
    const qrCanvas = document.getElementById('qrCanvas');
    const downloadQrLink = document.getElementById('download-qr-link');

    if (decodeFileInput && uploadedImagePreview && decodeButton) {
        decodeFileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImagePreview.src = e.target.result;
                    uploadedImagePreview.style.display = 'block';
                    decodeButton.disabled = false;
                    updateDecodeResult('Image loaded. Click "Decode Visual Code" to process.');
                }
                reader.readAsDataURL(file);
            } else {
                uploadedImagePreview.src = ''; // Clear src
                uploadedImagePreview.style.display = 'none';
                decodeButton.disabled = true;
                updateDecodeResult('No image selected or file cleared.');
            }
        });

        decodeButton.addEventListener('click', async function() {
            if (!uploadedImagePreview.src || uploadedImagePreview.src === '#' || uploadedImagePreview.style.display === 'none') {
                updateDecodeResult('Please select an image file first.');
                return;
            }

            // OpenCV readiness check removed. JSFeat is assumed to be loaded via script tag.
            // Math.js is also assumed to be loaded.

            if (typeof decodeVisualCodeFromImage === 'function') { // This function is from image_processing.js
                decodeButton.disabled = true;
                showSpinner('decode-spinner'); // UI function from this file

                const rawDecodedString = await decodeVisualCodeFromImage(uploadedImagePreview);

                hideSpinner('decode-spinner'); // UI function from this file
                decodeButton.disabled = false;

                if (rawDecodedString !== null && rawDecodedString !== undefined) {
                    const finalProcessedString = processDecodedStringWithMathJS(rawDecodedString); // UI function from this file
                    updateDecodeResult(`<strong>Successfully Decoded:</strong><br><pre>${escapeHtml(finalProcessedString)}</pre>`);
                } else {
                    // If rawDecodedString is null, image_processing.js should have set an error message.
                    // No need to override it here unless we want a generic fallback.
                    // console.info("ui.js: decodeVisualCodeFromImage returned null.");
                }
            } else {
                console.error('decodeVisualCodeFromImage function is not defined. Ensure image_processing.js is loaded.');
                updateDecodeResult('Error: Core decoding function (decodeVisualCodeFromImage) is missing.');
            }
        });
    } else {
        console.warn("ui.js: One or more decoding-related UI elements (decode-file-input, uploaded-image-preview, decode-button) were not found.");
    }

    if (encodeButton && encodeDataInput && qrCanvas && downloadQrLink) {
        encodeButton.addEventListener('click', async function() {
            const textToEncode = encodeDataInput.value;
            if (!textToEncode.trim()) { alert("Please enter text to encode."); return; }
            showSpinner('encode-spinner');
            encodeButton.disabled = true;
            try {
                if (typeof generateFullEncodedQrOnCanvas === 'function') { // This function is from custom_codec.js
                    await generateFullEncodedQrOnCanvas(textToEncode, qrCanvas, downloadQrLink);
                } else {
                    console.error("generateFullEncodedQrOnCanvas function is not defined. Ensure custom_codec.js is loaded.");
                    alert("Encoding function (generateFullEncodedQrOnCanvas) is missing.");
                }
            } catch (e) {
                console.error("Error during JS encoding UI interaction:", e.stack || e);
                alert("Error during encoding: " + e.message);
            } finally {
                hideSpinner('encode-spinner');
                encodeButton.disabled = false;
            }
        });
    } else {
        console.warn("ui.js: One or more encoding-related UI elements (encode-button, etc.) were not found.");
    }

    const clearDebugLogButton = document.getElementById('clear-debug-log-button');
    if (clearDebugLogButton) {
        clearDebugLogButton.addEventListener('click', function() {
            const debugLogDiv = document.getElementById('mobile-debug-log');
            if (debugLogDiv) {
                debugLogDiv.innerHTML = '';
                if(typeof logToScreen === "function") logToScreen("Debug log cleared by user."); // Log this action itself
                else console.log("Debug log cleared by user.");
            }
        });
    } else {
        console.warn("ui.js: Clear debug log button ('clear-debug-log-button') not found.");
    }

} // end setupUIEventListeners


/**
 * Logs a message to both the console and a dedicated on-screen debug div.
 * Prepends a timestamp to the message.
 * @param {String} message - The message to log.
 */
function logToScreen(message) {
    console.log(message); // Keep console logging

    const debugLogDiv = document.getElementById('mobile-debug-log');
    if (debugLogDiv) {
        const timestamp = new Date().toLocaleTimeString();
        const newLogEntry = document.createElement('div');
        newLogEntry.textContent = `[${timestamp}] ${message}`;
        debugLogDiv.appendChild(newLogEntry);
        // Scroll to the bottom
        debugLogDiv.scrollTop = debugLogDiv.scrollHeight;
    }
}
window.logToScreen = logToScreen; // Make it globally accessible


/**
 * Updates the content of the decoding result display area.
 * @param {String} message - The HTML string or plain text to display.
 */
function updateDecodeResult(message) {
    const resultDiv = document.getElementById('decode-result');
    if (resultDiv) resultDiv.innerHTML = message;
}
/** Shows a spinner element. @param {String} spinnerId - ID of the spinner element. */
function showSpinner(spinnerId) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) spinner.style.display = 'block';
}
/** Hides a spinner element. @param {String} spinnerId - ID of the spinner element. */
function hideSpinner(spinnerId) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) spinner.style.display = 'none';
}
/**
 * Escapes HTML special characters in a string to prevent XSS.
 * @param {String} unsafe - The potentially unsafe string.
 * @returns {String} The escaped string.
 */
function escapeHtml(unsafe) {
    if (unsafe === null || unsafe === undefined) return "";
    return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

/**
 * Processes a decoded string to evaluate embedded Math.js expressions for linear and power equations.
 * Appends the solution or an error message next to the respective tags.
 * @param {String} decodedString - The string potentially containing <linear>equation</linear> or <power>equation</power> tags.
 * @returns {String} The processed string with equation solutions appended.
 */
function processDecodedStringWithMathJS(decodedString) {
    if (typeof math === 'undefined' || !decodedString || typeof decodedString !== 'string') {
        console.warn("Math.js not available or invalid input for processDecodedStringWithMathJS.");
        return decodedString;
    }
    const linearRegex = /<linear>(.*?)<\/linear>/g;
    const powerRegex = /<power>(.*?)<\/power>/g;
    let resultString = decodedString;
    try {
        resultString = resultString.replace(linearRegex, (match, eqContent) => `${match} ${solveLinearEquationWithMathJS(eqContent)}`);
        resultString = resultString.replace(powerRegex, (match, eqContent) => `${match} ${solvePowerEquationWithMathJS(eqContent)}`);
    } catch (e) {
        console.error("Error during MathJS tag processing:", e.stack || e);
        // Potentially append a general error message to resultString or handle as needed
    }
    return resultString;
}

/** Solves a linear equation string using Math.js. @param {String} eqString - e.g., "2x+3=7". @returns {String} Solution or error message. */
function solveLinearEquationWithMathJS(eqString) {
    try { const parts = eqString.split('='); if (parts.length !== 2) return "[Lin Error: Format]";
        const exprNode = math.parse(`${parts[0].trim()} - (${parts[1].trim()})`); const simplifiedExpr = math.simplify(exprNode);
        let b_val, a_plus_b_val;
        try { b_val = simplifiedExpr.evaluate({x: 0}); a_plus_b_val = simplifiedExpr.evaluate({x: 1}); }
        catch (e) { if (simplifiedExpr.isConstantNode) return Math.abs(simplifiedExpr.value) < 1e-9 ? "[Lin: Infinite sols]" : "[Lin: No sol]"; return "[Lin Error: Eval form ("+e.message.substring(0,20)+"...)]"; }
        if (typeof b_val !== 'number' || typeof a_plus_b_val !== 'number' ) return "[Lin Error: Coeffs NaN]";
        const a_val = a_plus_b_val - b_val;
        if (Math.abs(a_val) < 1e-9) return Math.abs(b_val) < 1e-9 ? "[Lin: Infinite sols (0=0)]" : "[Lin: No sol (const!=0)]";
        return `[Calc: x = ${math.format(-b_val / a_val, {precision: 14})}]`;
    } catch (err) { return "[Lin Error: " + err.message.substring(0,30) + "...]"; }
}

/** Solves a power equation string using Math.js. @param {String} eqString - e.g., "x^2=9". @returns {String} Solution or error message. */
function solvePowerEquationWithMathJS(eqString) {
    try { const parts = eqString.split('='); if (parts.length !== 2) return "[Pow Error: Format]";
        const exprNode = math.parse(`${parts[0].trim()} - (${parts[1].trim()})`); const simplified = math.simplify(exprNode);
        let a = 1, n_val, C_val;
        if (simplified.isOperatorNode && simplified.op === '-' && simplified.args.length === 2) {
            C_val = simplified.args[1].evaluate({});
            let term_with_x = simplified.args[0];
            if (term_with_x.isOperatorNode && term_with_x.op === '^' && term_with_x.args[0].isSymbolNode && term_with_x.args[0].name === 'x' && term_with_x.args[1].isConstantNode) {
                n_val = term_with_x.args[1].value;
            } else if (term_with_x.isOperatorNode && term_with_x.op === '*' && term_with_x.args[0].isConstantNode && term_with_x.args[1].isOperatorNode && term_with_x.args[1].op === '^' && term_with_x.args[1].args[0].isSymbolNode && term_with_x.args[1].args[0].name === 'x' && term_with_x.args[1].args[1].isConstantNode) {
                a = term_with_x.args[0].value; n_val = term_with_x.args[1].args[1].value;
            } else { return "[Pow: Eq form not recognized (ax^n - C expected)]"; }
        } else { return "[Pow: Eq form not recognized (expr - C expected)]"; }
        if (a === 0) return "[Pow Error: Coeff 'a' is zero]"; const valForPower = C_val / a;
        if (valForPower < 0 && n_val % 2 === 0) return `[Pow: No real sol for ${a}*x^${n_val}=${C_val}]`;
        const principalRoot = math.evaluate(`${valForPower}^(1/${n_val})`); let solutions = [principalRoot];
        if (n_val % 2 === 0 && Math.abs(valForPower) > 1e-9 && Math.abs(principalRoot) > 1e-9) { solutions.push(-principalRoot); }
        return `[Calc: x = ${solutions.map(s => math.format(s, {precision: 14})).join(' or ')}]`;
    } catch (err) { return "[Pow Error: " + err.message.substring(0,30) + "...]"; }
}

// JSDoc stub (actual JSDoc comments are above function definitions)
// /** @fileoverview Handles UI interactions, event listeners, and DOM updates. */
