// js/ui.js

document.addEventListener('DOMContentLoaded', () => {
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
                uploadedImagePreview.style.display = 'none';
                decodeButton.disabled = true;
            }
        });

        decodeButton.addEventListener('click', async function() {
            if (!uploadedImagePreview.src || uploadedImagePreview.src === '#' || uploadedImagePreview.style.display === 'none') {
                updateDecodeResult('Please select an image file first.');
                return;
            }
            if (typeof decodeVisualCodeFromImage === 'function') { // from image_processing.js
                if (window.cv && window.cv.imread) { // OpenCV check
                    decodeButton.disabled = true;
                    showSpinner('decode-spinner');

                    // Call the main decoding function from image_processing.js
                    const rawDecodedString = await decodeVisualCodeFromImage(uploadedImagePreview);

                    hideSpinner('decode-spinner');
                    decodeButton.disabled = false;

                    if (rawDecodedString !== null && rawDecodedString !== undefined) {
                        // Process with MathJS for <linear>/<power> tags
                        const finalProcessedString = processDecodedStringWithMathJS(rawDecodedString); // from this ui.js file
                        updateDecodeResult(`<strong>Successfully Decoded:</strong><br><pre>${escapeHtml(finalProcessedString)}</pre>`);
                    } else {
                        // Error message should have been set by decodeVisualCodeFromImage or its sub-functions
                        // If rawDecodedString is null, it means decode failed at some point.
                        // updateDecodeResultUI in image_processing.js should handle intermediate error messages.
                        // If it reaches here as null, a final failure message is already displayed by image_processing.js
                        // console.log("decodeVisualCodeFromImage returned null, indicating failure.");
                    }
                } else {
                    updateDecodeResult('OpenCV.js is not ready. Please wait or check console.');
                    if (typeof checkOpenCVStatusAndAlert === 'function') checkOpenCVStatusAndAlert(); // from main.js
                }
            } else {
                console.error('decodeVisualCodeFromImage function is not defined.');
                updateDecodeResult('Error: Decoding function (decodeVisualCodeFromImage) not found.');
            }
        });
    }

    // Encoder UI
    if (encodeButton && encodeDataInput && qrCanvas && downloadQrLink) {
        encodeButton.addEventListener('click', async function() {
            const textToEncode = encodeDataInput.value;
            if (!textToEncode.trim()) {
                alert("Please enter text to encode.");
                return;
            }
            showSpinner('encode-spinner');
            encodeButton.disabled = true;
            try {
                // generateFullEncodedQrOnCanvas will be defined in custom_codec.js or a new encoder.js file later
                if (typeof generateFullEncodedQrOnCanvas === 'function') {
                    await generateFullEncodedQrOnCanvas(textToEncode, qrCanvas, downloadQrLink);
                } else {
                    console.error("generateFullEncodedQrOnCanvas not defined.");
                    alert("Encoding function not available. (generateFullEncodedQrOnCanvas missing)");
                }
            } catch (e) {
                console.error("Error during JS encoding:", e);
                alert("Error during encoding: " + e.message);
            } finally {
                hideSpinner('encode-spinner');
                encodeButton.disabled = false;
            }
        });
    } else {
        console.warn("Some UI elements for encoding were not found. Encoding UI might not work.");
    }
});

// UI Helper functions (updateDecodeResult, showSpinner, hideSpinner)
function updateDecodeResult(message) {
    const resultDiv = document.getElementById('decode-result');
    if (resultDiv) {
        resultDiv.innerHTML = message;
    }
}
function showSpinner(spinnerId) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) spinner.style.display = 'block';
}
function hideSpinner(spinnerId) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) spinner.style.display = 'none';
}
function escapeHtml(unsafe) {
    if (unsafe === null || unsafe === undefined) return "";
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

// Math.js processing functions
function solveLinearEquationWithMathJS(eqString) {
    try { const parts = eqString.split('='); if (parts.length !== 2) return "[Lin Error: Format]";
        const exprNode = math.parse(`${parts[0].trim()} - (${parts[1].trim()})`); const simplifiedExpr = math.simplify(exprNode);
        let b_val, a_plus_b_val;
        try { b_val = simplifiedExpr.evaluate({x: 0}); a_plus_b_val = simplifiedExpr.evaluate({x: 1}); }
        catch (e) { if (simplifiedExpr.isConstantNode) return Math.abs(simplifiedExpr.value) < 1e-9 ? "[Lin: Infinite sols]" : "[Lin: No sol]"; return "[Lin Error: Eval form ("+e.message.substring(0,20)+")]"; }
        if (typeof b_val !== 'number' || typeof a_plus_b_val !== 'number' ) return "[Lin Error: Coeffs NaN]";
        const a_val = a_plus_b_val - b_val;
        if (Math.abs(a_val) < 1e-9) return Math.abs(b_val) < 1e-9 ? "[Lin: Infinite sols (0=0)]" : "[Lin: No sol (const!=0)]";
        return `[Calc: x = ${math.format(-b_val / a_val, {precision: 14})}]`;
    } catch (err) { return "[Lin Error: " + err.message.substring(0,30) + "]"; }
}

function solvePowerEquationWithMathJS(eqString) {
    try { const parts = eqString.split('='); if (parts.length !== 2) return "[Pow Error: Format]";
        const exprNode = math.parse(`${parts[0].trim()} - (${parts[1].trim()})`); const simplified = math.simplify(exprNode);
        let a = 1, n_val, C_val; // Default a=1 for x^n = C form
        if (simplified.isOperatorNode && simplified.op === '-' && simplified.args.length === 2) {
            C_val = simplified.args[1].evaluate({});
            let term_with_x = simplified.args[0];
            if (term_with_x.isOperatorNode && term_with_x.op === '^' && term_with_x.args[0].isSymbolNode && term_with_x.args[0].name === 'x' && term_with_x.args[1].isConstantNode) {
                n_val = term_with_x.args[1].value; // x^n form
            } else if (term_with_x.isOperatorNode && term_with_x.op === '*' && term_with_x.args[0].isConstantNode && term_with_x.args[1].isOperatorNode && term_with_x.args[1].op === '^' && term_with_x.args[1].args[0].isSymbolNode && term_with_x.args[1].args[0].name === 'x' && term_with_x.args[1].args[1].isConstantNode) {
                a = term_with_x.args[0].value; n_val = term_with_x.args[1].args[1].value; // a*x^n form
            } else { return "[Pow: Equation form not recognized (ax^n - C expected)]"; }
        } else { return "[Pow: Equation form not recognized (expr - C expected)]"; }
        if (a === 0) return "[Pow Error: Coeff 'a' is zero]"; const valForPower = C_val / a;
        if (valForPower < 0 && n_val % 2 === 0) return `[Pow: No real sol for ${a}*x^${n_val}=${C_val}]`;
        const principalRoot = math.evaluate(`${valForPower}^(1/${n_val})`); let solutions = [principalRoot];
        if (n_val % 2 === 0 && Math.abs(valForPower) > 1e-9 && Math.abs(principalRoot) > 1e-9) { solutions.push(-principalRoot); }
        return `[Calc: x = ${solutions.map(s => math.format(s, {precision: 14})).join(' or ')}]`;
    } catch (err) { return "[Pow Error: " + err.message.substring(0,30) + "]"; }
}

function processDecodedStringWithMathJS(decodedString) {
    if (typeof math === 'undefined' || !decodedString || typeof decodedString !== 'string') {
        console.warn("Math.js not loaded or invalid input to processDecodedStringWithMathJS"); return decodedString;
    }
    const linearRegex = /<linear>(.*?)<\/linear>/g; const powerRegex = /<power>(.*?)<\/power>/g;
    let resultString = decodedString;
    resultString = resultString.replace(linearRegex, (match, eqContent) => `${match} ${solveLinearEquationWithMathJS(eqContent)}`);
    resultString = resultString.replace(powerRegex, (match, eqContent) => `${match} ${solvePowerEquationWithMathJS(eqContent)}`);
    return resultString;
}
// Appending JSDoc to js/ui.js

/** @fileoverview Handles UI interactions, event listeners, and DOM updates for the Custom Visual Code tool. */

// JSDoc for functions like updateDecodeResult, showSpinner, hideSpinner, escapeHtml,
// solveLinearEquationWithMathJS, solvePowerEquationWithMathJS, and processDecodedStringWithMathJS
// would be placed above their respective definitions. Example:

/**
 * Updates the content of the decoding result display area.
 * @param {String} message - The HTML string or plain text to display.
 */
// function updateDecodeResult(message) { ... }

/**
 * Processes a decoded string to evaluate embedded Math.js expressions for linear and power equations.
 * Appends the solution or an error message next to the respective tags.
 * @param {String} decodedString - The string potentially containing <linear>equation</linear> or <power>equation</power> tags.
 * @returns {String} The processed string with equation solutions appended.
 */
// function processDecodedStringWithMathJS(decodedString) { ... }
