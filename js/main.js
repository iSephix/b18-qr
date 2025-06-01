// js/main.js

/** @fileoverview Main script for application initialization, OpenCV loading, and test execution orchestration. */

// Callback queue for DOMContentLoaded
window.mainJsDOMContentLoadedCallbacks = window.mainJsDOMContentLoadedCallbacks || [];

let opencvLoadingStarted = false;
let opencvReadyNotified = false;

/**
 * Called when OpenCV.js is confirmed to be fully loaded and initialized.
 * Updates UI to indicate readiness.
 */
function onOpenCvReady() {
    console.info('OpenCV.js is ready (onOpenCvReady called).');
    // Enable UI elements that depend on OpenCV here if they were initially disabled.
    const decodeButton = document.getElementById('decode-button');
    if (decodeButton) {
        // The button's disabled state is primarily handled by image selection logic in ui.js,
        // but one could also explicitly enable it here if it started disabled.
        // decodeButton.disabled = false;
    }
    // Attempt to update UI status via functions possibly defined in ui.js or image_processing.js
    if (typeof updateDecodeResult === 'function') {
        updateDecodeResult('OpenCV.js loaded. Ready to decode images.');
    } else if (typeof updateDecodeResultUI === 'function') {
        updateDecodeResultUI('OpenCV.js loaded. Ready to decode images.');
    }
}


/**
 * Checks OpenCV status via polling. This is a fallback mechanism.
 * It will stop polling if cv.onRuntimeInitialized successfully notifies readiness.
 */
function checkOpenCVStatusPoll() {
    if (opencvReadyNotified) {
        console.info("Polling check: OpenCV readiness already notified. Stopping poll.");
        return;
    }

    if (typeof cv !== 'undefined' && cv.imread) { // Check for a function that should exist if CV is ready
        console.info("OpenCV.js detected as ready via polling mechanism.");
        opencvReadyNotified = true; // Set flag to prevent multiple notifications
        onOpenCvReady();
    } else {
        console.warn('OpenCV.js still not ready (polling). Retrying in 1 second...');
        // Optionally update UI about loading status
        if (typeof updateDecodeResult === 'function') {
            updateDecodeResult('OpenCV.js is taking a while to load. Please wait...');
        } else if (typeof updateDecodeResultUI === 'function') {
            updateDecodeResultUI('OpenCV.js is taking a while to load. Please wait...');
        }
        setTimeout(checkOpenCVStatusPoll, 1000); // Continue polling
    }
}

// Central DOMContentLoaded listener in main.js
document.addEventListener('DOMContentLoaded', () => {
    console.info("main.js: DOMContentLoaded event fired.");
    const opencvScriptTag = document.getElementById('opencv-script');

    if (opencvScriptTag) {
        // Handle script loading
        opencvScriptTag.onload = () => {
            console.info("OpenCV.js script file has loaded via 'onload' event.");
            opencvLoadingStarted = true;

            if (typeof cv !== 'undefined' && typeof cv.onRuntimeInitialized === 'function') {
                console.info("'cv.onRuntimeInitialized' is defined. Setting it as the primary callback for OpenCV readiness.");
                cv.onRuntimeInitialized = () => {
                    if (opencvReadyNotified) {
                        console.info("cv.onRuntimeInitialized: Readiness already notified. Skipping.");
                        return;
                    }
                    console.info("cv.onRuntimeInitialized() has been called by the OpenCV runtime.");
                    opencvReadyNotified = true;
                    onOpenCvReady();
                };

                // Check if cv is already usable (e.g. if onRuntimeInitialized fired very quickly or script was cached)
                if (typeof cv.imread === 'function' && !opencvReadyNotified) {
                    console.info("OpenCV seems usable immediately after script 'onload' (before onRuntimeInitialized explicitly fired, or it already did). Notifying readiness.");
                    opencvReadyNotified = true;
                    onOpenCvReady();
                } else if (!opencvReadyNotified) {
                     console.info("Waiting for cv.onRuntimeInitialized() to be called by OpenCV runtime...");
                     // Start a timeout for polling as a safety net in case onRuntimeInitialized doesn't fire
                     setTimeout(checkOpenCVStatusPoll, 2000); // Start polling after 2s if not ready
                }

            } else {
                // Fallback if cv.onRuntimeInitialized is not available (e.g. older OpenCV.js versions or non-WASM builds)
                console.warn("'cv.onRuntimeInitialized' is NOT defined. Using polling fallback for OpenCV readiness.");
                if (typeof cv !== 'undefined' && typeof cv.imread === 'function' && !opencvReadyNotified) {
                    console.info("OpenCV is usable directly after script 'onload' (no onRuntimeInitialized property found).");
                    opencvReadyNotified = true;
                    onOpenCvReady();
                } else if (!opencvReadyNotified) {
                    console.info("Starting polling for OpenCV readiness immediately.");
                    setTimeout(checkOpenCVStatusPoll, 100);
                }
            }
        };
        opencvScriptTag.onerror = () => {
            console.error("Failed to load OpenCV.js script from CDN.");
            opencvReadyNotified = true; // Prevent further polling attempts
            const errorMsg = "ERROR: Failed to load OpenCV.js. Decoding will not work.";
            if (typeof updateDecodeResult === 'function') updateDecodeResult(errorMsg);
            else if (typeof updateDecodeResultUI === 'function') updateDecodeResultUI(errorMsg);
        };
    } else {
        console.error("OpenCV script tag ('opencv-script') not found in HTML. Cannot initialize OpenCV.");
        opencvReadyNotified = true; // Prevent polling
        const errorMsg = "ERROR: OpenCV script tag missing. Decoding will not work.";
        if (typeof updateDecodeResult === 'function') updateDecodeResult(errorMsg);
        else if (typeof updateDecodeResultUI === 'function') updateDecodeResultUI(errorMsg);
    }

    // Call all registered DOMContentLoaded callbacks from other modules
    console.info("main.js: Executing " + window.mainJsDOMContentLoadedCallbacks.length + " queued DOMContentLoaded callbacks.");
    window.mainJsDOMContentLoadedCallbacks.forEach(callback => {
        try { callback(); } catch (e) { console.error("Error in DOMContentLoaded callback:", e);}
    });
    window.mainJsDOMContentLoadedCallbacks = []; // Clear queue after execution
});


/**
 * Runs all defined client-side JavaScript tests for different modules.
 * Logs results to console and updates a UI area with a summary.
 * @returns {Promise<Boolean>} A promise that resolves to true if all tests pass, false otherwise.
 */
async function runAllTests() {
    console.log("=== Running All Client-Side Tests ===");
    const resultDisplayArea = document.getElementById('decode-result');
    if(resultDisplayArea) resultDisplayArea.innerHTML = "<p>Running tests...</p>";

    if (typeof showSpinner === 'function') showSpinner('decode-spinner');
    else if (typeof showSpinnerUI === 'function') showSpinnerUI('decode-spinner');

    let allPass = true;
    try {
        if (typeof testGF19 === 'function') { console.info("Running GF19 tests..."); if(!testGF19()) allPass = false; }
        else { console.error("testGF19 not defined."); allPass = false; }

        if (typeof testPolynomialGF19 === 'function') { console.info("Running PolynomialGF19 tests..."); if(!testPolynomialGF19()) allPass = false; }
        else { console.error("testPolynomialGF19 not defined."); allPass = false; }

        if (typeof testReedSolomon === 'function') { console.info("Running ReedSolomon tests..."); if(!testReedSolomon()) allPass = false; }
        else { console.error("testReedSolomon not defined."); allPass = false; }

        if (typeof testCustomCodec === 'function') { console.info("Running CustomCodec tests..."); if(!await testCustomCodec()) allPass = false; }
        else { console.error("testCustomCodec not defined."); allPass = false; }
    } catch (e) {
        console.error("Error during test execution:", e.stack || e);
        allPass = false;
    }

    console.log(`=== ALL TESTS COMPLETED. Overall Result: ${allPass ? 'ALL PASS' : 'SOME FAILURES'} ===`);
    if (typeof hideSpinner === 'function') hideSpinner('decode-spinner');
    else if (typeof hideSpinnerUI === 'function') hideSpinnerUI('decode-spinner');

    const resultMsg = `<h3>Test Results:</h3><p>Overall: ${allPass ? 'ALL PASS' : 'SOME FAILURES (check console for details)'}</p>`;
    if (resultDisplayArea) resultDisplayArea.innerHTML = resultMsg;
    else if (typeof updateDecodeResult === 'function') updateDecodeResult(resultMsg); // Fallback if main area is missing
    else if (typeof updateDecodeResultUI === 'function') updateDecodeResultUI(resultMsg);

    return allPass;
}

// Register test button listener via the callback queue
window.mainJsDOMContentLoadedCallbacks.push(() => {
    const testButton = document.getElementById('run-tests-button');
    if (testButton) {
        testButton.addEventListener('click', runAllTests);
        console.info("Test button ('run-tests-button') listener added via main.js callback queue.");
    } else {
        console.warn("Test button ('run-tests-button') not found during main.js DOMContentLoaded setup.");
    }
});

// JSDoc stubs (actual JSDoc comments are above function definitions)
// /** @fileoverview Main script for application initialization, OpenCV loading, and test execution. */
// /** Handles OpenCV.js readiness. */
// function onOpenCvReady() { ... }
// /** Checks OpenCV status and alerts user or retries. */
// function checkOpenCVStatusPoll() { ... } // Renamed from checkOpenCVStatusAndAlert
// /** Runs all defined client-side tests. @returns {Promise<Boolean>} True if all tests pass. */
// async function runAllTests() { ... }
