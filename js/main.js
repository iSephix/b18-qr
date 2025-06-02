// js/main.js

/** @fileoverview Main script for application initialization and test execution orchestration. */

// Callback queue for DOMContentLoaded
window.mainJsDOMContentLoadedCallbacks = window.mainJsDOMContentLoadedCallbacks || [];

// Central DOMContentLoaded listener in main.js
document.addEventListener('DOMContentLoaded', () => {
    console.info("main.js: DOMContentLoaded event fired.");
    // OpenCV loading logic has been removed.
    // JSFeat is loaded directly via script tag in index.html and should be available.
    // Math.js is also loaded directly.

    // Call all registered DOMContentLoaded callbacks from other modules
    console.info("main.js: Executing " + (window.mainJsDOMContentLoadedCallbacks ? window.mainJsDOMContentLoadedCallbacks.length : 0) + " queued DOMContentLoaded callbacks.");
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

        if (typeof testImageProcessingLogic === 'function') { console.info("Running ImageProcessingLogic tests..."); if(!testImageProcessingLogic()) allPass = false; }
        else { console.error("testImageProcessingLogic not defined."); allPass = false; }
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
