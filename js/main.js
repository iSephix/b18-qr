// js/main.js

function onOpenCvReady() {
    console.log('OpenCV.js is ready.');
    // You can enable UI elements that depend on OpenCV here if needed
    const decodeButton = document.getElementById('decode-button');
    if (decodeButton) {
        // decodeButton.disabled = false; // Assuming it starts disabled until OpenCV is ready
        // UI script now handles enabling button after image load, so this might not be strictly needed here
        // but good for initial state.
    }
    // Initial message
    updateDecodeResult('OpenCV.js loaded. Ready to decode images.');
}

function checkOpenCVStatusAndAlert() {
    if (typeof cv !== 'undefined' && cv.imread) { // Check a function that should exist
        onOpenCVReady();
    } else {
        console.warn('OpenCV.js is not yet loaded. Retrying in 1 second...');
        updateDecodeResult('OpenCV.js is loading or failed to load. Please wait or check the browser console for errors related to opencv.js.');
        setTimeout(checkOpenCVStatusAndAlert, 1000); // Check again after a delay
    }
}


// Wait for the DOM to be fully loaded before trying to check OpenCV status
// or attaching event listeners that depend on DOM elements.
document.addEventListener('DOMContentLoaded', () => {
    // Check if the OpenCV script element exists
    const opencvScript = document.getElementById('opencv-script');
    if (opencvScript) {
        opencvScript.onload = () => {
            console.log("OpenCV script has loaded via onload event.");
            // Call onOpenCvReady or a more robust check
            if (typeof cv !== 'undefined' && cv.imread) {
                 onOpenCVReady();
            } else {
                 // Fallback if cv is not immediately available after onload (should be rare for 'async' but good practice)
                 console.warn("OpenCV script loaded, but 'cv' object not found immediately. Will retry.");
                 checkOpenCVStatusAndAlert();
            }
        };
        opencvScript.onerror = () => {
            console.error("Failed to load OpenCV.js script from CDN.");
            updateDecodeResult("ERROR: Failed to load OpenCV.js. Decoding will not work. Check your internet connection or ad-blockers.");
        };
    } else {
        console.error("OpenCV script tag not found in HTML.");
        updateDecodeResult("ERROR: OpenCV script tag not found. Decoding will not work.");
    }

    // Initial check in case script was already loaded (e.g. from cache and very fast)
    // but typically onload is more reliable for async scripts.
    if (typeof cv !== 'undefined' && cv.imread) {
        onOpenCVReady();
    } else if (opencvScript && !opencvScript.onload) { // If onload was not set because script tag was missing
        // This case is unlikely if the DOMContentLoaded is correctly waiting.
        // For robustness, one might initiate a polling check here if the script tag exists but no onload was attached.
        // However, with the current structure, the onload event on the script tag is the primary mechanism.
    }
});

// Ensure UI helper functions are globally available if called from other scripts directly
// or ensure they are only called from ui.js which defines them.
// For now, ui.js re-defines them, which is fine as long as image_processing.js uses the ones it defines.
// To avoid conflict and ensure image_processing.js can update UI:
// It's better if image_processing.js calls functions exposed from ui.js or main.js.
// For simplicity in this step, image_processing.js defines its own for now.
// This will be refactored later if needed.
// Appending runAllTests to js/main.js

async function runAllTests() {
    console.log("=== Running All Client-Side Tests ===");
    const resultDisplayArea = document.getElementById('decode-result'); // Use existing area
    resultDisplayArea.innerHTML = "<p>Running tests...</p>";
    showSpinner('decode-spinner');

    let allPass = true;
    if (typeof testGF19 === 'function') { if(!testGF19()) allPass = false; }
    else { console.error("testGF19 not defined."); allPass = false; }

    if (typeof testPolynomialGF19 === 'function') { if(!testPolynomialGF19()) allPass = false; }
    else { console.error("testPolynomialGF19 not defined."); allPass = false; }

    if (typeof testReedSolomon === 'function') { if(!testReedSolomon()) allPass = false; }
    else { console.error("testReedSolomon not defined."); allPass = false; }

    if (typeof testCustomCodec === 'function') { if(!await testCustomCodec()) allPass = false; }
    else { console.error("testCustomCodec not defined."); allPass = false; }

    console.log(`=== ALL TESTS COMPLETED. Overall Result: ${allPass ? 'ALL PASS' : 'SOME FAILURES'} ===`);
    hideSpinner('decode-spinner');

    if (resultDisplayArea) {
        resultDisplayArea.innerHTML = `<h3>Test Results:</h3><p>Overall: ${allPass ? 'ALL PASS' : 'SOME FAILURES (check console for details)'}</p>`;
    }
    return allPass;
}

// Modify DOMContentLoaded to add test button listener if button exists
// This assumes DOMContentLoaded might already be in main.js for OpenCV init
// We'll ensure it's structured to allow multiple initializations.
if (typeof mainJsDOMContentLoadedCallbacks === 'undefined') {
    window.mainJsDOMContentLoadedCallbacks = [];
}
mainJsDOMContentLoadedCallbacks.push(() => {
    const testButton = document.getElementById('run-tests-button');
    if (testButton) {
        testButton.addEventListener('click', runAllTests);
        console.log("Test button listener added.");
    }
});

// Central DOMContentLoaded listener in main.js
document.addEventListener('DOMContentLoaded', () => {
    // OpenCV related init (from previous main.js setup)
    const opencvScript = document.getElementById('opencv-script');
    if (opencvScript) {
        opencvScript.onload = () => {
            console.log("OpenCV script has loaded via onload event.");
            if (typeof cv !== 'undefined' && cv.imread) {
                 if(typeof onOpenCvReady === 'function') onOpenCvReady(); // onOpenCvReady is in main.js
            } else {
                 console.warn("OpenCV script loaded, but 'cv' object not found. Retrying check.");
                 if(typeof checkOpenCVStatusAndAlert === 'function') checkOpenCVStatusAndAlert(); // in main.js
            }
        };
        opencvScript.onerror = () => {
            console.error("Failed to load OpenCV.js script from CDN.");
            if(typeof updateDecodeResult === 'function') updateDecodeResult("ERROR: Failed to load OpenCV.js."); // updateDecodeResult from ui.js
            else if(typeof updateDecodeResultUI === 'function') updateDecodeResultUI("ERROR: Failed to load OpenCV.js."); // or from image_processing.js
        };
    } else {
        console.error("OpenCV script tag not found in HTML.");
         if(typeof updateDecodeResult === 'function') updateDecodeResult("ERROR: OpenCV script tag not found.");
    }
    if (typeof cv !== 'undefined' && cv.imread) { // Initial check if already loaded
        if(typeof onOpenCvReady === 'function') onOpenCvReady();
    }

    // Call all registered DOMContentLoaded callbacks
    if(window.mainJsDOMContentLoadedCallbacks) {
        window.mainJsDOMContentLoadedCallbacks.forEach(callback => callback());
    }
});

// onOpenCvReady and checkOpenCVStatusAndAlert should be in main.js from before
function onOpenCvReady() {
    console.log('OpenCV.js is ready (onOpenCvReady).');
    if(typeof updateDecodeResult === 'function') updateDecodeResult('OpenCV.js loaded. Ready to decode images.');
    else if(typeof updateDecodeResultUI === 'function') updateDecodeResultUI('OpenCV.js loaded. Ready to decode images.');
}

function checkOpenCVStatusAndAlert() {
    if (typeof cv !== 'undefined' && cv.imread) {
        onOpenCvReady();
    } else {
        console.warn('OpenCV.js is not yet loaded. Retrying in 1 second...');
        if(typeof updateDecodeResult === 'function') updateDecodeResult('OpenCV.js is loading or failed. Retrying...');
        else if(typeof updateDecodeResultUI === 'function') updateDecodeResultUI('OpenCV.js is loading or failed. Retrying...');
        setTimeout(checkOpenCVStatusAndAlert, 1000);
    }
}

// Appending JSDoc to js/main.js

/** @fileoverview Main script for application initialization, OpenCV loading checks, and test execution orchestration. */

/**
 * Called when OpenCV.js is confirmed to be loaded and ready.
 * Updates UI to indicate readiness.
 */
// function onOpenCvReady() { ... }

/**
 * Checks if OpenCV.js (cv object and a function like cv.imread) is available.
 * If not, schedules a retry and updates the UI. Calls onOpenCvReady upon success.
 */
// function checkOpenCVStatusAndAlert() { ... }

/**
 * Runs all defined client-side JavaScript tests for different modules.
 * Logs results to console and updates a UI area with a summary.
 * @returns {Promise<Boolean>} A promise that resolves to true if all tests pass, false otherwise.
 */
// async function runAllTests() { ... }
