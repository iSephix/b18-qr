// js/image_processing.js

// Constants (defined in previous steps)
const colorRgbMapJs = {
    'black': { r: 0, g: 0, b: 0 }, 'white': { r: 255, g: 0, b: 255 }, 'blue': { r: 0, g: 0, b: 255 },
    'green': { r: 0, g: 255, b: 0 }, 'yellow': { r: 255, g: 255, b: 0 }, 'red': { r: 255, g: 0, b: 0 },
    'gray': { r: 128, g: 128, b: 128 }, 'background': { r: 255, g: 255, b: 255 }
};
const shapesJs = ['square', 'circle', 'triangle'];
const colorsJs = ['black', 'white', 'blue', 'green', 'yellow', 'red'];
const symbolListJs = [];
for (const color of colorsJs) { for (const shape of shapesJs) { symbolListJs.push([color, shape]); }}
const specialSymbolJs = ['gray', 'square'];
const markerSymbolOuterJs = ['black', 'square'];
const markerSymbolInnerJs = ['white', 'square'];
const markerPatternJsDefinition = [
    [markerSymbolOuterJs, markerSymbolOuterJs, markerSymbolOuterJs],
    [markerSymbolOuterJs, markerSymbolInnerJs, markerSymbolOuterJs],
    [markerSymbolOuterJs, markerSymbolOuterJs, markerSymbolOuterJs]
];
const markerSizeCells = markerPatternJsDefinition.length;
const WARPED_IMAGE_SIZE = 300;
const POSSIBLE_GRID_DIMS = [21, 25, 29, 17, 33];
const MAX_ACCEPTABLE_COLOR_DISTANCE = 150;

/**
 * Identifies the visual symbol (color and shape) within a given cell matrix.
 * @param {cv.Mat} cellMat_rgba - The OpenCV Mat for a single cell, in RGBA format.
 * @returns {Array<String>} An array containing [colorName, shapeName], e.g., ['blue', 'circle'], or specialSymbolJs on failure.
 */
function identifySymbolJs(cellMat_rgba) {
    if (cellMat_rgba.empty() || cellMat_rgba.cols < 3 || cellMat_rgba.rows < 3) return specialSymbolJs;
    let detectedColorName = 'gray'; let detectedShape = 'square';
    let grayCell = null, binaryCellForShape = null, contours = null, hierarchy = null, approxContour = null, largestContour = null; // Ensure proper scope for finally
    try {
        const rows = cellMat_rgba.rows; const cols = cellMat_rgba.cols; const channels = cellMat_rgba.channels();
        const symbolPixelsR = []; const symbolPixelsG = []; const symbolPixelsB = [];
        const bgColor = colorRgbMapJs.background; const colorTolerance = 60;
        for (let y = 0; y < rows; y++) { for (let x = 0; x < cols; x++) {
            const r = cellMat_rgba.data[y * cellMat_rgba.step + x * channels + 0];
            const g = cellMat_rgba.data[y * cellMat_rgba.step + x * channels + 1];
            const b = cellMat_rgba.data[y * cellMat_rgba.step + x * channels + 2];
            const isBg = Math.abs(r - bgColor.r) <= colorTolerance && Math.abs(g - bgColor.g) <= colorTolerance && Math.abs(b - bgColor.b) <= colorTolerance;
            if (!isBg) { symbolPixelsR.push(r); symbolPixelsG.push(g); symbolPixelsB.push(b); } } }
        if (symbolPixelsR.length < (rows * cols * 0.1)) return specialSymbolJs;
        symbolPixelsR.sort((a, b) => a - b); symbolPixelsG.sort((a, b) => a - b); symbolPixelsB.sort((a, b) => a - b);
        const medianR = symbolPixelsR[Math.floor(symbolPixelsR.length / 2)];
        const medianG = symbolPixelsG[Math.floor(symbolPixelsG.length / 2)];
        const medianB = symbolPixelsB[Math.floor(symbolPixelsB.length / 2)];
        let minDistance = Infinity;
        for (const colorName in colorRgbMapJs) { if (colorName === 'background') continue;
            const mapColor = colorRgbMapJs[colorName];
            const distance = Math.sqrt(Math.pow(medianR - mapColor.r, 2) + Math.pow(medianG - mapColor.g, 2) + Math.pow(medianB - mapColor.b, 2));
            if (distance < minDistance) { minDistance = distance; detectedColorName = colorName; } }
        if (minDistance > MAX_ACCEPTABLE_COLOR_DISTANCE && detectedColorName !== 'gray') return specialSymbolJs;
        if (detectedColorName === 'gray') return specialSymbolJs;
        grayCell = new cv.Mat(); cv.cvtColor(cellMat_rgba, grayCell, cv.COLOR_RGBA2GRAY);
        binaryCellForShape = new cv.Mat(); cv.threshold(grayCell, binaryCellForShape, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
        if (cv.mean(binaryCellForShape)[0] > 128) { cv.bitwise_not(binaryCellForShape, binaryCellForShape); }
        contours = new cv.MatVector(); hierarchy = new cv.Mat();
        cv.findContours(binaryCellForShape, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        if (contours.size() === 0) { return [detectedColorName, 'square'];}
        largestContour = contours.get(0); let maxArea = cv.contourArea(largestContour);
        for (let i = 1; i < contours.size(); i++) { const c = contours.get(i); const area = cv.contourArea(c);
            if (area > maxArea) { if(largestContour && !largestContour.isDeleted()) largestContour.delete(); largestContour = c.clone(); maxArea = area; }
            if(c && !c.isDeleted()) c.delete();
        }
        if (maxArea < (cellMat_rgba.rows * cellMat_rgba.cols * 0.05)) { if(largestContour && !largestContour.isDeleted()) largestContour.delete(); return [detectedColorName, 'square'];}
        approxContour = new cv.Mat(); const epsilon = 0.035 * cv.arcLength(largestContour, true);
        cv.approxPolyDP(largestContour, approxContour, epsilon, true);
        const numVertices = approxContour.rows;
        if (numVertices === 3) detectedShape = 'triangle';
        else if (numVertices === 4) detectedShape = 'square';
        else if (numVertices > 4 && numVertices <= 8) detectedShape = 'circle';
        else detectedShape = 'square';
        if(largestContour && !largestContour.isDeleted()) largestContour.delete();
    } catch (e) { console.error("identifySymbolJs error:", e.stack); return [detectedColorName, 'square']; }
    finally {
        if (grayCell && !grayCell.isDeleted()) grayCell.delete();
        if (binaryCellForShape && !binaryCellForShape.isDeleted()) binaryCellForShape.delete();
        if (contours && !contours.isDeleted()) contours.delete();
        if (hierarchy && !hierarchy.isDeleted()) hierarchy.delete();
        if (approxContour && !approxContour.isDeleted()) approxContour.delete();
        if (largestContour && !largestContour.isDeleted()) largestContour.delete(); // Ensure cloned largestContour is deleted
    }
    return [detectedColorName, detectedShape];
}

/**
 * Main function to decode a visual code from an image element.
 * Orchestrates loading the image, finding markers, perspective correction,
 * grid iteration, symbol identification, and calling the Reed-Solomon decoding pipeline.
 * @param {HTMLImageElement} imageElement - The HTML image element containing the visual code.
 * @returns {Promise<String|null>} A promise that resolves to the decoded string, or null if decoding fails.
 */
async function decodeVisualCodeFromImage(imageElement) {
    if (!cv || !cv.imread) {
        console.error('OpenCV.js is not loaded yet.');
        updateDecodeResultUI('Error: OpenCV.js is not ready. Please try again in a moment.');
        return null;
    }
    console.info('Starting visual code decoding...');
    updateDecodeResultUI('Processing image...');
    showSpinnerUI('decode-spinner');

    let src = null, gray = null, binary = null, warpedImg = null;
    let decodedString = null;

    try {
        src = cv.imread(imageElement);
        gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
        binary = new cv.Mat(); cv.adaptiveThreshold(gray, binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);

        const markerObjects = findMarkers_JS(binary, markerPatternJsDefinition);
        if (!markerObjects || markerObjects.length < 3) {
            updateDecodeResultUI(`Decoding failed: Could not find enough markers. Found ${markerObjects ? markerObjects.length : 0}. Need 3.`);
            return null;
        }

        const cornerMarkerPoints = selectCornerMarkers_JS(markerObjects, src.cols, src.rows);
        if (!cornerMarkerPoints) {
           updateDecodeResultUI('Decoding failed: Could not select corner markers.');
           return null;
        }

        warpedImg = perspectiveTransform_JS(src, cornerMarkerPoints);
        if (!warpedImg || warpedImg.empty()) {
            updateDecodeResultUI('Decoding failed: Perspective transform failed or resulted in empty image.');
            return null;
        }
        console.info('Image warped. Identifying symbols from grid...');

        for (const gridDim of POSSIBLE_GRID_DIMS) {
            console.info(`Attempting decode with grid dimension: ${gridDim}x${gridDim}`);
            updateDecodeResultUI(`Trying grid size: ${gridDim}x${gridDim}...`);

            const singleCellSizePx = WARPED_IMAGE_SIZE / gridDim;
            const currentSymbolsFlatIndices = [];

            for (let rCell = 0; rCell < gridDim; rCell++) {
                for (let cCell = 0; cCell < gridDim; cCell++) {
                    const isTopLeftMarkerArea = (rCell < markerSizeCells && cCell < markerSizeCells);
                    const isTopRightMarkerArea = (rCell < markerSizeCells && cCell >= gridDim - markerSizeCells);
                    const isBottomLeftMarkerArea = (rCell >= gridDim - markerSizeCells && cCell < markerSizeCells);
                    if (isTopLeftMarkerArea || isTopRightMarkerArea || isBottomLeftMarkerArea) continue;

                    const xStart = Math.floor(cCell * singleCellSizePx);
                    const yStart = Math.floor(rCell * singleCellSizePx);
                    const cellWidth = Math.floor((cCell + 1) * singleCellSizePx) - xStart;
                    const cellHeight = Math.floor((rCell + 1) * singleCellSizePx) - yStart;
                    if (cellWidth <=0 || cellHeight <=0) continue;

                    const insetPx = Math.floor(singleCellSizePx * 0.15);
                    const roiX = xStart + insetPx; const roiY = yStart + insetPx;
                    const roiW = Math.max(1, cellWidth - 2 * insetPx); const roiH = Math.max(1, cellHeight - 2 * insetPx);

                    if (roiX < 0 || roiY < 0 || roiX + roiW > warpedImg.cols || roiY + roiH > warpedImg.rows || roiW <=0 || roiH <=0) {
                        currentSymbolsFlatIndices.push(symbolListJs.length); continue;
                    }

                    let cellMat_rgba = null; let identifiedSymbol = null;
                    try {
                        cellMat_rgba = warpedImg.roi(new cv.Rect(roiX, roiY, roiW, roiH));
                        identifiedSymbol = cellMat_rgba.empty() ? specialSymbolJs : identifySymbolJs(cellMat_rgba);
                    } catch(e) { console.error("Error in cell ROI processing:", e.stack); identifiedSymbol = specialSymbolJs; }
                    finally { if (cellMat_rgba && !cellMat_rgba.isDeleted()) cellMat_rgba.delete(); }

                    let symbolIndex = symbolListJs.findIndex(s => s[0] === identifiedSymbol[0] && s[1] === identifiedSymbol[1]);
                    if (symbolIndex === -1) symbolIndex = symbolListJs.length;
                    currentSymbolsFlatIndices.push(symbolIndex);
                }
            }

            if (currentSymbolsFlatIndices.length === 0) { console.warn(`Grid ${gridDim}: No symbols extracted.`); continue; }

            if (typeof fullDecodePipelineFromIndices === 'function') {
                const resultString = await fullDecodePipelineFromIndices(currentSymbolsFlatIndices);
                if (resultString !== null && resultString !== undefined) {
                    decodedString = resultString;
                    console.info(`Successfully decoded with grid ${gridDim}x${gridDim}!`);
                    break;
                } else { console.warn(`Grid ${gridDim}: fullDecodePipelineFromIndices returned null.`); updateDecodeResultUI(`Grid ${gridDim}x${gridDim} decode attempt failed. Trying next...`);}
            } else { console.error('fullDecodePipelineFromIndices function is not available.'); break; }
        }

        if (!decodedString) {
            updateDecodeResultUI('Failed to decode: Could not extract valid symbols or decode with any attempted grid dimension.');
        }
    } catch (error) {
        console.error('Error during image decoding pipeline:', error, error.stack);
        updateDecodeResultUI(`Error: ${error.message}`);
    } finally {
        if (src && !src.isDeleted()) src.delete();
        if (gray && !gray.isDeleted()) gray.delete();
        if (binary && !binary.isDeleted()) binary.delete();
        if (warpedImg && !warpedImg.isDeleted()) warpedImg.delete();
        hideSpinnerUI('decode-spinner');
    }
    return decodedString;
}

/**
 * Verifies if a candidate image region matches the expected 3x3 marker pattern.
 * @param {cv.Mat} markerCandidateImage - A binary OpenCV Mat of the candidate region.
 * @param {Array<Array<Array<String>>>} expectedPatternSymbols - The 3x3 marker pattern definition.
 * @returns {Boolean} True if the pattern matches, false otherwise.
 */
function verifyMarkerPatternCV_JS(markerCandidateImage, expectedPatternSymbols) {
    const height = markerCandidateImage.rows; const width = markerCandidateImage.cols;
    if (height === 0 || width === 0) return false;
    const cell_h = Math.floor(height / markerSizeCells); const cell_w = Math.floor(width / markerSizeCells);
    if (cell_h === 0 || cell_w === 0) return false;
    const expectedBinaryPattern = [];
    for (let r = 0; r < markerSizeCells; r++) { const currentRow = []; for (let c = 0; c < markerSizeCells; c++) {
            const colorName = expectedPatternSymbols[r][c][0];
            if (colorName === 'black') currentRow.push(255); else if (colorName === 'white') currentRow.push(0); else return false;
        } expectedBinaryPattern.push(currentRow); }
    const observedPattern = [];
    for (let r_cell = 0; r_cell < markerSizeCells; r_cell++) { const observedRow = []; for (let c_cell = 0; c_cell < markerSizeCells; c_cell++) {
            const cellRect = new cv.Rect(c_cell * cell_w, r_cell * cell_h, cell_w, cell_h);
            let cell = null; // Define for finally block
            try {
                cell = markerCandidateImage.roi(cellRect);
                if (cell.empty()) return false;
                const meanIntensity = cv.mean(cell)[0]; observedRow.push(meanIntensity > 127 ? 255 : 0);
            } finally { if(cell && !cell.isDeleted()) cell.delete(); }
        } observedPattern.push(observedRow); }
    for (let r = 0; r < markerSizeCells; r++) { for (let c = 0; c < markerSizeCells; c++) { if (observedPattern[r][c] !== expectedBinaryPattern[r][c]) return false; }}
    return true;
}

/**
 * Finds all 3x3 markers in a binary image.
 * @param {cv.Mat} binaryImageMat - An OpenCV binary image (THRESH_BINARY_INV, markers are white).
 * @param {Array<Array<Array<String>>>} expectedPattern - The 3x3 marker pattern definition.
 * @returns {Array<Object>} A list of verified marker objects, each like {rect: {x,y,w,h}, points: [{x,y},...]}.
 */
function findMarkers_JS(binaryImageMat, expectedPattern) {
    let contours = new cv.MatVector(); let hierarchy = new cv.Mat(); const foundMarkers = [];
    cv.findContours(binaryImageMat, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < contours.size(); ++i) {
        let currentContour = contours.get(i);
        let currentHierarchy = hierarchy.data32S.slice(i * 4, (i + 1) * 4);
        let area = cv.contourArea(currentContour);
        let approx = null, markerROI = null, resizedROI = null;
        try {
            if (area < 100) continue;
            approx = new cv.Mat(); cv.approxPolyDP(currentContour, approx, 0.04 * cv.arcLength(currentContour, true), true);
            if (approx.rows === 4 && cv.isContourConvex(approx)) {
                let rect = cv.boundingRect(approx); let ar = rect.width / rect.height;
                if (ar >= 0.75 && ar <= 1.25) {
                    let childIdx = currentHierarchy[2]; if (childIdx !== -1) {
                        let grandchildIdx = hierarchy.data32S[childIdx * 4 + 2]; if (grandchildIdx !== -1) {
                            markerROI = binaryImageMat.roi(rect);
                            let stdVerifySize = 3 * 15; let dVS = new cv.Size(stdVerifySize, stdVerifySize);
                            resizedROI = new cv.Mat(); cv.resize(markerROI, resizedROI, dVS, 0, 0, cv.INTER_NEAREST);
                            if (verifyMarkerPatternCV_JS(resizedROI, expectedPattern)) {
                                let points = []; for (let ptIdx = 0; ptIdx < approx.rows; ptIdx++) { points.push({ x: approx.data32S[ptIdx*2], y: approx.data32S[ptIdx*2+1] }); }
                                foundMarkers.push({ rect: rect, points: points, contourForNMS: approx.clone() }); // Clone for NMS
                            }
                        } } } }
        } catch(e) { console.error("Error in findMarkers_JS loop:", e.stack); }
        finally {
            if(currentContour && !currentContour.isDeleted()) currentContour.delete(); // From contours.get(i)
            if(approx && !approx.isDeleted()) approx.delete();
            if(markerROI && !markerROI.isDeleted()) markerROI.delete();
            if(resizedROI && !resizedROI.isDeleted()) resizedROI.delete();
        }
    }
    if(contours && !contours.isDeleted()) contours.delete();
    if(hierarchy && !hierarchy.isDeleted()) hierarchy.delete();
    if (foundMarkers.length === 0) return [];

    foundMarkers.sort((a, b) => (b.rect.width * b.rect.height) - (a.rect.width * a.rect.height));
    const finalMarkers = []; const iouThreshold = 0.3;
    while (foundMarkers.length > 0) {
        const current = foundMarkers.shift(); finalMarkers.push(current); let remaining = [];
        for (const other of foundMarkers) {
            const x1 = Math.max(current.rect.x, other.rect.x); const y1 = Math.max(current.rect.y, other.rect.y);
            const x2 = Math.min(current.rect.x + current.rect.width, other.rect.x + other.rect.width);
            const y2 = Math.min(current.rect.y + current.rect.height, other.rect.y + other.rect.height);
            const interW = Math.max(0, x2 - x1); const interH = Math.max(0, y2 - y1); const interA = interW * interH;
            const currentA = current.rect.width * current.rect.height; const otherA = other.rect.width * other.rect.height;
            const unionA = currentA + otherA - interA;
            if (unionA === 0 || (interA / unionA) < iouThreshold) { remaining.push(other); }
            else { if(other.contourForNMS && !other.contourForNMS.isDeleted()) other.contourForNMS.delete(); } // Delete suppressed
        } foundMarkers.splice(0, foundMarkers.length, ...remaining);
    }
    // Clean up remaining contours from finalMarkers
    finalMarkers.forEach(m => { if(m.contourForNMS && !m.contourForNMS.isDeleted()) m.contourForNMS.delete(); delete m.contourForNMS; });
    return finalMarkers;
}

function getCenter(rect) { return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }; }

/**
 * Selects the top-left (TL), top-right (TR), and bottom-left (BL) markers from a list of detected markers.
 * @param {Array<Object>} markerObjects - List of marker objects, each {rect: {x,y,w,h}, points: [...]}.
 * @param {Number} imageWidth - Width of the original image.
 * @param {Number} imageHeight - Height of the original image.
 * @returns {Object|null} An object {TL: {x,y}, TR: {x,y}, BL: {x,y}} for the corners of the overall code area, or null if selection fails.
 */
function selectCornerMarkers_JS(markerObjects, imageWidth, imageHeight) {
    if (markerObjects.length < 3) { console.warn("Not enough markers for corner selection:", markerObjects.length); return null; }
    const markers = markerObjects.map(m => ({ ...m, center: getCenterOfPoints(m.points) })); // Use center of points for better accuracy
    markers.sort((a, b) => (a.center.x + a.center.y) - (b.center.x + b.center.y));
    const tl = markers[0];
    const remaining = markers.slice(1).map(m => ({...m, dx: m.center.x - tl.center.x, dy: m.center.y - tl.center.y, angle: Math.atan2(m.center.y - tl.center.y, m.center.x - tl.center.x)}));
    if (remaining.length < 2) { console.warn("Not enough remaining markers after TL selection."); return null; }

    remaining.sort((a,b) => a.angle - b.angle);

    let tr = null, bl = null;
    let bestTR = null, bestBL = null;
    let minAngleDiffTR = Math.PI, minAngleDiffBL = Math.PI;

    for(const cand of remaining){
        if(cand.dx > 0.1 * tl.rect.width ){ let angleDiff = Math.abs(cand.angle); if(angleDiff < minAngleDiffTR){ minAngleDiffTR = angleDiff; bestTR = cand; }}
        if(cand.dy > 0.1 * tl.rect.height){ let angleDiff = Math.abs(cand.angle - Math.PI/2); if(angleDiff < minAngleDiffBL){ minAngleDiffBL = angleDiff; bestBL = cand; }}
    }

    if (bestTR && bestBL && bestTR !== bestBL) { tr = bestTR; bl = bestBL; }
    else {
        if(remaining.length >=2 && remaining[0] !== remaining[remaining.length-1]) { tr = remaining[0]; bl = remaining[remaining.length-1]; }
        else if (remaining.length >=1 && tl !== remaining[0]) { tr = remaining[0]; const others = markerObjects.filter(m => m !== tl && m !== tr); if (others.length > 0) bl = others[0]; else { console.warn("Could not find distinct BL fallback."); return null;} }
        else { console.warn("Fallback for TR/BL selection failed."); return null; }
    }

    if (!tr || !bl || tr === bl) { console.warn("selectCornerMarkers_JS: Failed to select distinct TR and BL markers with final checks."); return null; }

    const getCornerPt = (markerObj, type) => { // Uses actual contour points for precision
        if (!markerObj.points || markerObj.points.length !== 4) { // Fallback to rect if points are bad
            const r = markerObj.rect; if (type === 'TL') return {x: r.x, y: r.y}; if (type === 'TR') return {x: r.x + r.width, y: r.y}; if (type === 'BL') return {x: r.x, y: r.y + r.height}; return markerObj.center;
        }
        let pts = [...markerObj.points]; // Sort points for specific corner
        if (type === 'TL') pts.sort((a,b) => (a.x + a.y) - (b.x + b.y)); // Smallest sum x+y
        else if (type === 'TR') pts.sort((a,b) => (b.x - a.y) - (a.x - b.y)); // Largest x-y
        else if (type === 'BL') pts.sort((a,b) => (a.x - b.y) - (b.x - a.y)); // Smallest x-y
        return pts[0];
    };
    return { TL: getCornerPt(tl, 'TL'), TR: getCornerPt(tr, 'TR'), BL: getCornerPt(bl, 'BL') };
}

/**
 * Performs perspective transformation on an image based on three identified corner marker points.
 * @param {cv.Mat} srcImageMat - The source image (color).
 * @param {Object} cornerMarkerPoints - An object {TL: {x,y}, TR: {x,y}, BL: {x,y}} defining the source corners.
 * @returns {cv.Mat|null} The warped image Mat (color), or null on failure. Caller must delete the returned Mat.
 */
function perspectiveTransform_JS(srcImageMat, cornerMarkerPoints) {
    const { TL, TR, BL } = cornerMarkerPoints;
    const BR_est_x = TR.x + BL.x - TL.x; const BR_est_y = TR.y + BL.y - TL.y;
    if ([TL.x, TL.y, TR.x, TR.y, BL.x, BL.y, BR_est_x, BR_est_y].some(pt => isNaN(pt) || !isFinite(pt))) { console.error("Perspective Transform: Invalid coordinates."); return null;}
    const srcPtsData = [TL.x, TL.y, TR.x, TR.y, BR_est_x, BR_est_y, BL.x, BL.y];
    const dstPtsData = [0,0, WARPED_IMAGE_SIZE-1,0, WARPED_IMAGE_SIZE-1,WARPED_IMAGE_SIZE-1, 0,WARPED_IMAGE_SIZE-1];
    let srcPts = null, dstPts = null, M = null, warpedMat = new cv.Mat();
    try { srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, srcPtsData); dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, dstPtsData);
        if (srcPts.empty() || dstPts.empty()) throw new Error("Failed to create point matrices for perspective transform.");
        M = cv.getPerspectiveTransform(srcPts, dstPts); if (M.empty()) throw new Error("getPerspectiveTransform returned empty matrix.");
        cv.warpPerspective(srcImageMat, warpedMat, M, new cv.Size(WARPED_IMAGE_SIZE, WARPED_IMAGE_SIZE), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
    } catch(e) { console.error("Perspective transform error:", e, e.stack); if(warpedMat && !warpedMat.isDeleted()) warpedMat.delete(); return null; }
    finally { if(srcPts && !srcPts.isDeleted()) srcPts.delete(); if(dstPts && !dstPts.isDeleted()) dstPts.delete(); if(M && !M.isDeleted()) M.delete(); }
    if (warpedMat.empty()) { console.error("Warped image is empty after transform."); if(!warpedMat.isDeleted()) warpedMat.delete(); return null; }
    return warpedMat;
}

// UI Helper functions
/** Updates the decode result display area. @param {String} message - HTML string to display. */
function updateDecodeResultUI(message) { const resultDiv = document.getElementById('decode-result'); if (resultDiv) resultDiv.innerHTML = `<p>${message}</p>`; }
/** Shows a spinner element. @param {String} spinnerId - ID of the spinner element. */
function showSpinnerUI(spinnerId) { const spinner = document.getElementById(spinnerId); if (spinner) spinner.style.display = 'block'; }
/** Hides a spinner element. @param {String} spinnerId - ID of the spinner element. */
function hideSpinnerUI(spinnerId) { const spinner = document.getElementById(spinnerId); if (spinner) spinner.style.display = 'none'; }
