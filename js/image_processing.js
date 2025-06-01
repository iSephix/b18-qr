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

// OpenCV-dependent identifySymbolJs function has been removed.

/**
 * Main function to decode a visual code from an image element.
 * Orchestrates loading the image, finding markers, perspective correction,
 * grid iteration, symbol identification, and calling the Reed-Solomon decoding pipeline.
 * @param {HTMLImageElement} imageElement - The HTML image element containing the visual code.
 * @returns {Promise<String|null>} A promise that resolves to the decoded string, or null if decoding fails.
 */
async function decodeVisualCodeFromImage(imageElement) {
    console.info('Starting JSFeat image processing pipeline...');
    // updateDecodeResultUI('Processing image with JSFeat...'); // Intentionally commented, UI updates handled by caller or later
    // updateDecodeResultUI('Processing image with JSFeat...'); // Temporarily commented

    const width = imageElement.naturalWidth;
    const height = imageElement.naturalHeight;

    if (width === 0 || height === 0) {
        console.error('Image has zero width or height.');
        // updateDecodeResultUI('Error: Image has zero dimensions.'); // Temporarily commented
        return null;
    }

    // Create a temporary canvas to get ImageData
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    let ctx = tempCanvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, width, height);
    let imageData = null;
    try {
        imageData = ctx.getImageData(0, 0, width, height);
    } catch (e) {
        console.error('Error getting ImageData (possibly tainted canvas):', e);
        // updateDecodeResultUI('Error: Could not get image data. If loading a remote image, ensure it has CORS headers.'); // Temporarily commented
        return null;
    }

    // Initialize JSFeat matrices
    // jsfeat.matrix_t constructor: (columns, rows, data_type, data_buffer = undefined)
    let gray_img = new jsfeat.matrix_t(width, height, jsfeat.U8_t | jsfeat.C1_t);
    let img_u8_smooth = new jsfeat.matrix_t(width, height, jsfeat.U8_t | jsfeat.C1_t);

    // Convert to grayscale
    // jsfeat.imgproc.grayscale(source_data, width, height, dest_matrix, code = 0)
    // Assuming RGBA input from canvas ImageData
    jsfeat.imgproc.grayscale(imageData.data, width, height, gray_img, jsfeat.COLOR_RGBA2GRAY);
    console.info('Image converted to grayscale using JSFeat.');

    // Apply Gaussian blur
    // jsfeat.imgproc.gaussian_blur(source_matrix, dest_matrix, kernel_size, sigma = 0)
    // Kernel size should be odd and positive. Let's use 5x5 with sigma 0 (auto-calculated)
    const kernel_size = 5; // Example kernel size
    const sigma = 0;       // Auto-calculate sigma from kernel_size
    jsfeat.imgproc.gaussian_blur(gray_img, img_u8_smooth, kernel_size, sigma);
    console.info('Grayscale image blurred using JSFeat.');

    // For now, we'll log the dimensions of the processed image.
    // The rest of the OpenCV-dependent code will remain until replaced in later steps.
    console.log('JSFeat processed image (smooth):', img_u8_smooth.cols, 'x', img_u8_smooth.rows);

    // --- Start of JSFeat Binarization ---
    const otsu_thresh_val = otsu_threshold_jsfeat(img_u8_smooth);
    console.info('Calculated Otsu threshold:', otsu_thresh_val);

    let binary_img = new jsfeat.matrix_t(width, height, jsfeat.U8_t | jsfeat.C1_t);
    // Assuming markers are dark (low pixel values) on a light background.
    // THRESH_BINARY_INV makes pixels < thresh white (255).
    // So, if src_matrix.data[i] (marker pixel) <= threshold, it should be 255. This is invert = true.
    apply_threshold_jsfeat(img_u8_smooth, binary_img, otsu_thresh_val, true); // true for inverted behavior
    console.info('Image binarized using Otsu threshold with JSFeat. Dimensions:', binary_img.cols, 'x', binary_img.rows);
    // --- End of JSFeat Binarization ---

    const marker_candidates = findMarkerCandidates_jsfeat(binary_img);
    console.info(`Found ${marker_candidates.length} initial marker candidates using JSFeat CCL.`);
    // marker_candidates.forEach(candidate => { // Verbose logging, can be enabled for debug
    //     console.log(`  Candidate ${candidate.label}: x=${candidate.x}, y=${candidate.y}, w=${candidate.width}, h=${candidate.height}, area=${candidate.area}, aspect_ratio=${(candidate.width/candidate.height).toFixed(2)}`);
    // });

    const markerPatternForVerification = [ // Simpler, color-only, for JSFeat path
        ['black', 'black', 'black'],
        ['black', 'white', 'black'],
        ['black', 'black', 'black']
    ];

    console.time("verifyAndFilterMarkers_jsfeat");
    const verified_markers_raw = verifyAndFilterMarkers_jsfeat(marker_candidates, img_u8_smooth, markerPatternForVerification);
    console.timeEnd("verifyAndFilterMarkers_jsfeat");
    console.info(`Found ${verified_markers_raw.length} markers after pattern verification.`);

    console.time("nonMaxSuppression_jsfeat");
    const final_jsfeat_markers = nonMaxSuppression_jsfeat(verified_markers_raw, 0.3);
    console.timeEnd("nonMaxSuppression_jsfeat");
    console.info(`Found ${final_jsfeat_markers.length} markers after NMS.`);
    // final_jsfeat_markers.forEach(marker => { // Verbose
    //     console.log(`  Final JSFeat Marker: x=${marker.x}, y=${marker.y}, w=${marker.width}, h=${marker.height}, area=${marker.area}`);
    // });

    // --- JSFeat Global Perspective Transform ---
    console.time("selectCornerMarkers_jsfeat");
    const selected_jsfeat_code_corners = selectCornerMarkers_jsfeat(final_jsfeat_markers, width, height);
    console.timeEnd("selectCornerMarkers_jsfeat");

    let warped_img_jsfeat = null; // Will hold the final warped image from JSFeat path

    if (selected_jsfeat_code_corners) {
        console.info("Selected JSFeat code corners:", selected_jsfeat_code_corners);
        const { TL_code_corner: TL_c, TR_code_corner: TR_c, BL_code_corner: BL_c } = selected_jsfeat_code_corners;

        if (TL_c && TR_c && BL_c) {
            const BR_c_est = { x: TR_c.x + BL_c.x - TL_c.x, y: TR_c.y + BL_c.y - TL_c.y };
            const src_transform_pts_jsfeat = [TL_c, TR_c, BR_c_est, BL_c];

            const wis = WARPED_IMAGE_SIZE;
            const dst_transform_pts_jsfeat = [{x:0,y:0}, {x:wis-1,y:0}, {x:wis-1,y:wis-1}, {x:0,y:wis-1}];

            const flattened_src_jsfeat = [];
            src_transform_pts_jsfeat.forEach(p => { flattened_src_jsfeat.push(p.x); flattened_src_jsfeat.push(p.y); });
            const flattened_dst_jsfeat = [];
            dst_transform_pts_jsfeat.forEach(p => { flattened_dst_jsfeat.push(p.x); flattened_dst_jsfeat.push(p.y); });

            let H_global_matrix = new jsfeat.matrix_t(3, 3, jsfeat.F32_t | jsfeat.C1_t);
            let H_global_inv_matrix = new jsfeat.matrix_t(3, 3, jsfeat.F32_t | jsfeat.C1_t);

            console.time("jsfeat_perspective_transform_calc");
            jsfeat.math.perspective_4point_transform(H_global_matrix, flattened_src_jsfeat, flattened_dst_jsfeat);
            jsfeat.matmath.invert_3x3(H_global_matrix, H_global_inv_matrix); // Invert for warp_perspective
            console.timeEnd("jsfeat_perspective_transform_calc");

            warped_img_jsfeat = new jsfeat.matrix_t(wis, wis, jsfeat.U8_t | jsfeat.C1_t);

            console.time("jsfeat_warp_perspective");
            jsfeat.imgproc.warp_perspective(img_u8_smooth, warped_img_jsfeat, H_global_inv_matrix, 0);
            console.timeEnd("jsfeat_warp_perspective");

            console.info('Global perspective transform with JSFeat complete. Warped image dimensions:', warped_img_jsfeat.cols, 'x', warped_img_jsfeat.rows);
            // TODO: Next step is to use warped_img_jsfeat for grid symbol identification
        } else {
            console.error("JSFeat corner selection failed to return all three distinct corner points.");
             // updateDecodeResultUI("Decoding failed: JSFeat could not determine code corners."); // Temporarily commented
            return null; // Abort if corners aren't good.
        }
    } else {
        console.error("JSFeat marker corner selection failed.");
        // updateDecodeResultUI("Decoding failed: JSFeat marker corner selection failed."); // Temporarily commented
        return null; // Abort if corners not selected
    }
    // --- End JSFeat Global Perspective Transform ---

    let decodedString = null; // Declare decodedString here to be accessible by both paths

    // If JSFeat path produced a warped image, proceed with JSFeat symbol identification
    if (warped_img_jsfeat) {
        console.info("Attempting to decode symbols from JSFeat warped image...");

        for (const gridDim of POSSIBLE_GRID_DIMS) {
            console.info(`JSFeat Path: Attempting decode with grid dimension: ${gridDim}x${gridDim}`);
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

                    const insetRatio = 0.15;
                    const insetPxWidth = Math.floor(cellWidth * insetRatio);
                    const insetPxHeight = Math.floor(cellHeight * insetRatio);

                    const roiX = xStart + insetPxWidth;
                    const roiY = yStart + insetPxHeight;
                    const roiW = Math.max(1, cellWidth - 2 * insetPxWidth);
                    const roiH = Math.max(1, cellHeight - 2 * insetPxHeight);

                    if (roiX < 0 || roiY < 0 || roiX + roiW > WARPED_IMAGE_SIZE || roiY + roiH > WARPED_IMAGE_SIZE || roiW <=0 || roiH <=0) {
                        console.warn(`Cell ROI out of bounds: x=${roiX},y=${roiY},w=${roiW},h=${roiH} for warped_img_jsfeat (${warped_img_jsfeat.cols}x${warped_img_jsfeat.rows})`);
                        currentSymbolsFlatIndices.push(CUST_CODEC_SYMBOL_LIST.length);
                        continue;
                    }

                    let cell_matrix = new jsfeat.matrix_t(roiW, roiH, jsfeat.U8_t | jsfeat.C1_t);
                    for (let r_copy = 0; r_copy < roiH; ++r_copy) {
                        for (let c_copy = 0; c_copy < roiW; ++c_copy) {
                            const src_idx = (roiY + r_copy) * warped_img_jsfeat.cols + (roiX + c_copy);
                            const dst_idx = r_copy * roiW + c_copy;
                            if (src_idx < warped_img_jsfeat.data.length && dst_idx < cell_matrix.data.length) {
                                cell_matrix.data[dst_idx] = warped_img_jsfeat.data[src_idx];
                            }
                        }
                    }

                    const identifiedSymbol = identifySymbol_jsfeat(cell_matrix, CUST_CODEC_COLOR_RGB_MAP, CUST_CODEC_SYMBOL_LIST, CUST_CODEC_SPECIAL_SYMBOL);

                    let symbolIndex = CUST_CODEC_SYMBOL_LIST.findIndex(s => s[0] === identifiedSymbol[0] && s[1] === identifiedSymbol[1]);
                    if (symbolIndex === -1) {
                        symbolIndex = CUST_CODEC_SYMBOL_LIST.length;
                    }
                    currentSymbolsFlatIndices.push(symbolIndex);
                }
            }

            if (currentSymbolsFlatIndices.length === 0) {
                console.warn(`JSFeat Path: Grid ${gridDim}: No symbols extracted.`);
                continue;
            }

            if (typeof fullDecodePipelineFromIndices === 'function') {
                const resultString = await fullDecodePipelineFromIndices(currentSymbolsFlatIndices);
                if (resultString !== null && resultString !== undefined) {
                    decodedString = resultString;
                    console.info(`JSFeat Path: Successfully decoded with grid ${gridDim}x${gridDim}!`);
                    break;
                } else {
                    console.warn(`JSFeat Path: Grid ${gridDim} decode attempt failed (pipeline returned null).`);
                }
            } else {
                console.error('fullDecodePipelineFromIndices function is not available. JSFeat path cannot complete decoding.');
                decodedString = null;
                break;
            }
        }

        hideSpinnerUI('decode-spinner');
        if (decodedString) {
             console.info("Decoding successful using JSFeat pipeline. Skipping OpenCV fallback.");
             return decodedString;
        } else {
            console.warn("JSFeat pipeline could not decode the image. Will proceed to OpenCV fallback if enabled.");
        }
    } else {
         console.warn("JSFeat path did not produce a warped image. Skipping JSFeat symbol identification.");
    }
    // --- End of JSFeat Symbol Identification and Decoding Attempt ---

    // --- Original OpenCV Path (Fallback / To be removed later if JSFeat is reliable) ---
    if (!decodedString) { // Only run OpenCV path if JSFeat path failed to decode
        console.info("Proceeding with OpenCV path as JSFeat path did not yield a result.");
        showSpinnerUI('decode-spinner');

        let src = null, gray = null, binary = null, warpedImg = null; // OpenCV Mats
        try {
            src = cv.imread(imageElement);
            gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
            binary = new cv.Mat(); cv.adaptiveThreshold(gray, binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);

            const markerObjects = findMarkers_JS(binary, markerPatternJsDefinition);
            if (!markerObjects || markerObjects.length < 3) {
                console.warn(`OpenCV Path: Decoding failed: Could not find enough markers. Found ${markerObjects ? markerObjects.length : 0}. Need 3.`);
                if (src && !src.isDeleted()) src.delete(); if (gray && !gray.isDeleted()) gray.delete(); if (binary && !binary.isDeleted()) binary.delete();
                hideSpinnerUI('decode-spinner'); return null;
            }

            const cornerMarkerPoints = selectCornerMarkers_JS(markerObjects, src.cols, src.rows);
            if (!cornerMarkerPoints) {
               console.warn('OpenCV Path: Decoding failed: Could not select corner markers.');
               if (src && !src.isDeleted()) src.delete(); if (gray && !gray.isDeleted()) gray.delete(); if (binary && !binary.isDeleted()) binary.delete();
               hideSpinnerUI('decode-spinner'); return null;
            }

            warpedImg = perspectiveTransform_JS(src, cornerMarkerPoints);
            if (!warpedImg || warpedImg.empty()) {
                console.warn('OpenCV Path: Decoding failed: Perspective transform failed or resulted in empty image.');
                if (src && !src.isDeleted()) src.delete(); if (gray && !gray.isDeleted()) gray.delete(); if (binary && !binary.isDeleted()) binary.delete();
                hideSpinnerUI('decode-spinner'); return null;
            }
            console.info('OpenCV Path: Image warped. Identifying symbols from grid...');

            for (const gridDim of POSSIBLE_GRID_DIMS) {
                console.info(`OpenCV Path: Attempting decode with grid dimension: ${gridDim}x${gridDim}`);
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

                        let cellMat_rgba_cv = null; let identifiedSymbol_cv = null;
                        try {
                            cellMat_rgba_cv = warpedImg.roi(new cv.Rect(roiX, roiY, roiW, roiH));
                            identifiedSymbol_cv = cellMat_rgba_cv.empty() ? specialSymbolJs : identifySymbolJs(cellMat_rgba_cv);
                        } catch(e) { console.error("Error in OpenCV cell ROI processing:", e.stack); identifiedSymbol_cv = specialSymbolJs; }
                        finally { if (cellMat_rgba_cv && !cellMat_rgba_cv.isDeleted()) cellMat_rgba_cv.delete(); }

                        let symbolIndex_cv = symbolListJs.findIndex(s => s[0] === identifiedSymbol_cv[0] && s[1] === identifiedSymbol_cv[1]);
                        if (symbolIndex_cv === -1) symbolIndex_cv = symbolListJs.length;
                        currentSymbolsFlatIndices.push(symbolIndex_cv);
                    }
                }

                if (currentSymbolsFlatIndices.length === 0) { console.warn(`OpenCV Path: Grid ${gridDim}: No symbols extracted.`); continue; }

                if (typeof fullDecodePipelineFromIndices === 'function') {
                    const resultString = await fullDecodePipelineFromIndices(currentSymbolsFlatIndices);
                    if (resultString !== null && resultString !== undefined) {
                        decodedString = resultString;
                        console.info(`OpenCV Path: Successfully decoded with grid ${gridDim}x${gridDim}!`);
                        break;
                    } else {
                        console.warn(`OpenCV Path: Grid ${gridDim} decode attempt failed.`);
                    }
                } else {
                    console.error('OpenCV Path: fullDecodePipelineFromIndices function is not available.');
                    break;
                }
            }

            if (!decodedString) {
                console.warn("OpenCV Path: Failed to decode with any attempted grid dimension.");
            }

        } catch (error) {
            console.error('Error during OpenCV decoding pipeline:', error, error.stack);
        } finally {
            if (src && !src.isDeleted()) src.delete();
            if (gray && !gray.isDeleted()) gray.delete();
            if (binary && !binary.isDeleted()) binary.delete();
            if (warpedImg && !warpedImg.isDeleted()) warpedImg.delete();
            hideSpinnerUI('decode-spinner');
        }
    }
    return decodedString;
}


// --- JSFeat Symbol Identification ---
let precalculatedLuminances = null;

function getGrayscaleLuminance(r, g, b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

function precalculateColorLuminances(color_map) {
    // Assumes color_map is like CUST_CODEC_COLOR_RGB_MAP
    if (!precalculatedLuminances) {
        precalculatedLuminances = {};
        for (const colorName in color_map) {
            if (colorName === 'background') continue;
            const color = color_map[colorName];
            precalculatedLuminances[colorName] = getGrayscaleLuminance(color.r, color.g, color.b);
        }
    }
    return precalculatedLuminances;
}

/**
 * Identifies the visual symbol (color and shape) from a cell matrix using JSFeat.
 * @param {jsfeat.matrix_t} cell_matrix - Grayscale image of the cell.
 * @param {Object} color_map - e.g., CUST_CODEC_COLOR_RGB_MAP.
 * @param {Array} symbol_list - e.g., CUST_CODEC_SYMBOL_LIST.
 * @param {Array} special_symbol - e.g., CUST_CODEC_SPECIAL_SYMBOL.
 * @returns {Array<String>} An array containing [colorName, shapeName].
 */
function identifySymbol_jsfeat(cell_matrix, color_map, symbol_list, special_symbol) {
    if (!cell_matrix || cell_matrix.rows === 0 || cell_matrix.cols === 0 || !cell_matrix.data || cell_matrix.data.length === 0) {
        return [...special_symbol];
    }

    // Ensure CUST_CODEC constants are available or use local fallbacks for robustness
    const current_color_map = color_map || window.CUST_CODEC_COLOR_RGB_MAP || colorRgbMapJs; // Fallback to local if global/param not set
    const current_special_symbol = special_symbol || window.CUST_CODEC_SPECIAL_SYMBOL || specialSymbolJs;

    const luminances = precalculateColorLuminances(current_color_map);
    let detectedColorName = 'gray';

    let sum_intensity = 0;
    let num_pixels_sampled = 0;
    const borderSkipX = Math.floor(cell_matrix.cols * 0.2);
    const borderSkipY = Math.floor(cell_matrix.rows * 0.2);

    for (let r = borderSkipY; r < cell_matrix.rows - borderSkipY; ++r) {
        for (let c = borderSkipX; c < cell_matrix.cols - borderSkipX; ++c) {
            sum_intensity += cell_matrix.data[r * cell_matrix.cols + c];
            num_pixels_sampled++;
        }
    }
    if (num_pixels_sampled === 0) {
        for(let i=0; i < cell_matrix.data.length; ++i) sum_intensity += cell_matrix.data[i];
        num_pixels_sampled = cell_matrix.data.length;
    }
    if (num_pixels_sampled === 0) return [...current_special_symbol];

    const avg_cell_intensity = sum_intensity / num_pixels_sampled;

    let min_lum_diff = Infinity;
    let bestMatchColor = 'gray'; // Start with gray
    for (const colorName in luminances) { // Iterate over precalculated luminances
        const diff = Math.abs(avg_cell_intensity - luminances[colorName]);
        if (diff < min_lum_diff) {
            min_lum_diff = diff;
            bestMatchColor = colorName;
        }
    }

    // Tunable threshold for color matching. If no color is "close enough", consider it an error or 'gray'.
    if (min_lum_diff <= 50) {
        detectedColorName = bestMatchColor;
    } else {
        return [...current_special_symbol];
    }
    // If the best match is 'gray' from the color map (and it's a valid symbol color), it's fine.
    // If no color was close enough, we already returned special_symbol.

    let detectedShape = 'square'; // Default shape

    let binary_cell_matrix = new jsfeat.matrix_t(cell_matrix.cols, cell_matrix.rows, jsfeat.U8_t | jsfeat.C1_t);
    const cell_otsu_thresh = otsu_threshold_jsfeat(cell_matrix);
    apply_threshold_jsfeat(cell_matrix, binary_cell_matrix, cell_otsu_thresh, true);

    let fgPixels = 0;
    for(let i=0; i < binary_cell_matrix.data.length; ++i) if(binary_cell_matrix.data[i] === 255) fgPixels++;

    if (fgPixels / (binary_cell_matrix.data.length || 1) < 0.05) { // Reduced threshold for sparse cells
        return [detectedColorName, 'square'];
    }

    let min_fx = cell_matrix.cols, min_fy = cell_matrix.rows, max_fx = -1, max_fy = -1;
    let symbol_area_in_cell = 0;
    for (let r = 0; r < cell_matrix.rows; ++r) {
        for (let c = 0; c < cell_matrix.cols; ++c) {
            if (binary_cell_matrix.data[r * cell_matrix.cols + c] === 255) {
                symbol_area_in_cell++;
                min_fx = Math.min(min_fx, c);
                min_fy = Math.min(min_fy, r);
                max_fx = Math.max(max_fx, c);
                max_fy = Math.max(max_fy, r);
            }
        }
    }

    if (symbol_area_in_cell < (cell_matrix.rows * cell_matrix.cols * 0.03) || max_fx < min_fx || max_fy < min_fy ) { // Reduced min area
        return [detectedColorName, 'square'];
    }

    const bw = max_fx - min_fx + 1;
    const bh = max_fy - min_fy + 1;

    if (bw <=0 || bh <=0) return [detectedColorName, 'square'];

    const fill_ratio = symbol_area_in_cell / (bw * bh);
    const aspect_ratio = bw / bh;

    // Square: aspect ratio near 1, high fill ratio
    if (aspect_ratio > 0.7 && aspect_ratio < 1.4 && fill_ratio > 0.7) {
        detectedShape = 'square';
    }
    // Circle: aspect ratio near 1, moderate fill ratio (pi/4 ~ 0.785, but allow wider range for imperfect circles)
    else if (aspect_ratio > 0.65 && aspect_ratio < 1.5 && fill_ratio > 0.5 && fill_ratio < 0.85) {
         detectedShape = 'circle';
    }
    // Triangle: More varied aspect ratios, typically lower fill ratio than squares/circles
    // This is the hardest to distinguish with simple bounding box metrics.
    else if (fill_ratio > 0.3 && fill_ratio < 0.7) {
        // Could try to distinguish from elongated rectangles if needed, but this is very basic
        detectedShape = 'triangle';
    }
    // Default to square if heuristics are ambiguous
    else {
        detectedShape = 'square';
    }
    return [detectedColorName, detectedShape];
}


// --- JSFeat Corner Marker Selection ---
/**
 * Selects the top-left (TL), top-right (TR), and bottom-left (BL) markers from a list of JSFeat-detected markers.
 * @param {Array<Object>} jsfeat_markers - Array of marker objects, each with x, y, width, height, area, and points (bounding box corners).
 * @param {Number} imageWidth - Width of the original image.
 * @param {Number} imageHeight - Height of the original image.
 * @returns {Object|null} An object { TL_code_corner, TR_code_corner, BL_code_corner } for the corners of the overall code area, or null.
 */
function selectCornerMarkers_jsfeat(jsfeat_markers, imageWidth, imageHeight) {
    if (!jsfeat_markers || jsfeat_markers.length < 3) {
        console.warn("selectCornerMarkers_jsfeat: Not enough JSFeat markers for corner selection:", jsfeat_markers ? jsfeat_markers.length : 0);
        return null;
    }

    // Create a working copy and add center points
    const markers = jsfeat_markers.map(m => ({
        ...m, // Includes x, y, width, height, area, points (bbox corners)
        center: { x: m.x + m.width / 2, y: m.y + m.height / 2 }
    }));

    // Sort by sum of center coordinates to find Top-Left (TL) candidate
    markers.sort((a, b) => (a.center.x + a.center.y) - (b.center.x + b.center.y));
    const tl_marker = markers[0];

    // Calculate angles of other markers relative to TL's center
    const remaining_markers = markers.slice(1).map(m => {
        const dx = m.center.x - tl_marker.center.x;
        const dy = m.center.y - tl_marker.center.y;
        // Normalize angle to be 0 to 2PI
        let angle = Math.atan2(dy, dx);
        if (angle < 0) angle += 2 * Math.PI;
        return { ...m, angle: angle, dist_sq: dx*dx + dy*dy }; // Store distance for tie-breaking if needed
    });

    if (remaining_markers.length < 2) {
        console.warn("selectCornerMarkers_jsfeat: Not enough remaining markers after TL selection.");
        return null;
    }

    // Sort remaining by angle
    remaining_markers.sort((a, b) => a.angle - b.angle);

    let tr_marker = null;
    let bl_marker = null;

    // Attempt to find TR and BL based on angles and relative positions
    // TR: angle closest to 0 (or smallest positive).
    // BL: angle closest to PI/2.
    let best_tr_candidate = null;
    let min_tr_angle_diff = Math.PI / 4; // Expect TR to be within +/- 45 deg of positive x-axis from TL

    let best_bl_candidate = null;
    let min_bl_angle_diff = Math.PI / 4; // Expect BL to be within +/- 45 deg of positive y-axis from TL (angle PI/2)

    for (const cand of remaining_markers) {
        // TR candidate: should be to the right of TL
        if (cand.center.x > tl_marker.center.x) {
            let angle_diff_from_0 = Math.min(Math.abs(cand.angle), Math.abs(cand.angle - 2 * Math.PI)); // Handle angles near 0 or 2PI
            if (angle_diff_from_0 < min_tr_angle_diff) {
                 // Basic check: TR y should be somewhat aligned with TL y
                if (Math.abs(cand.center.y - tl_marker.center.y) < cand.height + tl_marker.height) { // Allow some y-offset
                    min_tr_angle_diff = angle_diff_from_0;
                    best_tr_candidate = cand;
                }
            }
        }

        // BL candidate: should be below TL
        if (cand.center.y > tl_marker.center.y) {
            let angle_diff_from_90 = Math.abs(cand.angle - Math.PI / 2);
            if (angle_diff_from_90 < min_bl_angle_diff) {
                // Basic check: BL x should be somewhat aligned with TL x
                if (Math.abs(cand.center.x - tl_marker.center.x) < cand.width + tl_marker.width) { // Allow some x-offset
                    min_bl_angle_diff = angle_diff_from_90;
                    best_bl_candidate = cand;
                }
            }
        }
    }

    tr_marker = best_tr_candidate;
    bl_marker = best_bl_candidate;

    // Fallback if specific candidates not found (e.g. due to rotation or few markers)
    if (!tr_marker || !bl_marker || tr_marker === bl_marker) {
        console.warn("Initial TR/BL selection logic failed or found same marker. Using simpler angle sort fallback.");
        // TR is likely the one with smallest angle (after TL)
        // BL is likely the one with largest angle (or angle closest to PI/2 if many markers)
        if (remaining_markers.length > 0) {
            if (!tr_marker) tr_marker = remaining_markers[0]; // Smallest angle
        }
        if (remaining_markers.length > 1) {
            // Try to find a BL distinct from TR
            let potential_bl = remaining_markers.find(m => m !== tr_marker && Math.abs(m.angle - Math.PI/2) < Math.PI/3 ); // angle within 60deg of PI/2
            if(!potential_bl) potential_bl = remaining_markers[remaining_markers.length-1]; // largest angle as fallback
            if (!bl_marker || bl_marker === tr_marker) bl_marker = (potential_bl !== tr_marker) ? potential_bl : null;
        }
    }


    if (!tl_marker || !tr_marker || !bl_marker || tl_marker === tr_marker || tl_marker === bl_marker || tr_marker === bl_marker) {
        console.warn("selectCornerMarkers_jsfeat: Failed to select three distinct corner markers. Aborting selection.");
        return null;
    }

    // Ensure points are ordered [TL, TR, BR, BL] for each marker's bounding box
    // Our current .points from analyzeBlobs_jsfeat is already in this order:
    // {x: minX, y: minY} (TL), {x: maxX, y: minY} (TR), {x: maxX, y: maxY} (BR), {x: minX, y: maxY} (BL)

    const TL_code_corner = tl_marker.points[0]; // Top-left of TL marker
    const TR_code_corner = tr_marker.points[1]; // Top-right of TR marker
    const BL_code_corner = bl_marker.points[3]; // Bottom-left of BL marker (using index 3 for BL)

    // Sanity check that selected corners are somewhat reasonable
    if (!(TL_code_corner && TR_code_corner && BL_code_corner &&
          TR_code_corner.x > TL_code_corner.x && // TR is to the right of TL
          BL_code_corner.y > TL_code_corner.y    // BL is below TL
        )) {
        console.warn("selectCornerMarkers_jsfeat: Selected code corners do not form a valid TL, TR, BL arrangement.", TL_code_corner, TR_code_corner, BL_code_corner);
        return null;
    }

    return { TL_code_corner, TR_code_corner, BL_code_corner };
}


// --- DSU Implementation ---
class DSU {
    constructor() { this.parent = {}; this.rank = {}; }
    makeSet(item) { if (!(item in this.parent)) { this.parent[item] = item; this.rank[item] = 0; } }
    find(item) {
        if (!this.parent.hasOwnProperty(item)) { this.makeSet(item); return item;} // Should not happen if makeSet is called correctly
        if (this.parent[item] === item) return item;
        return this.parent[item] = this.find(this.parent[item]);
    }
    union(item1, item2) { let root1 = this.find(item1); let root2 = this.find(item2);
        if (root1 !== root2) { if (this.rank[root1] < this.rank[root2]) [root1, root2] = [root2, root1];
            this.parent[root2] = root1; if (this.rank[root1] === this.rank[root2]) this.rank[root1]++; } }
}

/**
 * Performs Connected Components Labeling on a binary JSFeat matrix.
 * @param {jsfeat.matrix_t} binary_matrix - Input binary image (U8_C1, 255 for foreground).
 * @returns {jsfeat.matrix_t} A matrix of the same dimensions with S32_C1 type, containing component labels.
 */
function connectedComponentsLabeling_jsfeat(binary_matrix) {
    const rows = binary_matrix.rows;
    const cols = binary_matrix.cols;
    const labels_matrix = new jsfeat.matrix_t(cols, rows, jsfeat.S32_t | jsfeat.C1_t); // Int32 for labels
    const dsu = new DSU();
    let next_label = 1;

    // First Pass
    for (let r = 0; r < rows; ++r) {
        for (let c = 0; c < cols; ++c) {
            const pixel_idx = r * cols + c;
            if (binary_matrix.data[pixel_idx] === 255) { // Foreground pixel
                let neighbors = [];
                // Top neighbor
                if (r > 0 && binary_matrix.data[(r - 1) * cols + c] === 255) {
                    neighbors.push(labels_matrix.data[(r - 1) * cols + c]);
                }
                // Left neighbor
                if (c > 0 && binary_matrix.data[r * cols + (c - 1)] === 255) {
                    neighbors.push(labels_matrix.data[r * cols + (c - 1)]);
                }

                if (neighbors.length === 0) {
                    labels_matrix.data[pixel_idx] = next_label;
                    dsu.makeSet(next_label);
                    next_label++;
                } else {
                    const min_neighbor_label = Math.min(...neighbors.filter(l => l > 0));
                    labels_matrix.data[pixel_idx] = min_neighbor_label;
                    neighbors.forEach(label => {
                        if (label > 0 && label !== min_neighbor_label) {
                            dsu.union(min_neighbor_label, label);
                        }
                    });
                }
            } else {
                labels_matrix.data[pixel_idx] = 0; // Background
            }
        }
    }

    // Second Pass - Resolve labels
    for (let i = 0; i < labels_matrix.data.length; ++i) {
        if (labels_matrix.data[i] > 0) {
            labels_matrix.data[i] = dsu.find(labels_matrix.data[i]);
        }
    }
    return labels_matrix;
}

/**
 * Analyzes labeled components (blobs) from a labels matrix.
 * @param {jsfeat.matrix_t} labels_matrix - Matrix containing component labels (S32_C1).
 * @param {jsfeat.matrix_t} binary_matrix - Original binary matrix (U8_C1), used for dimensions.
 * @returns {Array<Object>} An array of blob objects with properties like label, pixels, bounds, area.
 */
function analyzeBlobs_jsfeat(labels_matrix, binary_matrix) {
    const rows = binary_matrix.rows;
    const cols = binary_matrix.cols;
    const blobs = {};

    for (let r = 0; r < rows; ++r) {
        for (let c = 0; c < cols; ++c) {
            const pixel_idx = r * cols + c;
            const label = labels_matrix.data[pixel_idx];
            if (label > 0) {
                if (!blobs[label]) {
                    blobs[label] = {
                        label: label,
                        pixels: [],
                        minX: c, minY: r,
                        maxX: c, maxY: r,
                        area: 0
                    };
                }
                // blobs[label].pixels.push({ x: c, y: r }); // Storing all pixels can be memory intensive
                blobs[label].minX = Math.min(blobs[label].minX, c);
                blobs[label].minY = Math.min(blobs[label].minY, r);
                blobs[label].maxX = Math.max(blobs[label].maxX, c);
                blobs[label].maxY = Math.max(blobs[label].maxY, r);
                blobs[label].area++;
            }
        }
    }

    const blob_array = Object.values(blobs);
    blob_array.forEach(blob => {
        blob.width = blob.maxX - blob.minX + 1;
        blob.height = blob.maxY - blob.minY + 1;
        blob.x = blob.minX;
        blob.y = blob.minY;
        // Define points for the bounding box of the blob
        blob.points = [
            {x: blob.minX, y: blob.minY},                            // Top-left
            {x: blob.maxX, y: blob.minY},                            // Top-right
            {x: blob.maxX, y: blob.maxY},                            // Bottom-right
            {x: blob.minX, y: blob.maxY}                             // Bottom-left
        ];
    });
    return blob_array;
}

/**
 * Finds marker candidates in a binary image using CCL and blob analysis.
 * @param {jsfeat.matrix_t} binary_matrix - The input binary image (U8_C1).
 * @returns {Array<Object>} An array of filtered marker candidate objects.
 */
function findMarkerCandidates_jsfeat(binary_matrix) {
    if (!binary_matrix || !binary_matrix.data || binary_matrix.data.length === 0) {
        console.warn("findMarkerCandidates_jsfeat: binary_matrix is empty or invalid.");
        return [];
    }
    console.time("CCL_jsfeat");
    const labels_matrix = connectedComponentsLabeling_jsfeat(binary_matrix);
    console.timeEnd("CCL_jsfeat");

    console.time("analyzeBlobs_jsfeat");
    const raw_blobs = analyzeBlobs_jsfeat(labels_matrix, binary_matrix);
    console.timeEnd("analyzeBlobs_jsfeat");

    // Initial conservative filters - these will likely need tuning
    const min_area = 30; // Adjusted from 50, depends on expected marker size at typical camera distance
    const max_area = (binary_matrix.cols * binary_matrix.rows) / 9; // Max 1/9th of image area
    const min_aspect_ratio = 0.4; // Adjusted from 0.5
    const max_aspect_ratio = 2.5; // Adjusted from 2.0

    const marker_candidates = raw_blobs.filter(blob => {
        const aspect_ratio = blob.width / blob.height;
        return blob.area >= min_area &&
               blob.area <= max_area &&
               aspect_ratio >= min_aspect_ratio &&
               aspect_ratio <= max_aspect_ratio;
    });
    // Sort by area, descending - largest are often better candidates or easier to verify first
    marker_candidates.sort((a,b) => b.area - a.area);
    return marker_candidates;
}


/**
 * Calculates the histogram of a JSFeat U8_C1 matrix.
 * @param {jsfeat.matrix_t} src_matrix - The source grayscale image.
 * @returns {Array<number>} An array of 256 elements representing the histogram.
 */
function calculate_histogram_jsfeat(src_matrix) {
    let hist = new Array(256).fill(0);
    for (let i = 0; i < src_matrix.data.length; ++i) {
        hist[src_matrix.data[i]]++;
    }
    return hist;
}

/**
 * Calculates the optimal threshold using Otsu's method.
 * @param {jsfeat.matrix_t} src_matrix - The source grayscale image.
 * @returns {number} The calculated Otsu threshold.
 */
function otsu_threshold_jsfeat(src_matrix) {
    const hist = calculate_histogram_jsfeat(src_matrix);
    const total_pixels = src_matrix.rows * src_matrix.cols;

    let sum = 0;
    for (let i = 0; i < 256; ++i) {
        sum += i * hist[i];
    }

    let sum_bg = 0;
    let weight_bg = 0;
    let weight_fg = 0;
    let max_variance = 0;
    let threshold = 0;

    for (let i = 0; i < 256; ++i) {
        weight_bg += hist[i];
        if (weight_bg === 0) continue;

        weight_fg = total_pixels - weight_bg;
        if (weight_fg === 0) break;

        sum_bg += i * hist[i];

        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum - sum_bg) / weight_fg;

        let variance_between = weight_bg * weight_fg * Math.pow(mean_bg - mean_fg, 2);

        if (variance_between > max_variance) {
            max_variance = variance_between;
            threshold = i;
        }
    }
    return threshold;
}

/**
 * Applies a threshold to a JSFeat U8_C1 matrix to produce a binary image.
 * @param {jsfeat.matrix_t} src_matrix - The source grayscale image.
 * @param {jsfeat.matrix_t} dest_matrix - The destination binary image.
 * @param {number} threshold - The threshold value.
 * @param {boolean} invert - If true, pixels <= threshold become 255 (white), else 0.
 */
function apply_threshold_jsfeat(src_matrix, dest_matrix, threshold, invert = false) {
    const white = 255;
    const black = 0;
    for (let i = 0; i < src_matrix.data.length; ++i) {
        if (invert) {
            dest_matrix.data[i] = src_matrix.data[i] <= threshold ? white : black;
        } else {
            dest_matrix.data[i] = src_matrix.data[i] > threshold ? white : black;
        }
    }
}

// All old OpenCV-dependent helper functions (verifyMarkerPatternCV_JS, findMarkers_JS, getCenter, selectCornerMarkers_JS, perspectiveTransform_JS)
// have been removed as part of the full JSFeat migration.

// UI Helper functions
/** Updates the decode result display area. @param {String} message - HTML string to display. */
function updateDecodeResultUI(message) {
    const resultDiv = document.getElementById('decode-result');
    if (resultDiv) {
        resultDiv.innerHTML = `<p>${message}</p>`;
    } else {
        console.warn("UI element 'decode-result' not found for message:", message);
    }
}
/** Shows a spinner element. @param {String} spinnerId - ID of the spinner element. */
function showSpinnerUI(spinnerId) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) {
        spinner.style.display = 'block';
    } else {
        console.warn("Spinner UI element not found:", spinnerId);
    }
}
/** Hides a spinner element. @param {String} spinnerId - ID of the spinner element. */
function hideSpinnerUI(spinnerId) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) {
        spinner.style.display = 'none';
    } else {
        console.warn("Spinner UI element not found:", spinnerId);
    }
}
