// js/image_processing.js

// Constants (defined in previous steps)
const colorRgbMapJs = {
    'black': { r: 0, g: 0, b: 0 }, 'white': { r: 255, g: 255, b: 255 }, 'blue': { r: 0, g: 0, b: 255 },
    'green': { r: 0, g: 255, b: 0 }, 'yellow': { r: 255, g: 255, b: 0 }, 'red': { r: 255, g: 0, b: 0 },
    'gray': { r: 128, g: 128, b: 128 }, 'background': { r: 255, g: 0, b: 255 }
};

// Define expected RGB values for marker components
const EXPECTED_MARKER_BLACK_RGB = { r: 0, g: 0, b: 0 };
const EXPECTED_MARKER_WHITE_RGB = { r: 255, g: 255, b: 255 };

// Define a threshold for color matching.
// This is the maximum sum of absolute differences between R, G, and B channels
// for a color to be considered 'close enough' to a target color.
// E.g., if target is black (0,0,0), a pixel (10,20,30) has distance 10+20+30=60.
// If COLOR_DISTANCE_THRESHOLD_MARKER is 150, this pixel would be considered 'black-ish'.
const COLOR_DISTANCE_THRESHOLD_MARKER = 150; // Tune this value as needed

/**
 * Calculates the Manhattan distance between two RGB colors.
 * @param {{r:number, g:number, b:number}} rgb1 - The first RGB color.
 * @param {{r:number, g:number, b:number}} rgb2 - The second RGB color.
 * @returns {number} The sum of absolute differences of R, G, B channels.
 */
function calculateColorDistance(rgb1, rgb2) {
    if (!rgb1 || !rgb2) return Infinity; // Should not happen with valid inputs
    return Math.abs(rgb1.r - rgb2.r) + Math.abs(rgb1.g - rgb2.g) + Math.abs(rgb1.b - rgb2.b);
}

/**
 * Finds marker candidates in a color image by identifying regions matching a target color (e.g., black).
 * @param {ImageData} imageData - The raw image data from a canvas (contains data, width, height).
 * @param {{r:number, g:number, b:number}} target_marker_color_rgb - The RGB color to look for (e.g., EXPECTED_MARKER_BLACK_RGB).
 * @param {number} color_match_threshold - The tolerance for color matching (e.g., COLOR_DISTANCE_THRESHOLD_MARKER).
 * @returns {Array<Object>} An array of blob objects representing candidate regions.
 */
function findMarkerCandidates_color(imageData, target_marker_color_rgb, color_match_threshold) {
    window.logToScreen("findMarkerCandidates_color: Processing started.");
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data; // This is a Uint8ClampedArray: [R,G,B,A, R,G,B,A, ...]

    // Create a binary mask for jsfeat processing
    let binary_mask_matrix = new jsfeat.matrix_t(width, height, jsfeat.U8_t | jsfeat.C1_t);

    let current_pixel_rgb = { r: 0, g: 0, b: 0 };
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4; // Index for R channel of pixel (x,y)
            current_pixel_rgb.r = data[i];
            current_pixel_rgb.g = data[i + 1];
            current_pixel_rgb.b = data[i + 2];
            // Alpha is data[i+3]

            const distance = calculateColorDistance(current_pixel_rgb, target_marker_color_rgb);

            if (distance <= color_match_threshold) {
                binary_mask_matrix.data[y * width + x] = 255; // Foreground
            } else {
                binary_mask_matrix.data[y * width + x] = 0;   // Background
            }
        }
    }
    window.logToScreen("findMarkerCandidates_color: Binary mask created based on target color.");

    // Use existing JSFeat CCL and blob analysis functions
    window.logToScreen("findMarkerCandidates_color: Starting CCL on binary mask.");
    console.time("CCL_jsfeat_color");
    const labels_matrix = connectedComponentsLabeling_jsfeat(binary_mask_matrix);
    console.timeEnd("CCL_jsfeat_color");
    window.logToScreen("findMarkerCandidates_color: CCL complete. Labels matrix created.");

    window.logToScreen("findMarkerCandidates_color: Starting blob analysis.");
    console.time("analyzeBlobs_jsfeat_color");
    const raw_blobs = analyzeBlobs_jsfeat(labels_matrix, binary_mask_matrix);
    console.timeEnd("analyzeBlobs_jsfeat_color");
    window.logToScreen("findMarkerCandidates_color: Blob analysis complete. Found " + raw_blobs.length + " raw blobs.");

    // Apply similar filtering as in the old findMarkerCandidates_jsfeat (area, aspect ratio)
    // These might need tuning for color-based candidates.
    const min_area = 5; // Temporarily very low for diagnostics
    const max_area = (width * height) / 4; // Temporarily larger for diagnostics (1/4 of image area)
    const min_aspect_ratio = 0.1; // Temporarily very permissive
    const max_aspect_ratio = 10.0; // Temporarily very permissive

    window.logToScreen(`findMarkerCandidates_color: DIAGNOSTIC MODE - Using relaxed filters: min_area=${min_area}, max_area=${max_area.toFixed(0)}, min_aspect=${min_aspect_ratio}, max_aspect=${max_aspect_ratio}`);

    const marker_candidates = raw_blobs.filter(blob => {
        const aspect_ratio = blob.width / blob.height;
        const condition = blob.area >= min_area &&
               blob.area <= max_area &&
               aspect_ratio >= min_aspect_ratio &&
               aspect_ratio <= max_aspect_ratio;
        if (!condition && blob.area >= min_area / 2) { // Log near misses for tuning
            window.logToScreen(`findMarkerCandidates_color: Blob ${blob.label} filtered out: area=${blob.area}, asp=${aspect_ratio.toFixed(2)}`);
        }
        return condition;
    });

    // Sort by area, descending
    marker_candidates.sort((a, b) => b.area - a.area);
    window.logToScreen("findMarkerCandidates_color: Filtered down to " + marker_candidates.length + " candidates. Processing finished.");
    return marker_candidates;
}

/**
 * Verifies marker candidates by checking their 3x3 cell pattern against expected colors.
 * @param {Array<Object>} marker_candidates - Blobs from findMarkerCandidates_color (assumed to be black frames).
 * @param {ImageData} imageData - Raw image data from a canvas.
 * @param {{r:number, g:number, b:number}} expected_black - Expected RGB for black cells.
 * @param {{r:number, g:number, b:number}} expected_white - Expected RGB for white cells.
 * @param {number} color_match_threshold - Tolerance for color matching.
 * @param {Array<Array<String>>} marker_pattern_definition - e.g., [['black', 'black', 'black'], ['black', 'white', 'black'], ...]
 * @returns {Array<Object>} Filtered list of verified marker objects.
 */
function verifyAndFilterMarkers_color(
    marker_candidates,
    imageData,
    expected_black,
    expected_white,
    color_match_threshold,
    marker_pattern_definition
) {
    window.logToScreen("verifyAndFilterMarkers_color: Function entry. Received " + marker_candidates.length + " candidates.");
    const verified_markers = [];
    const image_width = imageData.width;
    const image_height = imageData.height;
    const pixel_data = imageData.data;

    for (let i = 0; i < marker_candidates.length; i++) {
        const candidate = marker_candidates[i];
        window.logToScreen(`verifyAndFilterMarkers_color: Processing candidate ${i}: x=${candidate.x}, y=${candidate.y}, w=${candidate.width}, h=${candidate.height}`);

        if (candidate.width < 3 || candidate.height < 3) {
            window.logToScreen(`verifyAndFilterMarkers_color: Candidate ${i} too small (w:${candidate.width},h:${candidate.height}), skipping.`);
            continue;
        }

        const cell_width_float = candidate.width / 3.0;
        const cell_height_float = candidate.height / 3.0;
        const derived_color_pattern = [['', '', ''], ['', '', ''], ['', '', '']];
        let possible_marker = true;

        // Optional: Verify average color of the candidate blob itself is 'black-ish'
        // This could be added for extra robustness if findMarkerCandidates_color is too loose.
        // For now, trust candidates from findMarkerCandidates_color are predominantly black.

        for (let r_cell = 0; r_cell < 3; r_cell++) { // Pattern rows
            for (let c_cell = 0; c_cell < 3; c_cell++) { // Pattern columns
                // Use inset logic for sampling robustness
                const inset_ratio = 0.20; // 20% inset
                const base_cell_x_start = candidate.x + c_cell * cell_width_float;
                const base_cell_y_start = candidate.y + r_cell * cell_height_float;
                const base_cell_x_end = candidate.x + (c_cell + 1) * cell_width_float;
                const base_cell_y_end = candidate.y + (r_cell + 1) * cell_height_float;

                const current_cell_w_float = base_cell_x_end - base_cell_x_start;
                const current_cell_h_float = base_cell_y_end - base_cell_y_start;

                const inset_w_px = Math.floor(current_cell_w_float * inset_ratio);
                const inset_h_px = Math.floor(current_cell_h_float * inset_ratio);

                const sample_x_start = Math.floor(base_cell_x_start + inset_w_px);
                const sample_y_start = Math.floor(base_cell_y_start + inset_h_px);
                const sample_x_end = Math.floor(base_cell_x_end - inset_w_px);
                const sample_y_end = Math.floor(base_cell_y_end - inset_h_px);

                window.logToScreen(`verifyAndFilterMarkers_color: Cand ${i} Cell[${r_cell}][${c_cell}] Base X:${base_cell_x_start.toFixed(1)}-${base_cell_x_end.toFixed(1)}, Y:${base_cell_y_start.toFixed(1)}-${base_cell_y_end.toFixed(1)}. Sample X:${sample_x_start}-${sample_x_end}, Y:${sample_y_start}-${sample_y_end}`);


                if (sample_x_start >= sample_x_end || sample_y_start >= sample_y_end) {
                    window.logToScreen(`verifyAndFilterMarkers_color: Candidate ${i}, cell [${r_cell}][${c_cell}] has zero/negative sample dimension after inset. Skipping candidate.`);
                    possible_marker = false;
                    break;
                }

                let sum_r = 0, sum_g = 0, sum_b = 0;
                let num_pixels_in_cell = 0;

                for (let y_px = sample_y_start; y_px < sample_y_end; y_px++) {
                    for (let x_px = sample_x_start; x_px < sample_x_end; x_px++) {
                        if (x_px >= 0 && x_px < image_width && y_px >= 0 && y_px < image_height) {
                            const pixel_idx_start = (y_px * image_width + x_px) * 4;
                            sum_r += pixel_data[pixel_idx_start];
                            sum_g += pixel_data[pixel_idx_start + 1];
                            sum_b += pixel_data[pixel_idx_start + 2];
                            num_pixels_in_cell++;
                        }
                    }
                }

                if (num_pixels_in_cell === 0) {
                    window.logToScreen(`verifyAndFilterMarkers_color: Candidate ${i}, cell [${r_cell}][${c_cell}] had no pixels in source image bounds or sample area. Skipping candidate.`);
                    possible_marker = false;
                    break;
                }

                const avg_cell_color = {
                    r: sum_r / num_pixels_in_cell,
                    g: sum_g / num_pixels_in_cell,
                    b: sum_b / num_pixels_in_cell
                };
                window.logToScreen(`verifyAndFilterMarkers_color: Cand ${i} Cell[${r_cell}][${c_cell}] AvgRGB:(${avg_cell_color.r.toFixed(0)},${avg_cell_color.g.toFixed(0)},${avg_cell_color.b.toFixed(0)}) from ${num_pixels_in_cell}px`);

                const dist_to_black = calculateColorDistance(avg_cell_color, expected_black);
                const dist_to_white = calculateColorDistance(avg_cell_color, expected_white);

                if (dist_to_black <= color_match_threshold && dist_to_black < dist_to_white) {
                    derived_color_pattern[r_cell][c_cell] = 'black';
                } else if (dist_to_white <= color_match_threshold && dist_to_white < dist_to_black) {
                    derived_color_pattern[r_cell][c_cell] = 'white';
                } else {
                     derived_color_pattern[r_cell][c_cell] = 'other'; // Neither black nor white enough
                     window.logToScreen(`verifyAndFilterMarkers_color: Cand ${i} Cell[${r_cell}][${c_cell}] color is 'other'. DistBlack:${dist_to_black.toFixed(0)}, DistWhite:${dist_to_white.toFixed(0)}`);
                }
            }
            if (!possible_marker) break;
        }

        if (!possible_marker) {
            window.logToScreen(`verifyAndFilterMarkers_color: Candidate ${i} skipped due to cell processing issues.`);
            continue;
        }

        window.logToScreen(`verifyAndFilterMarkers_color: Candidate ${i}: Derived color pattern: ${JSON.stringify(derived_color_pattern)}`);
        window.logToScreen(`verifyAndFilterMarkers_color: Candidate ${i}: Expected pattern: ${JSON.stringify(marker_pattern_definition)}`);

        let match = true;
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                if (derived_color_pattern[r][c] !== marker_pattern_definition[r][c]) {
                    match = false;
                    break;
                }
            }
            if (!match) break;
        }

        if (match) {
            window.logToScreen(`verifyAndFilterMarkers_color: Candidate ${i} MATCHED! Adding to verified list.`);
            verified_markers.push(candidate);
        } else {
            window.logToScreen(`verifyAndFilterMarkers_color: Candidate ${i} did NOT match.`);
        }
    }

    window.logToScreen("verifyAndFilterMarkers_color: Found " + verified_markers.length + " verified markers.");
    window.logToScreen("verifyAndFilterMarkers_color: Function exit.");
    return verified_markers;
}

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
window.decodeVisualCodeFromImage = async function(imageElement) {
    if (typeof window.logToScreen !== 'function') { // Fallback if logToScreen somehow isn't global
        console.warn("logToScreen function not available. Using console.log for debug messages in image_processing.js.");
        window.logToScreen = console.log; // Simple fallback
    }
    window.logToScreen("decodeVisualCodeFromImage: Processing started.");

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
    let gray_img = new jsfeat.matrix_t(width, height, jsfeat.U8_t | jsfeat.C1_t);
    let img_u8_smooth = new jsfeat.matrix_t(width, height, jsfeat.U8_t | jsfeat.C1_t);

    let decodedString = null;

    try { // Wrap main JSFeat processing in a try-catch for unexpected errors
        // --- Color-based Marker Detection ---
        window.logToScreen("decodeVisualCodeFromImage: Starting color-based marker detection.");
        const marker_candidates = findMarkerCandidates_color(imageData, EXPECTED_MARKER_BLACK_RGB, COLOR_DISTANCE_THRESHOLD_MARKER);
        window.logToScreen("decodeVisualCodeFromImage: Color-based marker candidates found: " + marker_candidates.length);

        const markerPatternForVerification = [ // This pattern is string-based and matches verifyAndFilterMarkers_color's output
            ['black', 'black', 'black'],
            ['black', 'white', 'black'],
            ['black', 'black', 'black']
        ];

        // Note: verifyAndFilterMarkers_color uses imageData directly, not a grayscale/smoothed image yet.
        const verified_markers_raw = verifyAndFilterMarkers_color(
            marker_candidates,
            imageData, // Pass original color image data
            EXPECTED_MARKER_BLACK_RGB,
            EXPECTED_MARKER_WHITE_RGB,
            COLOR_DISTANCE_THRESHOLD_MARKER,
            markerPatternForVerification
        );
        window.logToScreen("decodeVisualCodeFromImage: Verified color markers (pre-NMS): " + verified_markers_raw.length);

        const final_jsfeat_markers = nonMaxSuppression_jsfeat(verified_markers_raw, 0.3); // NMS is independent of color/grayscale
    window.logToScreen("JSFeat: Final markers (post-NMS): " + final_jsfeat_markers.length);

    if (!final_jsfeat_markers || final_jsfeat_markers.length < 3) {
        window.logToScreen("JSFeat WARNING: <3 final markers (" + (final_jsfeat_markers ? final_jsfeat_markers.length : 0) + "). Attempting corner selection.");
        updateDecodeResultUI('Error: Could not find enough distinct markers (JSFeat). Try adjusting image or lighting.');
    }

    // --- JSFeat Global Perspective Transform ---
    const selected_jsfeat_code_corners = selectCornerMarkers_jsfeat(final_jsfeat_markers, width, height);

    let warped_img_jsfeat = null;

    if (selected_jsfeat_code_corners) {
        window.logToScreen("JSFeat: Selected code corners: TL(" + selected_jsfeat_code_corners.TL_code_corner.x + ","+selected_jsfeat_code_corners.TL_code_corner.y + "), TR, BL");
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

            // --- Deferred Grayscale Conversion and Blur (For Warping and Symbol ID) ---
            window.logToScreen("JSFeat: Performing deferred grayscale conversion for warping/symbol ID.");
            // gray_img and img_u8_smooth are initialized at the top of decodeVisualCodeFromImage

            jsfeat.imgproc.grayscale(imageData.data, width, height, gray_img, jsfeat.COLOR_RGBA2GRAY);
            window.logToScreen('JSFeat: Deferred grayscale conversion complete.');

            const kernel_size = 5; // Consistent kernel size
            const sigma = 0;       // Auto-calculate sigma
            jsfeat.imgproc.gaussian_blur(gray_img, img_u8_smooth, kernel_size, sigma);
            window.logToScreen('JSFeat: Deferred Gaussian blur complete. Smooth image for warp: ' + img_u8_smooth.cols + 'x' + img_u8_smooth.rows);
            // --- End Deferred Grayscale ---

            let H_global_matrix = new jsfeat.matrix_t(3, 3, jsfeat.F32_t | jsfeat.C1_t);
            let H_global_inv_matrix = new jsfeat.matrix_t(3, 3, jsfeat.F32_t | jsfeat.C1_t);

            if (window.logToScreen) {
                window.logToScreen(`JSFeat DEBUG: Perspective Transform Src Pts: [${flattened_src_jsfeat.join(', ')}]`);
                window.logToScreen(`JSFeat DEBUG: Perspective Transform Dst Pts: [${flattened_dst_jsfeat.join(', ')}]`);
                window.logToScreen(`JSFeat DEBUG: H_global_matrix (before perspective_4point_transform): ${H_global_matrix.cols}x${H_global_matrix.rows}, type: ${H_global_matrix.type}`);
            }

            jsfeat.math.perspective_4point_transform(H_global_matrix, flattened_src_jsfeat, flattened_dst_jsfeat);

            if (window.logToScreen) {
                window.logToScreen(`JSFeat DEBUG: H_global_matrix (after perspective_4point_transform, before invert): ${H_global_matrix.cols}x${H_global_matrix.rows}, data: [${H_global_matrix.data.slice(0,9).join(', ')}]`);
                window.logToScreen(`JSFeat DEBUG: H_global_inv_matrix (before invert_3x3): ${H_global_inv_matrix.cols}x${H_global_inv_matrix.rows}, type: ${H_global_inv_matrix.type}`);
            }

            jsfeat.matmath.invert_3x3(H_global_matrix, H_global_inv_matrix);

            warped_img_jsfeat = new jsfeat.matrix_t(wis, wis, jsfeat.U8_t | jsfeat.C1_t);

            if (window.logToScreen) {
                window.logToScreen(`JSFeat DEBUG: H_global_inv_matrix (after invert_3x3, before warp): ${H_global_inv_matrix.cols}x${H_global_inv_matrix.rows}, data: [${H_global_inv_matrix.data.slice(0,9).join(', ')}]`);
                window.logToScreen(`JSFeat DEBUG: img_u8_smooth (source for warp): ${img_u8_smooth.cols}x${img_u8_smooth.rows}, type: ${img_u8_smooth.type}`); // This must now be populated correctly
                window.logToScreen(`JSFeat DEBUG: warped_img_jsfeat (dest for warp, before warp): ${warped_img_jsfeat.cols}x${warped_img_jsfeat.rows}, type: ${warped_img_jsfeat.type}`);
            }

            // Warp perspective expects a grayscale image (img_u8_smooth)
            jsfeat.imgproc.warp_perspective(img_u8_smooth, warped_img_jsfeat, H_global_inv_matrix, 0);

            if (window.logToScreen) {
                window.logToScreen(`JSFeat DEBUG: warped_img_jsfeat (after warp): ${warped_img_jsfeat.cols}x${warped_img_jsfeat.rows}, first few data: [${warped_img_jsfeat.data.slice(0,10).join(', ')}]`);
            }
            window.logToScreen("JSFeat: Global perspective warp complete.");

            if (!warped_img_jsfeat || warped_img_jsfeat.rows === 0 || warped_img_jsfeat.cols === 0) {
                window.logToScreen("JSFeat ERROR: Warped image is invalid.");
                updateDecodeResultUI('Error: Failed to create a usable rectified image (JSFeat).');
                hideSpinnerUI('decode-spinner');
                return null;
            }
        } else {
            window.logToScreen("JSFeat ERROR: Failed to select corner markers (TL, TR, or BL is null/undefined after selection).");
            updateDecodeResultUI('Error: Critical failure in corner marker processing (JSFeat).');
            hideSpinnerUI('decode-spinner');
            return null;
        }
    } else {
        // This block is executed if selectCornerMarkers_jsfeat returns null
        if (window.logToScreen) window.logToScreen("JSFeat ERROR: Failed to select corner markers. Aborting decode pipeline.");
        updateDecodeResultUI('Error: Could not determine corner markers from found patterns (JSFeat).');
        hideSpinnerUI('decode-spinner');
        return null;
    }
        // --- End JSFeat Global Perspective Transform ---

        // If JSFeat path produced a warped image, proceed with JSFeat symbol identification
        if (warped_img_jsfeat) {
            window.logToScreen("JSFeat: Attempting symbol identification from warped image.");
            showSpinnerUI('decode-spinner');

            for (const gridDim of POSSIBLE_GRID_DIMS) {
                window.logToScreen("JSFeat: Trying grid: " + gridDim + "x" + gridDim);
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
                            window.logToScreen(`JSFeat WARN: Cell ROI out of bounds: x=${roiX},y=${roiY},w=${roiW},h=${roiH}`);
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
                    window.logToScreen(`JSFeat: No symbols extracted for grid ${gridDim}.`);
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

            // This hideSpinnerUI was for the spinner shown before the gridDim loop.
            hideSpinnerUI('decode-spinner');
            if (decodedString) {
                 console.info("Decoding successful using JSFeat pipeline.");
            } else {
                // This else block is hit if the JSFeat symbol identification loop completes for all grid dimensions
                // without successfully decoding.
                console.warn('JSFeat Path: Decoding failed for all attempted grid dimensions.');
                updateDecodeResultUI('Error: Could not decode symbols with any attempted grid configuration (JSFeat).');
            }
            return decodedString; // Return whatever JSFeat path produced (null if failed).
        } else {
             // This case is if warped_img_jsfeat itself is null (e.g. corner selection/warp failed)
             // Error messages for these specific failures are now set before this point.
             console.warn("JSFeat path did not produce a warped image; cannot proceed with symbol identification.");
             // hideSpinnerUI is called in the error paths that lead here.
             return null; // Ensure we return null if warping failed.
        }
    } catch (e) {
        console.error("Unhandled error in decodeVisualCodeFromImage JSFeat pipeline:", e.stack || e);
        updateDecodeResultUI('Critical error during image processing: ' + e.message + '. Please check console.');
        hideSpinnerUI('decode-spinner'); // Ensure spinner is hidden on unexpected error
        return null; // Ensure the function returns, allowing ui.js to proceed
    }
    // The original OpenCV path has been removed.
    // If JSFeat path fails and doesn't return early, decodedString will be null here.
    // The catch block above (lines 298-303) correctly handles errors for the main JSFeat pipeline try block (started line 61).
    // This second catch block (previously lines 307-312) was orphaned and caused a syntax error
    // due to an extra '}' at its beginning which prematurely closed the main function.
    // Removing this entire invalid block.

    // This final return is for cases where the try block completes successfully (or an error was caught and handled by returning null),
    // and decodedString holds the result (which could be null if decoding steps failed logically but not via exception).
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
 * Verifies marker candidates against a predefined pattern.
 * @param {Array<Object>} marker_candidates - Objects from analyzeBlobs_jsfeat.
 * @param {jsfeat.matrix_t} source_image - Grayscale source image (e.g., img_u8_smooth).
 * @param {Array<Array<String>>} marker_pattern_definition - 2D array defining the expected pattern.
 * @param {number} otsu_threshold_val - Pre-calculated Otsu threshold for the image.
 * @returns {Array<Object>} Filtered list of verified marker objects.
 */
function verifyAndFilterMarkers_jsfeat(marker_candidates, source_image, marker_pattern_definition, otsu_threshold_val) {
    // Function logic will be implemented here
    window.logToScreen("verifyAndFilterMarkers_jsfeat: Function entry.");
    window.logToScreen("verifyAndFilterMarkers_jsfeat: Received " + marker_candidates.length + " candidates.");

    const verified_markers = [];

    for (let i = 0; i < marker_candidates.length; i++) {
        const candidate = marker_candidates[i];
        window.logToScreen(`verifyAndFilterMarkers_jsfeat: Processing candidate ${i}: x=${candidate.x}, y=${candidate.y}, w=${candidate.width}, h=${candidate.height}`);

        if (candidate.width < 3 || candidate.height < 3) {
            window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i} too small, skipping.`);
            continue;
        }

        const cell_width = candidate.width / 3.0;
        const cell_height = candidate.height / 3.0;
        const sampled_pattern_intensity = [[0,0,0],[0,0,0],[0,0,0]];
        const derived_binary_pattern = [['', '', ''], ['', '', ''], ['', '', '']];

        let possible_marker = true;

        for (let r = 0; r < 3; r++) { // Pattern rows
            for (let c = 0; c < 3; c++) { // Pattern columns
                const cell_x_start = Math.floor(candidate.x + c * cell_width);
                const cell_y_start = Math.floor(candidate.y + r * cell_height);
                const cell_x_end = Math.floor(candidate.x + (c + 1) * cell_width);
                const cell_y_end = Math.floor(candidate.y + (r + 1) * cell_height);

                let current_cell_width = cell_x_end - cell_x_start;
                let current_cell_height = cell_y_end - cell_y_start;

                if (current_cell_width <= 0 || current_cell_height <= 0) {
                     window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i}, cell [${r}][${c}] has zero or negative dimension, skipping candidate.`);
                     possible_marker = false;
                     break;
                }

                const inset_ratio = 0.20; // 20% inset from each side
                const inset_w = Math.floor(current_cell_width * inset_ratio);
                const inset_h = Math.floor(current_cell_height * inset_ratio);

                let sample_x_start = cell_x_start + inset_w;
                let sample_y_start = cell_y_start + inset_h;
                let sample_x_end = cell_x_end - inset_w;
                let sample_y_end = cell_y_end - inset_h;

                // Log sampling bounds
                window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i}, cell [${r}][${c}]: sampling original bounds x:${cell_x_start}-${cell_x_end}, y:${cell_y_start}-${cell_y_end}. Insets: w=${inset_w},h=${inset_h}. Sample area x:${sample_x_start}-${sample_x_end}, y:${sample_y_start}-${sample_y_end}`);

                if (sample_x_start >= sample_x_end || sample_y_start >= sample_y_end) {
                    window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i}, cell [${r}][${c}] has zero or negative sample dimension after inset. Using original cell for sampling.`);
                    // Fallback to original cell boundaries if insets make it invalid
                    sample_x_start = cell_x_start;
                    sample_y_start = cell_y_start;
                    sample_x_end = cell_x_end;
                    sample_y_end = cell_y_end;
                }

                let sum_intensity = 0;
                let num_pixels = 0;

                for (let y_px = sample_y_start; y_px < sample_y_end; y_px++) {
                    for (let x_px = sample_x_start; x_px < sample_x_end; x_px++) {
                        if (x_px >= 0 && x_px < source_image.cols && y_px >= 0 && y_px < source_image.rows) {
                            const pixel_idx = y_px * source_image.cols + x_px;
                            sum_intensity += source_image.data[pixel_idx];
                            num_pixels++;
                        }
                    }
                }

                if (num_pixels === 0) {
                    window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i}, cell [${r}][${c}] had no pixels in sampling area (num_pixels = 0), skipping candidate.`);
                    possible_marker = false;
                    break;
                }

                const avg_intensity = sum_intensity / num_pixels;
                sampled_pattern_intensity[r][c] = avg_intensity;
                // Log average intensity but defer binary conversion
                window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i}, cell [${r}][${c}]: avg_intensity = ${avg_intensity.toFixed(2)} (from ${num_pixels} pixels)`);
            }
            if (!possible_marker) break; // Break from cell row loop
        }

        if (!possible_marker) { // Check if any cell failed
            window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i} skipped due to cell processing issues before local thresholding.`);
            continue; // Skip to the next candidate
        }

        // All 9 cell intensities are collected, now calculate local threshold
        const cell_intensities = [];
        for (let r_idx = 0; r_idx < 3; r_idx++) {
            for (let c_idx = 0; c_idx < 3; c_idx++) {
                cell_intensities.push(sampled_pattern_intensity[r_idx][c_idx]);
            }
        }

        if (cell_intensities.length !== 9) {
            window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i} did not have 9 cell intensities for local threshold. Skipping.`);
            continue;
        }

        const min_intensity = Math.min(...cell_intensities);
        const max_intensity = Math.max(...cell_intensities);

        // Check if min and max are substantially different. If not, it's likely not a valid marker.
        // A very small difference might also lead to an unstable threshold.
        if (max_intensity - min_intensity < 10) { // Threshold can be tuned, e.g. 10 gray levels
            window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i}: min/max cell intensity difference too small (${(max_intensity - min_intensity).toFixed(2)}). Unlikely to be a valid marker. Skipping.`);
            continue;
        }

        const local_marker_thresh = (min_intensity + max_intensity) / 2.0;
        window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i}: min_cell_intensity=${min_intensity.toFixed(2)}, max_cell_intensity=${max_intensity.toFixed(2)}, local_marker_thresh=${local_marker_thresh.toFixed(2)}`);

        // Now, derive the binary pattern using the local_marker_thresh
        for (let r_pat = 0; r_pat < 3; r_pat++) {
            for (let c_pat = 0; c_pat < 3; c_pat++) {
                derived_binary_pattern[r_pat][c_pat] = (sampled_pattern_intensity[r_pat][c_pat] <= local_marker_thresh) ? 'black' : 'white';
            }
        }

        window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i}: Derived pattern (local thresh): ${JSON.stringify(derived_binary_pattern)}`);
        window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i}: Expected pattern: ${JSON.stringify(marker_pattern_definition)}`);

        // Compare derived_binary_pattern with marker_pattern_definition
        let match = true;
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                if (derived_binary_pattern[r][c] !== marker_pattern_definition[r][c]) {
                    match = false;
                    break;
                }
            }
            if (!match) break;
        }

        if (match) {
            window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i} MATCHED! Adding to verified list.`);
            verified_markers.push(candidate);
        } else {
            window.logToScreen(`verifyAndFilterMarkers_jsfeat: Candidate ${i} did NOT match.`);
        }
    }

    window.logToScreen("verifyAndFilterMarkers_jsfeat: Found " + verified_markers.length + " verified markers.");
    window.logToScreen("verifyAndFilterMarkers_jsfeat: Function exit.");
    return verified_markers;
}

/**
 * Performs Non-Maximum Suppression on a list of markers.
 * @param {Array<Object>} markers - Array of marker objects (with x, y, width, height, area).
 * @param {number} overlapThresh - Maximum allowed Intersection over Union (IoU).
 * @returns {Array<Object>} Filtered list of markers after NMS.
 */
function nonMaxSuppression_jsfeat(markers, overlapThresh) {
    window.logToScreen(`nonMaxSuppression_jsfeat: Entry. Received ${markers.length} markers. Overlap threshold: ${overlapThresh}`);

    if (!markers || markers.length === 0) {
        window.logToScreen("nonMaxSuppression_jsfeat: No markers to process. Exiting.");
        return [];
    }

    // Ensure all markers have an area property.
    // analyzeBlobs_jsfeat should provide this, but as a fallback:
    markers.forEach(marker => {
        if (typeof marker.area === 'undefined') {
            marker.area = marker.width * marker.height;
            window.logToScreen(`nonMaxSuppression_jsfeat: Marker area recalculated for marker at x:${marker.x},y:${marker.y}`);
        }
    });

    // It's often beneficial to sort markers by a confidence score.
    // Since we don't have one, using area (descending) as a proxy.
    // findMarkerCandidates_jsfeat already sorts by area descending.
    // If they were not pre-sorted, uncomment: markers.sort((a, b) => b.area - a.area);


    let final_markers = [];
    let remaining_markers = [...markers]; // Work with a copy

    while (remaining_markers.length > 0) {
        // Select the 'best' marker (first one, assuming sorted by area/confidence)
        const current_marker = remaining_markers[0];
        final_markers.push(current_marker);
        window.logToScreen(`nonMaxSuppression_jsfeat: Selected marker: x=${current_marker.x}, y=${current_marker.y}, w=${current_marker.width}, h=${current_marker.height}, area=${current_marker.area}`);

        // Remove current_marker from remaining_markers for the next main loop iteration
        // and prepare a list for IoU checks in this iteration.
        const markers_to_check_iou = remaining_markers.slice(1);
        remaining_markers = []; // This will be rebuilt with markers that don't overlap too much

        for (let i = 0; i < markers_to_check_iou.length; i++) {
            const other_marker = markers_to_check_iou[i];

            // Calculate Intersection Rect
            const ix1 = Math.max(current_marker.x, other_marker.x);
            const iy1 = Math.max(current_marker.y, other_marker.y);
            const ix2 = Math.min(current_marker.x + current_marker.width, other_marker.x + other_marker.width);
            const iy2 = Math.min(current_marker.y + current_marker.height, other_marker.y + other_marker.height);

            const i_width = Math.max(0, ix2 - ix1);
            const i_height = Math.max(0, iy2 - iy1);
            const intersection_area = i_width * i_height;

            // Calculate Union Area
            // Area properties should exist from analyzeBlobs_jsfeat or fallback calculation
            const union_area = current_marker.area + other_marker.area - intersection_area;

            let iou = 0;
            if (union_area > 0) {
                iou = intersection_area / union_area;
            }
            window.logToScreen(`nonMaxSuppression_jsfeat: Comparing with marker (x=${other_marker.x},y=${other_marker.y}): IoU = ${iou.toFixed(3)}`);

            if (iou <= overlapThresh) {
                remaining_markers.push(other_marker); // Keep this marker for next iteration
            } else {
                window.logToScreen(`nonMaxSuppression_jsfeat: Suppressed marker: x=${other_marker.x}, y=${other_marker.y} due to IoU ${iou.toFixed(3)} > ${overlapThresh}`);
            }
        }
    }

    window.logToScreen(`nonMaxSuppression_jsfeat: Exiting. Returning ${final_markers.length} markers.`);
    return final_markers;
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
