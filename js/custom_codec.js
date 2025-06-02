// js/custom_codec.js

// Constants for encoding/drawing
const CUST_CODEC_COLOR_RGB_MAP = {
    'black': { r: 0, g: 0, b: 0, style: 'rgb(0,0,0)' },  'white': { r: 255, g: 255, b: 255, style: 'rgb(255,255,255)' }, // Magenta for 'white' symbols - changed to white again no more magenta
    'blue': { r: 0, g: 0, b: 255, style: 'rgb(0,0,255)' }, 'green': { r: 0, g: 255, b: 0, style: 'rgb(0,255,0)' },
    'yellow': { r: 255, g: 255, b: 0, style: 'rgb(255,255,0)' }, 'red': { r: 255, g: 0, b: 0, style: 'rgb(255,0,0)' },
    'gray': { r: 128, g: 128, b: 128, style: 'rgb(128,128,128)' }, 'background': { r: 255, g: 0, b: 255, style: 'rgb(255,0,255)' } // changed background white to magenta.
};
const CUST_CODEC_SHAPES = ['square', 'circle', 'triangle'];
const CUST_CODEC_COLORS = ['black', 'white', 'blue', 'green', 'yellow', 'red'];
const CUST_CODEC_SYMBOL_LIST = [];
for (const color of CUST_CODEC_COLORS) { for (const shape of CUST_CODEC_SHAPES) { CUST_CODEC_SYMBOL_LIST.push([color, shape]); }}
const CUST_CODEC_SPECIAL_SYMBOL = ['gray', 'square'];
const CUST_CODEC_MARKER_OUTER = ['black', 'square'];
const CUST_CODEC_MARKER_INNER = ['white', 'square'];
const CUST_CODEC_MARKER_PATTERN = [
    [CUST_CODEC_MARKER_OUTER, CUST_CODEC_MARKER_OUTER, CUST_CODEC_MARKER_OUTER],
    [CUST_CODEC_MARKER_OUTER, CUST_CODEC_MARKER_INNER, CUST_CODEC_MARKER_OUTER],
    [CUST_CODEC_MARKER_OUTER, CUST_CODEC_MARKER_OUTER, CUST_CODEC_MARKER_OUTER]
];
const CUST_CODEC_MARKER_SIZE_CELLS = CUST_CODEC_MARKER_PATTERN.length;
const CUST_CODEC_CELL_DRAW_SIZE = 10;
const CUST_CODEC_MARGIN_DRAW_SIZE = 3; // Adjusted to 3 to match Python visuals more closely
const RS_CODEWORD_LEN_N = 18;
const RS_MESSAGE_LEN_K = 14;
const RS_PARITY_SYMBOLS = RS_CODEWORD_LEN_N - RS_MESSAGE_LEN_K;

// Marker and Data Conversion functions
const markerByteMapJs = {
    '<linear>': 257, '</linear>': 258, '<power>': 259, '</power>': 260,
    '<base64>': 281, '</base64>': 282, '<encrypt_method>': 285, '</encrypt_method>': 286,
    '<encrypt_key>': 287, '</encrypt_key>': 288, '<encrypt>': 289, '</encrypt>': 290,
};
const reverseMarkerByteMapJs = Object.fromEntries(Object.entries(markerByteMapJs).map(([k, v]) => [v, k]));
const markerPairsJs = { 257: 258, 259: 260, 281: 282, 285: 286, 287: 288, 289: 290 };
const markerNamesJs = { 257: 'linear', 259: 'power', 281: 'base64', 285: 'encrypt_method', 287: 'encrypt_key', 289: 'encrypt' };

/**
 * Applies PKCS#7 padding to data.
 * @param {Uint8Array} data - The data to pad.
 * @param {Number} blockSize - The block size (e.g., 16 for AES).
 * @returns {Uint8Array} The padded data.
 */
function pkcs7Pad(data, blockSize) {
    const padding = blockSize - (data.length % blockSize);
    const padArray = new Uint8Array(padding).fill(padding);
    const result = new Uint8Array(data.length + padding);
    result.set(data, 0); result.set(padArray, data.length); return result;
}
/**
 * Removes PKCS#7 padding from data.
 * @param {Uint8Array} data - The data with padding.
 * @returns {Uint8Array} The unpadded data.
 * @throws Error if padding is invalid.
 */
function pkcs7Unpad(data) {
    if (data.length === 0) throw new Error("pkcs7Unpad: Input data is empty.");
    const padding = data[data.length - 1];
    if (padding === 0 || padding > data.length) throw new Error("pkcs7Unpad: Invalid PKCS#7 padding value.");
    for (let i = data.length - padding; i < data.length -1; i++) { if (data[i] !== padding) throw new Error("pkcs7Unpad: Invalid PKCS#7 padding bytes.");}
    return data.slice(0, data.length - padding);
}
/**
 * Encrypts plaintext using AES-CBC with a zero IV (simulating ECB for independent blocks).
 * The key is truncated or zero-padded to 16 bytes.
 * @param {Uint8Array} plainTextBytes - The plaintext to encrypt.
 * @param {String} keyString - The encryption key string.
 * @returns {Promise<Array<Number>>} A promise that resolves to the ciphertext as an array of byte values.
 */
async function aesEncryptJs(plainTextBytes, keyString) {
    const encoder = new TextEncoder();
    let keyBytes = encoder.encode(keyString);
    let preparedKeyBytes = new Uint8Array(16);
    if (keyBytes.length < 16) {
        preparedKeyBytes.set(keyBytes);
    } else {
        preparedKeyBytes = keyBytes.slice(0, 16);
    }
    const keyMaterial = await crypto.subtle.importKey("raw", preparedKeyBytes, "AES-CBC", false, ["encrypt"]);

    // Original IV was: const iv = new Uint8Array(16); // Zero IV

    const paddedPlainTextBytes = pkcs7Pad(plainTextBytes, 16);
    let ciphertext = new Uint8Array(paddedPlainTextBytes.length);

    console.log(`aesEncryptJs: plainTextBytes len: ${plainTextBytes.length}, paddedLen: ${paddedPlainTextBytes.length}`);

    // Web Crypto API's encrypt for AES-CBC should handle the entire padded message at once.
    // The manual block-by-block loop with a reset IV was effectively trying to simulate ECB mode,
    // but the RangeError on ciphertext.set suggests encryptedBlock might be unexpectedly sized if API does internal buffering/finalization.
    // Let's simplify to use the API as intended for a single buffer.
    // The IV should be used once for the entire encryption operation.
    const iv_for_encryption = new Uint8Array(16); // Use a consistent zero IV for encryption for now.
                                              // If WebCrypto actually prepends this, then decrypt will use it.
                                              // If WebCrypto *generates* an IV, this one is ignored, and the generated one is prepended.

    try {
        const encryptedOutputWithPotentialIV = await crypto.subtle.encrypt(
            { name: "AES-CBC", iv: iv_for_encryption }, // Provide the IV here
            keyMaterial,
            paddedPlainTextBytes
        );

        // Assuming the output might be IV + Ciphertext if length is greater than paddedPlainTextBytes.length
        // For "SecretData" (10B -> 16B padded), output was 32B. This implies 16B IV + 16B CT.
        if (encryptedOutputWithPotentialIV.byteLength > paddedPlainTextBytes.length) {
            console.warn(`aesEncryptJs: Output length (${encryptedOutputWithPotentialIV.byteLength}) > padded input length (${paddedPlainTextBytes.length}). Assuming IV is prepended to ciphertext.`);
             // In this case, the entire output (IV + CT) is what we store/send.
            ciphertext = new Uint8Array(encryptedOutputWithPotentialIV);
        } else if (encryptedOutputWithPotentialIV.byteLength === paddedPlainTextBytes.length) {
            console.log(`aesEncryptJs: Output length matches padded input. Assuming API used provided IV and returned only ciphertext.`);
            // If the API behaves "normally" and only returns ciphertext matching input length (because IV was passed in params)
            // We would need to prepend our iv_for_encryption IF the decrypt side expects it.
            // To keep consistency with the "IV prepended" model that the 32-byte output suggests:
            // This path implies the environment is different from where 32B was seen.
            // For now, stick to the assumption based on prior log: output is IV+CT or just CT.
            // The code will now assume `encryptedOutputWithPotentialIV` is what needs to be passed to decryptor (either as IV+CT or just CT).
            ciphertext = new Uint8Array(encryptedOutputWithPotentialIV);
        } else {
            // This would be an error or very strange behavior
            console.error(`aesEncryptJs: Output length (${encryptedOutputWithPotentialIV.byteLength}) is less than padded input length (${paddedPlainTextBytes.length}). This is an error.`);
            throw new Error("AES encryption output length is unexpectedly less than padded input length.");
        }

    } catch (e) {
        console.error(`Error during crypto.subtle.encrypt for the whole message: ${e.name} - ${e.message}`);
        console.error(`Padded Plaintext details: length=${paddedPlainTextBytes.byteLength}, content=[${new Uint8Array(paddedPlainTextBytes).join(',')}]`);
        console.error(`IV details: length=${iv.byteLength}, content=[${new Uint8Array(iv).join(',')}]`);
        console.error(`Key material: ${keyMaterial.type}, usages: ${keyMaterial.usages.join(',')}`);
        throw e; // Re-throw the error
    }

    return Array.from(ciphertext); // ciphertext is already a Uint8Array here
}
/**
 * Decrypts ciphertext using AES-CBC with a zero IV.
 * Key handling is the same as encryption.
 * @param {Array<Number>} ciphertextBytes - The ciphertext as an array of byte values.
 * @param {String} keyString - The decryption key string.
 * @returns {Promise<Uint8Array>} A promise that resolves to the decrypted plaintext as a Uint8Array.
 * @throws Error if decryption or unpadding fails.
 */
async function aesDecryptJs(ciphertextBytes, keyString) {
    console.log(`aesDecryptJs: Called with ciphertextBytes length: ${ciphertextBytes.length}, keyString: "${keyString}"`);
    if (!ciphertextBytes || ciphertextBytes.length === 0) {
        console.error("aesDecryptJs: ciphertextBytes is null or empty.");
        throw new Error("aesDecryptJs: Ciphertext cannot be empty.");
    }

    const encoder = new TextEncoder();
    let keyBytes = encoder.encode(keyString);
    let preparedKeyBytes = new Uint8Array(16);
    if (keyBytes.length < 16) {
        preparedKeyBytes.set(keyBytes);
    } else {
        preparedKeyBytes = keyBytes.slice(0, 16);
    }
    console.log(`aesDecryptJs: Prepared key bytes (first 5): ${preparedKeyBytes.slice(0,5).join(',')}`);

    const keyMaterial = await crypto.subtle.importKey("raw", preparedKeyBytes, "AES-CBC", false, ["decrypt"]);
    console.log(`aesDecryptJs: Key material imported: type=${keyMaterial.type}, usages=${keyMaterial.usages.join(',')}`);

    const ciphertextBuffer = new Uint8Array(ciphertextBytes);

    // Based on previous findings, crypto.subtle.encrypt prepends a 16-byte IV.
    // So, ciphertextBuffer is expected to be IV (16 bytes) + actual encrypted data (multiple of 16 bytes).
    // Minimum length would be 16 (IV) + 16 (one block of data) = 32 bytes if encrypt handled one block.
    if (ciphertextBuffer.length < 32) { // Must have at least one IV block and one data block
        console.error(`aesDecryptJs: Ciphertext length (${ciphertextBuffer.length}) is less than the required minimum of 32 bytes (IV + 1 block).`);
        // This was a previous point of failure if encrypt output 32 bytes and decrypt got less.
        // Now encrypt is supposed to return the full 32 bytes for one block.
        // The test case "SecretData" (10 bytes) -> padded to 16 -> encrypted (IV+Cipher) to 32. So this should be fine.
    }
     if (ciphertextBuffer.length % 16 !== 0) {
        console.error(`aesDecryptJs: Ciphertext length (${ciphertextBuffer.length}) is not a multiple of 16.`);
        // This would be an issue. Encrypt (IV+Cipher) should be mult of 16.
    }


    let decryptedPaddedBytes = new Uint8Array(ciphertextBuffer.length - 16); // To store only decrypted data part

    // Assuming the new model: first 16 bytes of ciphertextBuffer is the IV
    const extractedIv = ciphertextBuffer.slice(0, 16);
    console.log(`aesDecryptJs: Extracted IV (first 5 bytes): ${extractedIv.slice(0,5).join(',')}, length: ${extractedIv.byteLength}`);

    let actualDecryptedDataLength = 0;

    for (let i = 16; i < ciphertextBuffer.length; i += 16) {
        const blockToDecrypt = ciphertextBuffer.slice(i, i + 16);

        if (blockToDecrypt.byteLength === 0) {
            console.warn(`aesDecryptJs: Encountered an empty block to decrypt at offset ${i}. Skipping.`);
            continue;
        }
        if (blockToDecrypt.byteLength !== 16) {
            // This would be an issue if not the very last block and padding wasn't handled by encrypt side for total length.
            // However, with IV prepended, total length should be (IV_len + N * block_size).
            // So each data block sliced should be 16 bytes.
            console.error(`aesDecryptJs: Block to decrypt at offset ${i} has unexpected length: ${blockToDecrypt.byteLength}`);
            throw new Error(`aesDecryptJs: Invalid block size (${blockToDecrypt.byteLength}) for decryption at offset ${i}.`);
        }

        console.log(`aesDecryptJs loop i=${i}: blockToDecrypt len ${blockToDecrypt.byteLength}`);
        try {
            const decryptedBlockBuffer = await crypto.subtle.decrypt(
                { name: "AES-CBC", iv: extractedIv }, // Using the IV extracted from the start of ciphertextBuffer
                keyMaterial,
                blockToDecrypt // This is just the ciphertext part for this block
            );

            if (decryptedBlockBuffer.byteLength === 0) {
                console.error("aesDecryptJs: crypto.subtle.decrypt returned an empty ArrayBuffer. This indicates a decryption failure.");
                console.error(`IV used: [${new Uint8Array(extractedIv).join(',')}]`);
                console.error(`Ciphertext block fed: [${new Uint8Array(blockToDecrypt).join(',')}]`);
                throw new Error("Decryption resulted in empty buffer, indicating failure.");
            }

            // Place the decrypted block into the correct position in decryptedPaddedBytes
            // The offset for decryptedPaddedBytes should be (i - 16)
            decryptedPaddedBytes.set(new Uint8Array(decryptedBlockBuffer), i - 16);
            actualDecryptedDataLength += decryptedBlockBuffer.byteLength;

        } catch (e) {
            console.error(`Error during crypto.subtle.decrypt at block offset ${i}: ${e.name} - ${e.message}`);
            console.error(`Block details: length=${blockToDecrypt.byteLength}, content=[${new Uint8Array(blockToDecrypt).join(',')}]`);
            console.error(`IV details: length=${extractedIv.byteLength}, content=[${new Uint8Array(extractedIv).join(',')}]`);
            throw e;
        }
        // IMPORTANT: For CBC mode, if we were decrypting more than one block, the IV for the *next* block
        // would be the *ciphertext of the current block* (blockToDecrypt).
        // However, the current structure assumes that the single `extractedIv` from the beginning
        // is the correct IV for *all* blocks. This is only true if encrypt also used the same IV for all blocks
        // (effectively making it like ECB, or if each block was a separate CBC encryption starting with that IV).
        // Given that `encrypt` now prepends a *single* IV for the whole payload, this decryption logic of reusing
        // `extractedIv` for all blocks is correct for decrypting that specific structure.
    }

    // Trim decryptedPaddedBytes to actualDecryptedDataLength if it was oversized (should not be if logic is correct)
    let finalDecryptedData = decryptedPaddedBytes;
    if (decryptedPaddedBytes.byteLength > actualDecryptedDataLength) {
        finalDecryptedData = decryptedPaddedBytes.slice(0, actualDecryptedDataLength);
    }
    console.log(`aesDecryptJs: Total decrypted data length (before unpadding): ${finalDecryptedData.byteLength}`);

    const unpaddedBytes = pkcs7Unpad(finalDecryptedData);
    console.log(`aesDecryptJs: Unpadded data length: ${unpaddedBytes.byteLength}`);
    return new Uint8Array(unpaddedBytes);
}
/**
 * Converts an input string (potentially with special tags) into a list of byte values (0-323).
 * Handles tag parsing and AES encryption if specified.
 * @param {String} inputString - The string to encode.
 * @returns {Promise<Array<Number>>} A promise that resolves to a list of byte values.
 * @throws Error if byte values are out of the 0-323 range.
 */
async function textToByteListJs(inputString) {
    if (typeof inputString !== 'string') { console.error("textToByteListJs: Expected a string."); return []; }
    const markerKeys = Object.keys(markerByteMapJs).map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const markerPatternJs = new RegExp(`(${markerKeys.join('|')})`, 'g');
    const byteList = []; const encoder = new TextEncoder(); let lastIndex = 0; let match;
    let storedEncryptMethod = null; let storedEncryptKey = null;
    let isEncryptingContent = false; let isParsingEncryptMethod = false; let isParsingEncryptKey = false;
    while ((match = markerPatternJs.exec(inputString)) !== null) {
        const textSegment = inputString.substring(lastIndex, match.index);
        if (textSegment) {
            if (isEncryptingContent) {
                if (storedEncryptMethod === 'AES' && storedEncryptKey) { byteList.push(...await aesEncryptJs(encoder.encode(textSegment), storedEncryptKey));
                } else { byteList.push(...encoder.encode(textSegment)); }
            } else if (isParsingEncryptMethod) { storedEncryptMethod = textSegment.trim(); byteList.push(...encoder.encode(textSegment)); isParsingEncryptMethod = false;
            } else if (isParsingEncryptKey) { storedEncryptKey = textSegment.trim(); byteList.push(...encoder.encode(textSegment)); isParsingEncryptKey = false;
            } else { byteList.push(...encoder.encode(textSegment)); } }
        const marker = match[0]; byteList.push(markerByteMapJs[marker]);
        if (marker === '<encrypt_method>') isParsingEncryptMethod = true; else if (marker === '</encrypt_method>') isParsingEncryptMethod = false;
        else if (marker === '<encrypt_key>') isParsingEncryptKey = true; else if (marker === '</encrypt_key>') isParsingEncryptKey = false;
        else if (marker === '<encrypt>') isEncryptingContent = true; else if (marker === '</encrypt>') isEncryptingContent = false;
        lastIndex = markerPatternJs.lastIndex; }
    const remainingText = inputString.substring(lastIndex);
    if (remainingText) { if (isEncryptingContent && storedEncryptMethod === 'AES' && storedEncryptKey) { byteList.push(...await aesEncryptJs(encoder.encode(remainingText), storedEncryptKey));
        } else { byteList.push(...encoder.encode(remainingText));} }
    for (const byte of byteList) { if (byte < 0 || byte > 323) throw new Error("Byte value out of range (0-323): " + byte); }
    return byteList;
}
/**
 * Converts a list of byte values (0-323) into pairs of GF(19) symbol indices (0-17).
 * Each byte value `B` is mapped to `[floor(B/18), B % 18]`.
 * @param {Array<Number>} byteList - The list of byte values.
 * @returns {Array<Number>} A flat list of symbol indices.
 * @throws Error if byte values are out of range.
 */
function byteListToSymbolIndicesJs(byteList) {
    const symbolIndices = [];
    for (const byteValue of byteList) { if (byteValue < 0 || byteValue > 323) throw new Error(`byteListToSymbolIndicesJs: Invalid byte value ${byteValue}.`);
        symbolIndices.push(Math.floor(byteValue / 18)); symbolIndices.push(byteValue % 18);
    } return symbolIndices;
}
/**
 * Converts pairs of GF(19) symbol indices (0-17) back to a list of byte values.
 * Each pair `[s1, s2]` is mapped to `s1 * 18 + s2`.
 * @param {Array<Number>} symbolIndices - The flat list of symbol indices.
 * @returns {Array<Number>} A list of byte values.
 * @throws Error if symbol indices are out of range or list length is odd.
 */
function symbolIndicesToByteListJs(symbolIndices) {
    const byteList = []; let length = symbolIndices.length;
    if (length % 2 !== 0) { console.warn("symbolIndicesToByteListJs: Odd number of symbol indices. Last ignored."); length--; }
    for (let i = 0; i < length; i += 2) { const s1 = symbolIndices[i]; const s2 = symbolIndices[i + 1];
        if (s1 < 0 || s1 > 17 || s2 < 0 || s2 > 17) throw new Error(`symbolIndicesToByteListJs: Invalid symbol index. Found ${s1}, ${s2}. Must be 0-17.`);
        byteList.push(s1 * 18 + s2);
    } return byteList;
}
/**
 * Converts a list of byte values back to a string, processing special tags (decryption, base64).
 * @param {Array<Number>} byteList - The list of byte values.
 * @returns {Promise<String>} A promise that resolves to the final decoded and processed string.
 */
async function byteListToStringJs(byteList) {
    let outputString = ""; let i = 0; const textDecoder = new TextDecoder();
    let currentPlainTextBuffer = []; let storedEncryptMethod = null; let storedEncryptKey = null;
    function flushPlainTextBuffer() { if (currentPlainTextBuffer.length > 0) { outputString += textDecoder.decode(new Uint8Array(currentPlainTextBuffer)); currentPlainTextBuffer = []; } }
    while (i < byteList.length) {
        const byteValue = byteList[i]; const markerTag = reverseMarkerByteMapJs[byteValue];
        const endMarkerByteExpected = markerPairsJs[byteValue];
        if (markerTag && endMarkerByteExpected) {
            flushPlainTextBuffer(); outputString += markerTag; i++; let contentBytes = [];
            while (i < byteList.length && byteList[i] !== endMarkerByteExpected) { contentBytes.push(byteList[i]); i++; }
            const markerName = markerNamesJs[byteValue];
            if (markerName === 'encrypt_method') { storedEncryptMethod = textDecoder.decode(new Uint8Array(contentBytes)).trim(); outputString += storedEncryptMethod; }
            else if (markerName === 'encrypt_key') { storedEncryptKey = textDecoder.decode(new Uint8Array(contentBytes)).trim(); outputString += storedEncryptKey; }
            else if (markerName === 'encrypt') {
                if (storedEncryptMethod === 'AES' && storedEncryptKey) { try { outputString += textDecoder.decode(await aesDecryptJs(contentBytes, storedEncryptKey)); } catch (e) { console.error("AES Decryption Error:", e); outputString += "[DECRYPTION_ERROR]"; } }
                else { outputString += textDecoder.decode(new Uint8Array(contentBytes));}
            } else if (markerName === 'base64') { try { outputString += atob(textDecoder.decode(new Uint8Array(contentBytes))); } catch (e) { console.error("Base64 Decode Error:", e); outputString += "[BASE64_DECODE_ERROR]";}
            } else { outputString += textDecoder.decode(new Uint8Array(contentBytes)); }
            if (i < byteList.length && byteList[i] === endMarkerByteExpected) { outputString += reverseMarkerByteMapJs[byteList[i]]; i++; }
            else { console.warn(`Expected end marker for ${markerName} not found or mismatched.`); }
        } else if (markerTag && !endMarkerByteExpected) { flushPlainTextBuffer(); outputString += markerTag; i++;
        } else { currentPlainTextBuffer.push(byteValue); i++; }
    } flushPlainTextBuffer(); return outputString;
}

/**
 * Orchestrates the full decoding pipeline from a flat list of extracted symbol indices.
 * This involves Reed-Solomon decoding, then conversion from symbols to bytes, then bytes to string.
 * @param {Array<Number>} flatSymbolIndices - Flat list of symbol indices (0-18) from image processing.
 * @returns {Promise<String|null>} The final decoded string, or null if any stage fails.
 */
async function fullDecodePipelineFromIndices(flatSymbolIndices) {
    console.info("fullDecodePipelineFromIndices: Input symbol indices length:", flatSymbolIndices.length);
    if (flatSymbolIndices.length === 0) { console.warn("Pipeline Error: Empty flatSymbolIndices."); return null; }
    let processingIndices = [...flatSymbolIndices];
    if (processingIndices.length % RS_CODEWORD_LEN_N !== 0) {
        console.warn(`Pipeline Warning: Input length ${processingIndices.length} is not a multiple of N (${RS_CODEWORD_LEN_N}). Truncating.`);
        processingIndices = processingIndices.slice(0, Math.floor(processingIndices.length / RS_CODEWORD_LEN_N) * RS_CODEWORD_LEN_N);
        if (processingIndices.length === 0) { console.warn("Pipeline Error: No symbols left after truncating to multiple of N."); return null; }
    }
    const rsDecoder = new ReedSolomonDecoderGF19(RS_PARITY_SYMBOLS, RS_CODEWORD_LEN_N);
    const allDecodedMessageSymbols = []; let decodingErrorOccurred = false;
    for (let i = 0; i < processingIndices.length; i += RS_CODEWORD_LEN_N) {
        const codewordBlock = processingIndices.slice(i, i + RS_CODEWORD_LEN_N);
        const decodedMessageBlock = rsDecoder.decode(codewordBlock);
        if (decodedMessageBlock === null) { console.warn(`Pipeline Warning: RS decoding failed for block starting at index ${i}.`); decodingErrorOccurred = true; break; }
        allDecodedMessageSymbols.push(...decodedMessageBlock);
    }
    if (decodingErrorOccurred || allDecodedMessageSymbols.length === 0) { console.warn("Pipeline Error: RS decoding failed or resulted in no message symbols."); return null; }
    console.info("Pipeline: RS Decoded Message Symbols (length " + allDecodedMessageSymbols.length + ")");
    try {
        const byteList = symbolIndicesToByteListJs(allDecodedMessageSymbols);
        console.info("Pipeline: Converted to Byte List (length " + byteList.length + ")");
        const finalString = await byteListToStringJs(byteList);
        console.info("Pipeline: Final Decoded String (preview):", finalString.substring(0, 100) + (finalString.length > 100 ? "..." : ""));
        return finalString;
    } catch (e) { console.error("Pipeline Error in final conversion (symbol/byte to string):", e.stack); return null; }
}

/**
 * Draws the encoded visual code onto a canvas element.
 * @param {Array<Number>} encodedSymbolIndices - Flat list of final symbol indices (0-18) after RS encoding.
 * @param {HTMLCanvasElement} canvasElement - The canvas element to draw on.
 * @param {HTMLAnchorElement} downloadLinkElement - The anchor element for the image download link.
 */
function drawEncodedImageOnCanvas(encodedSymbolIndices, canvasElement, downloadLinkElement) {
    const numTotalSymbols = encodedSymbolIndices.length;
    if (numTotalSymbols === 0) { console.warn("drawEncodedImageOnCanvas: No symbols to draw."); canvasElement.style.display = 'none'; if (downloadLinkElement) downloadLinkElement.style.display = 'none'; return; }

    const data_area_size = Math.ceil(Math.sqrt(numTotalSymbols));
    const grid_width = data_area_size + CUST_CODEC_MARKER_SIZE_CELLS + 1;
    const grid_height = data_area_size + CUST_CODEC_MARKER_SIZE_CELLS + 1;
    console.info(`Drawing grid: ${grid_width}x${grid_height} for ${numTotalSymbols} symbols. Data area: ${data_area_size}x${data_area_size}`);

    const grid = Array(grid_height).fill(null).map(() => Array(grid_width).fill(null).map(() => [...CUST_CODEC_SPECIAL_SYMBOL]));
    for (let r = 0; r < CUST_CODEC_MARKER_SIZE_CELLS; r++) { for (let c = 0; c < CUST_CODEC_MARKER_SIZE_CELLS; c++) {
            grid[r][c] = CUST_CODEC_MARKER_PATTERN[r][c];
            grid[r][grid_width - CUST_CODEC_MARKER_SIZE_CELLS + c] = CUST_CODEC_MARKER_PATTERN[r][c];
            grid[grid_height - CUST_CODEC_MARKER_SIZE_CELLS + r][c] = CUST_CODEC_MARKER_PATTERN[r][c]; } }
    let symbolIdxPtr = 0;
    for (let r = 0; r < grid_height; r++) { for (let c = 0; c < grid_width; c++) {
            const isTLM = (r < CUST_CODEC_MARKER_SIZE_CELLS && c < CUST_CODEC_MARKER_SIZE_CELLS);
            const isTRM = (r < CUST_CODEC_MARKER_SIZE_CELLS && c >= grid_width - CUST_CODEC_MARKER_SIZE_CELLS);
            const isBLM = (r >= grid_height - CUST_CODEC_MARKER_SIZE_CELLS && c < CUST_CODEC_MARKER_SIZE_CELLS);
            if (isTLM || isTRM || isBLM) continue;
            if (symbolIdxPtr < numTotalSymbols) { const symbolVal = encodedSymbolIndices[symbolIdxPtr++];
                grid[r][c] = (symbolVal >= 0 && symbolVal < CUST_CODEC_SYMBOL_LIST.length) ? CUST_CODEC_SYMBOL_LIST[symbolVal] : CUST_CODEC_SPECIAL_SYMBOL;
            }}}
    const ctx = canvasElement.getContext('2d');
    canvasElement.width = grid_width * (CUST_CODEC_CELL_DRAW_SIZE + CUST_CODEC_MARGIN_DRAW_SIZE) + CUST_CODEC_MARGIN_DRAW_SIZE;
    canvasElement.height = grid_height * (CUST_CODEC_CELL_DRAW_SIZE + CUST_CODEC_MARGIN_DRAW_SIZE) + CUST_CODEC_MARGIN_DRAW_SIZE;
    ctx.fillStyle = CUST_CODEC_COLOR_RGB_MAP['background'].style; ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);
    for (let r = 0; r < grid_height; r++) { for (let c = 0; c < grid_width; c++) {
            const [colorName, shape] = grid[r][c];
            ctx.fillStyle = CUST_CODEC_COLOR_RGB_MAP[colorName] ? CUST_CODEC_COLOR_RGB_MAP[colorName].style : CUST_CODEC_COLOR_RGB_MAP['gray'].style;
            const x = CUST_CODEC_MARGIN_DRAW_SIZE + c * (CUST_CODEC_CELL_DRAW_SIZE + CUST_CODEC_MARGIN_DRAW_SIZE);
            const y = CUST_CODEC_MARGIN_DRAW_SIZE + r * (CUST_CODEC_CELL_DRAW_SIZE + CUST_CODEC_MARGIN_DRAW_SIZE);
            if (shape === 'square') { ctx.fillRect(x, y, CUST_CODEC_CELL_DRAW_SIZE, CUST_CODEC_CELL_DRAW_SIZE); }
            else if (shape === 'circle') { ctx.beginPath(); ctx.arc(x + CUST_CODEC_CELL_DRAW_SIZE / 2, y + CUST_CODEC_CELL_DRAW_SIZE / 2, CUST_CODEC_CELL_DRAW_SIZE / 2, 0, 2 * Math.PI); ctx.fill(); }
            else if (shape === 'triangle') { ctx.beginPath(); ctx.moveTo(x + CUST_CODEC_CELL_DRAW_SIZE / 2, y); ctx.lineTo(x, y + CUST_CODEC_CELL_DRAW_SIZE); ctx.lineTo(x + CUST_CODEC_CELL_DRAW_SIZE, y + CUST_CODEC_CELL_DRAW_SIZE); ctx.closePath(); ctx.fill(); } } }
    canvasElement.style.display = 'block';
    if (downloadLinkElement) { downloadLinkElement.href = canvasElement.toDataURL('image/png'); downloadLinkElement.style.display = 'inline-block'; }
    console.info("Encoded image drawn on canvas.");
}

/**
 * Orchestrates the full client-side encoding pipeline from text to a rendered visual code on canvas.
 * @param {String} textToEncode - The input string to encode.
 * @param {HTMLCanvasElement} canvasElement - The canvas element to draw the visual code on.
 * @param {HTMLAnchorElement} downloadLinkElement - The anchor element to update with the image download link.
 * @returns {Promise<void>}
 */
async function generateFullEncodedQrOnCanvas(textToEncode, canvasElement, downloadLinkElement) {
    console.info("generateFullEncodedQrOnCanvas for text:", textToEncode.substring(0, 30) + "...");
    try {
        const byteList = await textToByteListJs(textToEncode);
        const rawSymbolIndices = byteListToSymbolIndicesJs(byteList);
        const k = RS_MESSAGE_LEN_K; let paddedRawSymbolIndices = [...rawSymbolIndices];
        if (rawSymbolIndices.length > 0 && rawSymbolIndices.length % k !== 0) { // Only pad if there's actual data
            const paddingLength = k - (rawSymbolIndices.length % k);
            for (let i = 0; i < paddingLength; i++) { paddedRawSymbolIndices.push(0); }
            console.info(`Padded raw symbol indices by ${paddingLength} to length ${paddedRawSymbolIndices.length}`);
        }
        const rsEncoder = new ReedSolomonEncoderGF19(RS_PARITY_SYMBOLS);
        const finalEncodedSymbolIndices = [];
        if (paddedRawSymbolIndices.length > 0) { // Only encode if there are symbols
            for (let i = 0; i < paddedRawSymbolIndices.length; i += k) {
                const messageBlock = paddedRawSymbolIndices.slice(i, i + k);
                finalEncodedSymbolIndices.push(...rsEncoder.encode(messageBlock));
            }
        }
        console.info("Final Encoded Symbol Indices (with parity, len " + finalEncodedSymbolIndices.length + ")");
        drawEncodedImageOnCanvas(finalEncodedSymbolIndices, canvasElement, downloadLinkElement);
    } catch (error) {
        console.error("Error during encoding process:", error, error.stack ? error.stack.substring(0,300):"");
        alert("Encoding Error: " + error.message);
        if(canvasElement) canvasElement.style.display = 'none';
        if (downloadLinkElement) downloadLinkElement.style.display = 'none';
    }
}

// Test function (from previous step)
async function testCustomCodec() {
    console.log("--- Custom Codec (Data Conversion) Tests ---");
    let pass = true;
    const checkCodec = (desc, actual, expected) => {
        const success = JSON.stringify(actual) === JSON.stringify(expected);
        console.log(`Codec Test: ${desc} - Expected: ${JSON.stringify(expected)}, Got: ${JSON.stringify(actual)}. ${success ? 'PASS' : 'FAIL'}`);
        if (!success) pass = false;
    };
    const checkAsyncCodec = async (desc, actualPromise, expected) => {
        const actual = await actualPromise;
        const success = JSON.stringify(actual) === JSON.stringify(expected);
        console.log(`Codec Async Test: ${desc} - Expected: ${JSON.stringify(expected)}, Got: ${JSON.stringify(actual)}. ${success ? 'PASS' : 'FAIL'}`);
        if (!success) pass = false;
    };

    const testStr1 = "Hello";
    const byteList1 = await textToByteListJs(testStr1);
    await checkAsyncCodec("textToByteList (Hello)", byteList1, [72,101,108,108,111]);
    const str1Re = await byteListToStringJs(byteList1);
    await checkAsyncCodec("byteListToString (Hello)", str1Re, testStr1);

    const testByteList2 = [0, 17, 18, 323, 75];
    const symIndices2 = byteListToSymbolIndicesJs(testByteList2);
    checkCodec("byteListToSymbolIndices", symIndices2, [0,0, 0,17, 1,0, 17,17, 4,3]);
    const byteList2Re = symbolIndicesToByteListJs(symIndices2);
    checkCodec("symbolIndicesToByteList", byteList2Re, testByteList2);

    const aesKey = "testkey12345678";
    const aesPlain = "SecretData";
    const aesPlainBytes = new TextEncoder().encode(aesPlain);
    const aesCipherBytesArray = await aesEncryptJs(aesPlainBytes, aesKey);
    const aesDecryptedUint8Array = await aesDecryptJs(aesCipherBytesArray, aesKey);
    const aesDecryptedStr = new TextDecoder().decode(aesDecryptedUint8Array);
    checkCodec("AES Decrypt(Encrypt(SecretData))", aesDecryptedStr, aesPlain);

    const encTestStr = "<encrypt_method>AES</encrypt_method><encrypt_key>" + aesKey + "</encrypt_key><encrypt>" + aesPlain + "</encrypt>Trailer";
    const encByteList = await textToByteListJs(encTestStr);
    const decStrFromEnc = await byteListToStringJs(encByteList);
    const expectedDecStr = "<encrypt_method>AES</encrypt_method><encrypt_key>" + aesKey + "</encrypt_key>" + aesPlain + "Trailer"; // AES content is plain text here as byteListToStringJs handles decryption
    checkCodec("text -> bytes -> string (with AES)", decStrFromEnc, expectedDecStr);

    const messageSymsForPipeline = byteListToSymbolIndicesJs([72,101,108,108,111]); // "Hello"
    const pipelineTestBytes = symbolIndicesToByteListJs(messageSymsForPipeline);
    const pipelineTestString = await byteListToStringJs(pipelineTestBytes);
    checkCodec("Manual pipeline: syms->bytes->string (Hello)", pipelineTestString, "Hello");

    // New test case for combined tags
    const comboStr = "Data: <linear>3x-6=0</linear> and <base64>SEVMTE8=</base64> then normal text.";
    const expectedComboStr = "Data: <linear>3x-6=0</linear> and HELLO then normal text."; // Assuming base64 is decoded
    const comboByteList = await textToByteListJs(comboStr);
    // Expected byte list for comboStr (manual calculation for verification, not for checkCodec directly)
    // "Data: " -> [68, 97, 116, 97, 58, 32]
    // "<linear>" -> [257]
    // "3x-6=0" -> [51, 120, 45, 54, 61, 48]
    // "</linear>" -> [258]
    // " and " -> [32, 97, 110, 100, 32]
    // "<base64>" -> [281]
    // "SEVMTE8=" -> [83, 69, 86, 77, 84, 69, 56, 61]
    // "</base64>" -> [282]
    // " then normal text." -> [32, 116, 104, 101, 110, 32, 110, 111, 114, 109, 97, 108, 32, 116, 101, 120, 116, 46]
    // This can be used to verify textToByteListJs if needed, but the end-to-end string check is the main goal.
    const comboStrRe = await byteListToStringJs(comboByteList);
    await checkAsyncCodec("textToByteListToString (Combined Tags)", comboStrRe, expectedComboStr);


    console.log(`CustomCodec All Tests Result: ${pass ? 'PASS' : 'FAIL'}`);
    return pass;
}
