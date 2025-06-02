// js/reedsolomon.js
// Depends on GF19 from gf19.js and PolynomialGF19 from polynomial.js

/**
 * @fileoverview Reed-Solomon Encoder and Decoder over GF(19).
 * Depends on GF19 and PolynomialGF19.
 */

/**
 * Reed-Solomon Encoder over GF(19).
 * Handles message encoding by adding parity symbols.
 */
class ReedSolomonEncoderGF19 {
    /**
     * Creates an encoder instance.
     * @param {Number} numParitySymbols - The number of parity symbols (n-k) to add. This determines the error correction capability.
     * @throws Error if numParitySymbols is not positive.
     */
    constructor(numParitySymbols) {
        if (numParitySymbols <= 0) throw new Error("RS Encoder: Number of parity symbols must be positive.");
        this.numParitySymbols = numParitySymbols;
        this.generatorPolynomial = this.generateGeneratorPolynomial(numParitySymbols);
        // console.info(`RS Encoder Initialized. g(x) = [${this.generatorPolynomial.join(',')}] (coeffs for x^0, x^1, ...)`);
    }

    /**
     * Generates the Reed-Solomon generator polynomial g(x) = (x-a^0)(x-a^1)...(x-a^{d-1}).
     * @param {Number} numSymbols - Number of roots, equal to numParitySymbols.
     * @returns {Array<Number>} Coefficients of g(x) in ascending order of powers (e.g., [c0, c1, ..., cd]).
     */
    generateGeneratorPolynomial(numSymbols) {
        const primitive = GF19.primitiveElement;
        let g = [1]; // Start with polynomial '1'
        for (let i = 0; i < numSymbols; i++) {
            const rootVal = GF19.power(primitive, i);
            const term = [GF19.subtract(0, rootVal), 1]; // Represents (x - rootVal) as [-rootVal, 1]
            g = PolynomialGF19.multiply(g, term);
        }
        return PolynomialGF19.normalize(g);
    }

    /**
     * Encodes a message block by adding parity symbols using systematic encoding.
     * Message is m(x), codeword c(x) = m(x)*x^d - (m(x)*x^d mod g(x)).
     * The output is [message_symbols..., parity_symbols...].
     * @param {Array<Number>} messageSymbolIndices - The message block (k symbols, each 0-17). Coefficients are [m0, m1, ... mk-1].
     * @returns {Array<Number>} The encoded codeword (n = k + numParitySymbols symbols).
     * @throws Error if messageSymbolIndices is empty.
     */
    encode(messageSymbolIndices) {
        if (!messageSymbolIndices || messageSymbolIndices.length === 0) throw new Error("RS Encoder: Message cannot be empty.");
        const m_poly = [...messageSymbolIndices]; // Polynomial is [m0, m1*x, m2*x^2...]

        const shiftedMessagePoly = PolynomialGF19.multiplyByXPowerM(m_poly, this.numParitySymbols);
        const { remainder } = PolynomialGF19.divide(shiftedMessagePoly, this.generatorPolynomial);

        let finalParityCoeffs = PolynomialGF19.multiplyByScalar(remainder, GF19.subtract(0,1)); // p(x) = -remainder(x)
        finalParityCoeffs = PolynomialGF19.normalize(finalParityCoeffs);

        // Ensure parity polynomial has exactly numParitySymbols coefficients [p0, p1, ..., p_{d-1}]
        const currentParityDegree = PolynomialGF19.degree(finalParityCoeffs);
        let correctlySizedParity = new Array(this.numParitySymbols).fill(0);
        for(let i=0; i <= currentParityDegree; ++i){
            if(i < this.numParitySymbols) { // Should always be true if logic is correct
                correctlySizedParity[i] = finalParityCoeffs[i];
            }
        }

        return [...messageSymbolIndices, ...correctlySizedParity];
    }
}

/**
 * Reed-Solomon Decoder over GF(19).
 * Capable of correcting up to t = numParitySymbols / 2 errors.
 */
class ReedSolomonDecoderGF19 {
    /**
     * Creates a decoder instance.
     * @param {Number} numParitySymbols - The number of parity symbols (2t or d) in the codeword.
     * @param {Number} codewordLengthN - The total length (n) of a codeword.
     * @throws Error if parameters are invalid.
     */
    constructor(numParitySymbols, codewordLengthN) {
        if (numParitySymbols <= 0) throw new Error("RS Decoder: Number of parity symbols must be positive.");
        if (codewordLengthN <= numParitySymbols) throw new Error("RS Decoder: Codeword length N must be greater than numParitySymbols.");
        this.numParitySymbols = numParitySymbols;
        this.codewordLengthN = codewordLengthN;
        this.primitive = GF19.primitiveElement;
        this.DEBUG = false;
    }

    /**
     * Calculates syndromes for a received codeword. S_j = R(alpha^j).
     * @param {Array<Number>} receivedCodewordSymbols - The received codeword [c0, c1, ..., cN-1].
     * @returns {Array<Number>} List of syndromes [S0, S1, ..., S_{d-1}] as coefficients of S(x).
     */
    calculateSyndromes(receivedCodewordSymbols) {
        const syndromes = [];
        if (this.DEBUG) console.log(`RS Decode LOG: Calculating Syndromes for received: [${receivedCodewordSymbols.join(',')}]`);
        const receivedPoly = [...receivedCodewordSymbols];
        for (let j = 0; j < this.numParitySymbols; j++) {
            const root = GF19.power(this.primitive, j);
            const syndromeVal = PolynomialGF19.evaluate(receivedPoly, root);
            syndromes.push(syndromeVal);
        }
        if (this.DEBUG) console.log(`RS Decode LOG: Syndromes S(x) coeffs [S0..Sd-1]: [${syndromes.join(',')}]`);
        return syndromes;
    }
    /**
     * Implements Berlekamp-Massey algorithm to find error locator polynomial sigma(x).
     * @param {Array<Number>} syndromes - Syndromes [S0, S1, ..., S_{d-1}].
     * @returns {Array<Number>} Coefficients of sigma(x) [sigma0, sigma1, ...].
     */
    berlekampMassey(syndromes) {
        if (this.DEBUG) console.log(`RS Decode LOG: Berlekamp-Massey input syndromes: [${syndromes.join(',')}]`);
        let C = [1]; let B = [1]; let L = 0; let m = 1; let b = 1;
        for (let n = 0; n < syndromes.length; n++) {
            let discrepancy = syndromes[n];
            for (let i = 1; i <= L; i++) {
                if (C[i] !== undefined && (n - i) >= 0 && syndromes[n-i] !== undefined) {
                    discrepancy = GF19.add(discrepancy, GF19.multiply(C[i], syndromes[n - i]));
                }
            }
            discrepancy = (Array.isArray(discrepancy) && discrepancy.length > 0) ? discrepancy[0] : discrepancy;
            discrepancy = (Array.isArray(discrepancy) && discrepancy.length === 0) ? 0 : discrepancy; // Handle if it becomes []
            discrepancy = (discrepancy === undefined) ? 0 : discrepancy; // Ensure it's a number

            if (this.DEBUG) console.log(`  BM iter n=${n}: L=${L}, m=${m}, b=${b}, C=[${C.join(',')}], B=[${B.join(',')}], discrepancy=${discrepancy}`);

            if (discrepancy === 0) { m = m + 1; }
            else {
                const C_old = [...C];
                const term_coeff = GF19.divide(discrepancy, b);
                let B_scaled_shifted = PolynomialGF19.multiplyByScalar(B, term_coeff);
                B_scaled_shifted = PolynomialGF19.multiplyByXPowerM(B_scaled_shifted, m);
                C = PolynomialGF19.subtract(C, B_scaled_shifted);
                C = PolynomialGF19.normalize(C);
                if (2 * L <= n) { L = n + 1 - L; B = C_old; b = discrepancy; m = 1; }
                else { m = m + 1; }
            }
        }
        if (this.DEBUG) console.log(`RS Decode LOG: Berlekamp-Massey output sigma C(x): [${C.join(',')}]`);
        return C; // Already normalized in loop
    }
    /**
     * Finds error locations (indices) using Chien search by finding roots of sigma(x).
     * Roots are of the form alpha^(-i), where i is the error position.
     * @param {Array<Number>} sigma - Error locator polynomial coefficients.
     * @returns {Array<Number>} Sorted list of error positions (0-indexed from left of codeword).
     */
    chienSearch(sigma) {
        if (this.DEBUG) console.log(`RS Decode LOG: Chien Search input sigma: [${sigma.join(',')}]`);
        const errorLocationsIndices = [];
        const N = this.codewordLengthN;
        const fieldOrderMinus1 = GF19.FIELD_SIZE - 1;
        for (let i = 0; i < N; i++) {
            const alpha_val_for_eval = GF19.power(this.primitive, i); // Test sigma(alpha^i)
            const evalResult = PolynomialGF19.evaluate(sigma, alpha_val_for_eval);
            if (evalResult === 0) { // If sigma(alpha^i) == 0, then alpha^i is a root.
                                  // This root X_j_inv = alpha^i. Error position is j = -i mod (p-1)
                const errorPosition = (fieldOrderMinus1 - i + fieldOrderMinus1) % fieldOrderMinus1;
                errorLocationsIndices.push(errorPosition);
                if (this.DEBUG) console.log(`  Chien Search: sigma(alpha^${i}=${alpha_val_for_eval}) = 0. Root Xj_inv = alpha^${i}. Error at pos j = (-${i}) mod ${fieldOrderMinus1} => ${errorPosition}`);
            }
        }
        if (PolynomialGF19.degree(sigma) !== errorLocationsIndices.length && PolynomialGF19.degree(sigma) > 0) {
             console.warn(`RS Decode WARN: Chien Search: Num roots found (${errorLocationsIndices.length}) != deg(sigma) (${PolynomialGF19.degree(sigma)}).`);
        }
        if (this.DEBUG) console.log(`RS Decode LOG: Chien Search output error_locations (0-indexed from left/coeff0): [${errorLocationsIndices.sort((a,b)=>a-b).join(',')}]`);
        return errorLocationsIndices.sort((a, b) => a - b);
    }

    /**
     * Calculates error evaluator polynomial Omega(x) = S(x) * sigma(x) mod x^d.
     * @param {Array<Number>} syndromesPolyS - Syndromes as coefficients of S(x) = [S0, S1, ...].
     * @param {Array<Number>} sigma - Error locator polynomial sigma(x).
     * @returns {Array<Number>} Coefficients of Omega(x).
     */
    calculateErrorEvaluatorPolynomialOmega(syndromesPolyS, sigma) {
    if (this.DEBUG) console.log(`RS Decode LOG: Omega Calc: input S(x)_coeffs: [${syndromesPolyS.join(',')}], sigma(x)_coeffs: [${sigma.join(',')}]`);
    const product = PolynomialGF19.multiply(syndromesPolyS, sigma);

    // Omega(x) is (S(x) * sigma(x)) mod x^d.
    // product.slice(0, this.numParitySymbols) correctly takes the coefficients
    // corresponding to powers x^0 through x^(d-1).
    let omega_coeffs_mod_xd = product.slice(0, this.numParitySymbols);

    // Ensure omega always has exactly 'numParitySymbols' coefficients, padding with zeros if necessary.
    // This is important if the product S(x)sigma(x) has a degree less than d-1,
    // or if slice itself returns fewer than numParitySymbols elements (e.g. if product is short).
    let final_omega_coeffs = new Array(this.numParitySymbols).fill(0);
    for (let i = 0; i < omega_coeffs_mod_xd.length; i++) {
        // Defensive check, slice should ensure this, but good to be safe.
        if (i < this.numParitySymbols) {
             final_omega_coeffs[i] = omega_coeffs_mod_xd[i];
        }
    }

    if (this.DEBUG) console.log(`RS Decode LOG: Omega Calc: S*sigma product: [${product.join(',')}], Omega(x) (mod x^d): [${final_omega_coeffs.join(',')}]`);
    return final_omega_coeffs;
}

    /**
     * Calculates error magnitudes using Forney's algorithm: e_j = -Omega(X_j_inv) / sigma'(X_j_inv), where X_j_inv is a root of sigma.
     * @param {Array<Number>} sigma - Error locator polynomial sigma(x).
     * @param {Array<Number>} omega - Error evaluator polynomial Omega(x).
     * @param {Array<Number>} errorLocationsIndices - List of 0-indexed error positions.
     * @returns {Array<Number>} List of error magnitudes corresponding to errorLocations.
     * @throws Error if sigma_prime(X_j_inv) is zero.
     */
    forneysAlgorithm(sigma, omega, errorLocationsIndices) {
        if (this.DEBUG) console.log(`RS Decode LOG: Forney: sigma: [${sigma.join(',')}], omega: [${omega.join(',')}], errLocs: [${errorLocationsIndices.join(',')}]`);
        const errorMagnitudes = [];
        const sigma_prime = PolynomialGF19.derivative(sigma);
        if (this.DEBUG) console.log(`RS Decode LOG: Forney: sigma_prime: [${sigma_prime.join(',')}]`);
        const fieldOrderMinus1 = GF19.FIELD_SIZE - 1;

        for (const errLocIdx of errorLocationsIndices) {
            // The error location index 'errLocIdx' (0 to N-1) corresponds to X_j.
            // The root of sigma is X_j_inv = alpha^(-errLocIdx).
            const Xj_inv_exponent = (fieldOrderMinus1 - errLocIdx + fieldOrderMinus1) % fieldOrderMinus1;
            const Xj_inv = GF19.power(this.primitive, Xj_inv_exponent);
            if (this.DEBUG) console.log(`  Forney for errLoc ${errLocIdx}: Xj_inv_exp=${Xj_inv_exponent}, Xj_inv_val=${Xj_inv}`);

            const omega_at_Xj_inv = PolynomialGF19.evaluate(omega, Xj_inv);
            const sigma_prime_at_Xj_inv = PolynomialGF19.evaluate(sigma_prime, Xj_inv);
            if (this.DEBUG) console.log(`  Forney: omega(${Xj_inv})=${omega_at_Xj_inv}, sigma_prime(${Xj_inv})=${sigma_prime_at_Xj_inv}`);

            if (sigma_prime_at_Xj_inv === 0) {
                console.error("RS Decode ERROR: Forney's algorithm: sigma_prime(X_j^-1) is zero, cannot divide.");
                throw new Error("Forney: sigma_prime(X_j^-1) is zero.");
            }
            const numerator = GF19.subtract(0, omega_at_Xj_inv);
            const errorMagnitude = GF19.divide(numerator, sigma_prime_at_Xj_inv);
            errorMagnitudes.push(errorMagnitude);
            if (this.DEBUG) console.log(`  Forney: errMag for loc ${errLocIdx} = ${errorMagnitude}`);
        }
        if (this.DEBUG) console.log(`RS Decode LOG: Forney output errorMagnitudes: [${errorMagnitudes.join(',')}]`);
        return errorMagnitudes;
    }

    /**
     * Corrects errors in a received codeword using located error positions and calculated magnitudes.
     * Corrected_c_j = Received_c_j - e_j.
     * @param {Array<Number>} receivedCodewordPoly - Received codeword as polynomial coefficients [c0, c1, ...].
     * @param {Array<Number>} errorLocationsIndices - 0-indexed error positions.
     * @param {Array<Number>} errorMagnitudes - Corresponding error magnitudes.
     * @returns {Array<Number>} Corrected codeword polynomial.
     * @throws Error if counts mismatch or location is out of bounds.
     */
    correctErrors(receivedCodewordPoly, errorLocationsIndices, errorMagnitudes) {
        const correctedCodeword = [...receivedCodewordPoly];
        if (errorLocationsIndices.length !== errorMagnitudes.length) throw new Error("CorrectErrors: Mismatch between error locations and magnitudes count.");
        if (this.DEBUG) console.log(`RS Decode LOG: CorrectErrors: received: [${receivedCodewordPoly.join(',')}], locs: [${errorLocationsIndices.join(',')}], mags: [${errorMagnitudes.join(',')}]`);
        for (let i = 0; i < errorLocationsIndices.length; i++) {
            const pos = errorLocationsIndices[i];
            const mag = errorMagnitudes[i];
            if (pos >= correctedCodeword.length) {
                console.error(`RS Decode ERROR: CorrectErrors: Error location ${pos} is out of bounds for codeword length ${correctedCodeword.length}`);
                throw new Error("CorrectErrors: Error location out of bounds.");
            }
            correctedCodeword[pos] = GF19.subtract(correctedCodeword[pos], mag);
        }
        if (this.DEBUG) console.log(`RS Decode LOG: CorrectErrors: corrected codeword: [${correctedCodeword.join(',')}]`);
        return correctedCodeword;
    }

    /**
     * Decodes a received codeword block.
     * @param {Array<Number>} receivedCodewordSymbols - The received codeword (n symbols, [c0, c1, ...]).
     * @returns {Array<Number>|null} The decoded message block (k symbols), or null if errors are uncorrectable.
     */
    decode(receivedCodewordSymbols) {
        // For specific debugging of a call, set this.DEBUG = true before calling decode.
        // Example: const rsDecoder = new ReedSolomonDecoderGF19(...); rsDecoder.DEBUG = true; rsDecoder.decode(...);
        if (this.DEBUG) console.log(`RS DECODE START for codeword: [${receivedCodewordSymbols.join(',')}]`);
        if (receivedCodewordSymbols.length !== this.codewordLengthN) {
            console.error(`RS Decode Error: Received block length ${receivedCodewordSymbols.length} != expected N ${this.codewordLengthN}`);
            return null;
        }
        const receivedPoly = [...receivedCodewordSymbols];

        const syndromes = this.calculateSyndromes(receivedPoly);
        if (syndromes.every(s => s === 0)) {
            if (this.DEBUG) console.log("RS Decode INFO: No errors detected (all syndromes zero).");
            return receivedPoly.slice(0, this.codewordLengthN - this.numParitySymbols);
        }
        try {
            const sigma = this.berlekampMassey(syndromes);
            const errorLocations = this.chienSearch(sigma);
            const numErrorsFound = errorLocations.length;
            const sigmaDegree = PolynomialGF19.degree(sigma);

            if (numErrorsFound === 0 && !syndromes.every(s=>s===0) ) {
                 console.warn("RS Decode WARN: Syndromes non-zero, but Chien search found no error locations from sigma:", sigma);
                 return null;
            }
            if (numErrorsFound !== sigmaDegree && sigmaDegree >= 0) { // sigmaDegree can be -1 if sigma is [0]
                 console.warn(`RS Decode WARN: Number of error locations (${numErrorsFound}) from Chien Search does not match degree of sigma (${sigmaDegree}). Sigma: [${sigma.join(',')}]`);
                 return null;
            }

            const maxCorrectableErrors = Math.floor(this.numParitySymbols / 2);
            if (numErrorsFound > maxCorrectableErrors) {
                console.warn(`RS Decode WARN: Found ${numErrorsFound} errors, which is more than correctable capacity of ${maxCorrectableErrors}.`);
                return null;
            }

            const omega = this.calculateErrorEvaluatorPolynomialOmega(syndromes, sigma);
            const errorMagnitudes = this.forneysAlgorithm(sigma, omega, errorLocations);
            const correctedCodeword = this.correctErrors(receivedPoly, errorLocations, errorMagnitudes);

            const finalSyndromes = this.calculateSyndromes(correctedCodeword);
            if (!finalSyndromes.every(s => s === 0)) {
                console.warn("RS Decode WARN: Corrected codeword still has non-zero syndromes. Correction likely failed. Final Syndromes:", finalSyndromes);
                if (this.DEBUG) console.log("RS Decode LOG: Failed Correction Details: Orig S:", syndromes, "Sigma:", sigma, "Omega:", omega, "ErrLocs:", errorLocations, "ErrMags:", errorMagnitudes, "Corrected:", correctedCodeword);
                return null;
            }
            if (this.DEBUG) console.log("RS Decode INFO: Successfully corrected. Final syndromes all zero.");
            return correctedCodeword.slice(0, this.codewordLengthN - this.numParitySymbols);
        } catch (e) {
            console.error("RS Decode Exception:", e.message, e.stack ? e.stack.substring(0,300):"");
            return null;
        } finally {
            // if (this.DEBUG) this.DEBUG = false; // Optionally reset debug flag
        }
    }
}

// Test function for ReedSolomon
function testReedSolomon() {
    if (typeof GF19 === 'undefined' || typeof PolynomialGF19 === 'undefined') {
        console.error("ReedSolomon tests: Dependencies not loaded!"); return false;
    }
    console.log("--- Reed-Solomon Encoder/Decoder Tests (with Debug Flag for 2-error case) ---");
    let pass = true;
    const check = (desc, actual, expected) => {
        const success = JSON.stringify(actual) === JSON.stringify(expected);
        console.log(`RS Test: ${desc} - Expected: ${JSON.stringify(expected)}, Got: ${JSON.stringify(actual)}. ${success ? 'PASS' : 'FAIL'}`);
        if (!success) pass = false;
    };

    const numParity = 4; const N = 18; const K = N - numParity;
    const rsEncoder = new ReedSolomonEncoderGF19(numParity);
    const rsDecoder = new ReedSolomonDecoderGF19(numParity, N);

    const expectedGenPoly = PolynomialGF19.normalize([7, 13, 13, 4, 1]);
    check("Encoder g(x)", rsEncoder.generatorPolynomial, expectedGenPoly);

    const testMessage = Array.from({length: K}, (_, i) => (i + 1)); // [1,2,...,14]
    const encodedMessage = rsEncoder.encode(testMessage);
    check("Encoded message length", encodedMessage.length, N);

    let rootsEvalZero = true;
    const encodedPolyForEval = [...encodedMessage];
    for (let i = 0; i < numParity; i++) {
        const root = GF19.power(GF19.primitiveElement, i);
        if (PolynomialGF19.evaluate(encodedPolyForEval, root) !== 0) { rootsEvalZero = false; break;}
    }
    check("Codeword evaluation at g(x) roots", rootsEvalZero, true);

    let decodedMessage = rsDecoder.decode([...encodedMessage]);
    check("Decode (no error)", decodedMessage, testMessage);

    const encodedWithError1 = [...encodedMessage];
    encodedWithError1[0] = GF19.add(encodedWithError1[0], 5);
    decodedMessage = rsDecoder.decode(encodedWithError1);
    check("Decode (1 error at pos 0)", decodedMessage, testMessage);

    // Enable DEBUG for the specific 2-error case
    console.log("\nRS Test: Decoding 2-error case with DEBUG enabled:");
    rsDecoder.DEBUG = true;
    // Ensure GF19 context is available for this log
    if (typeof GF19 !== 'undefined' && typeof GF19.add !== 'undefined') {
        const R_corrupt_for_S0_debug = [6,2,3,4,5,9,7,8,9,10,11,12,13,14,12,15,15,5]; // As per original log
        let manual_S0_for_R_corrupt_debug = 0;
        for(let val_debug of R_corrupt_for_S0_debug) {
            manual_S0_for_R_corrupt_debug = GF19.add(manual_S0_for_R_corrupt_debug, val_debug);
        }
        console.log("RS Test DEBUG: Manually calculated S0 for R_corrupt [6,2,...] is: " + manual_S0_for_R_corrupt_debug);
    } else {
        console.log("RS Test DEBUG: GF19 or GF19.add not available for manual S0 calculation in testReedSolomon.");
    }
    const encodedWithError2 = [...encodedMessage];
    encodedWithError2[0] = GF19.add(encodedWithError2[0], 5);
    encodedWithError2[5] = GF19.add(encodedWithError2[5], 3);
    console.log("RS Test: Corrupted 2-error codeword:", encodedWithError2.join(','));
    decodedMessage = rsDecoder.decode(encodedWithError2);
    rsDecoder.DEBUG = false; // Disable debug after the test
    check("Decode (2 errors at pos 0, 5)", decodedMessage, testMessage);
    console.log("RS Test: End of 2-error case debug.\n");

    const encodedWithError3 = [...encodedMessage];
    encodedWithError3[0] = GF19.add(encodedWithError3[0], 5);
    encodedWithError3[5] = GF19.add(encodedWithError3[5], 3);
    encodedWithError3[10] = GF19.add(encodedWithError3[10], 7);
    decodedMessage = rsDecoder.decode(encodedWithError3);
    check("Decode (3 errors - uncorrectable)", decodedMessage, null);

    console.log(`ReedSolomon All Tests Result: ${pass ? 'PASS' : 'FAIL'}`);
    return pass;
}

// Appending JSDoc (already present from previous step)
/**
 * @fileoverview Reed-Solomon Encoder and Decoder over GF(19).
 * Depends on GF19 and PolynomialGF19.
 */
