// js/reedsolomon.js
// Depends on GF19 from gf19.js and PolynomialGF19 from polynomial.js

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
        // console.info("RS Encoder Initialized. Generator Poly:", this.generatorPolynomial);
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
     * @param {Array<Number>} messageSymbolIndices - The message block (k symbols, each 0-17).
     * @returns {Array<Number>} The encoded codeword (n = k + numParitySymbols symbols).
     * @throws Error if messageSymbolIndices is empty.
     */
    encode(messageSymbolIndices) {
        if (!messageSymbolIndices || messageSymbolIndices.length === 0) throw new Error("RS Encoder: Message cannot be empty.");
        // messageSymbolIndices is [m0, m1, ..., mk-1]
        const m_poly = [...messageSymbolIndices]; // Polynomial already in [m0, m1x, m2x^2...] form

        const shiftedMessagePoly = PolynomialGF19.multiplyByXPowerM(m_poly, this.numParitySymbols);
        const { remainder } = PolynomialGF19.divide(shiftedMessagePoly, this.generatorPolynomial);

        // Parity p(x) = -remainder(x). remainder is [r0, r1, ...]
        let finalParityCoeffs = PolynomialGF19.multiplyByScalar(remainder, GF19.subtract(0,1)); // -1 is 18 in GF19
        finalParityCoeffs = PolynomialGF19.normalize(finalParityCoeffs);

        // Ensure parity has correct length (numParitySymbols) by padding with leading zeros if necessary
        // These are lower-degree coefficients for the polynomial p(x)
        while (finalParityCoeffs.length < this.numParitySymbols) {
            finalParityCoeffs.unshift(0);
        }
        finalParityCoeffs = finalParityCoeffs.slice(0, this.numParitySymbols);

        // For systematic encoding, codeword is [message_symbols, parity_symbols]
        // Parity symbols are coefficients of p(x) typically in order p0, p1, ..., pd-1
        return [...messageSymbolIndices, ...finalParityCoeffs];
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
        // console.info(`RS Decoder Initialized: N=${codewordLengthN}, K=${codewordLengthN-numParitySymbols}, ParitySyms=${numParitySymbols}`);
    }

    /**
     * Calculates syndromes for a received codeword. S_j = R(alpha^j).
     * @param {Array<Number>} receivedCodewordSymbols - The received codeword [c0, c1, ..., cN-1].
     * @returns {Array<Number>} List of syndromes [S0, S1, ..., S_{d-1}] as coefficients of S(x).
     */
    calculateSyndromes(receivedCodewordSymbols) {
        const syndromes = []; // S(x) = S0 + S1x + ...
        const receivedPoly = [...receivedCodewordSymbols]; // Assumed [c0, c1, ...]
        for (let j = 0; j < this.numParitySymbols; j++) {
            const root = GF19.power(this.primitive, j);
            const syndromeVal = PolynomialGF19.evaluate(receivedPoly, root);
            syndromes.push(syndromeVal);
        }
        return syndromes;
    }
    /**
     * Implements Berlekamp-Massey algorithm to find error locator polynomial sigma(x).
     * @param {Array<Number>} syndromes - Syndromes [S0, S1, ..., S_{d-1}].
     * @returns {Array<Number>} Coefficients of sigma(x) [sigma0, sigma1, ...].
     */
    berlekampMassey(syndromes) {
        let C = [1]; let B = [1]; let L = 0; let m = 1; let b = 1; // C = sigma(x), B = previous sigma
        for (let n = 0; n < syndromes.length; n++) {
            let discrepancy = syndromes[n]; // S_n
            for (let i = 1; i <= L; i++) {
                if (C[i] !== undefined && (n - i) >= 0 && syndromes[n-i] !== undefined) {
                    discrepancy = GF19.add(discrepancy, GF19.multiply(C[i], syndromes[n - i]));
                }
            }
            // discrepancy should be scalar, normalize if it became an array due to GF19 ops returning array for 0
            discrepancy = (Array.isArray(discrepancy) && discrepancy.length > 0) ? discrepancy[0] : discrepancy;


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
        return PolynomialGF19.normalize(C);
    }
    /**
     * Finds error locations (indices) using Chien search by finding roots of sigma(x).
     * Roots are of the form alpha^(-i), where i is the error position.
     * @param {Array<Number>} sigma - Error locator polynomial coefficients.
     * @returns {Array<Number>} Sorted list of error positions (0-indexed from left of codeword).
     */
    chienSearch(sigma) {
        const errorLocationsIndices = [];
        const N = this.codewordLengthN;
        const fieldOrderMinus1 = GF19.FIELD_SIZE - 1;

        for (let i = 0; i < N; i++) {
            const alpha_i_inv = GF19.power(this.primitive, (fieldOrderMinus1 - i + fieldOrderMinus1) % fieldOrderMinus1 ); // alpha^(-i)
            const evalResult = PolynomialGF19.evaluate(sigma, alpha_i_inv);

            if (evalResult === 0) {
                errorLocationsIndices.push(i); // Position 'i' is an error location
            }
        }
        if (PolynomialGF19.degree(sigma) !== errorLocationsIndices.length && PolynomialGF19.degree(sigma) > 0) {
             console.warn("Chien Search: Number of roots found (" + errorLocationsIndices.length + ") does not match degree of sigma (" + PolynomialGF19.degree(sigma) + "). This may indicate an uncorrectable error pattern.");
        }
        return errorLocationsIndices.sort((a, b) => a - b);
    }

    /**
     * Calculates error evaluator polynomial Omega(x) = S(x) * sigma(x) mod x^d.
     * @param {Array<Number>} syndromesPolyS - Syndromes as coefficients of S(x) = [S0, S1, ...].
     * @param {Array<Number>} sigma - Error locator polynomial sigma(x).
     * @returns {Array<Number>} Coefficients of Omega(x).
     */
    calculateErrorEvaluatorPolynomialOmega(syndromesPolyS, sigma) {
        const product = PolynomialGF19.multiply(syndromesPolyS, sigma);
        let omega = product.slice(0, this.numParitySymbols);
        return PolynomialGF19.normalize(omega);
    }

    /**
     * Calculates error magnitudes using Forney's algorithm: e_j = -Omega(X_j) / sigma'(X_j), where X_j = alpha^(-error_pos_idx).
     * @param {Array<Number>} sigma - Error locator polynomial sigma(x).
     * @param {Array<Number>} omega - Error evaluator polynomial Omega(x).
     * @param {Array<Number>} errorLocationsIndices - List of 0-indexed error positions.
     * @returns {Array<Number>} List of error magnitudes corresponding to errorLocations.
     * @throws Error if sigma_prime(X_j) is zero.
     */
    forneysAlgorithm(sigma, omega, errorLocationsIndices) {
        const errorMagnitudes = [];
        const sigma_prime = PolynomialGF19.derivative(sigma);
        const fieldOrderMinus1 = GF19.FIELD_SIZE - 1;

        for (const errLocIdx of errorLocationsIndices) {
            const Xj_inv_exp = (fieldOrderMinus1 - errLocIdx + fieldOrderMinus1) % fieldOrderMinus1; // exponent for alpha^(-errLocIdx)
            const Xj_inv = GF19.power(this.primitive, Xj_inv_exp);

            const omega_at_Xj_inv = PolynomialGF19.evaluate(omega, Xj_inv);
            const sigma_prime_at_Xj_inv = PolynomialGF19.evaluate(sigma_prime, Xj_inv);

            if (sigma_prime_at_Xj_inv === 0) throw new Error("Forney's algorithm: sigma_prime(X_j^-1) is zero, cannot divide.");

            const numerator = GF19.subtract(0, omega_at_Xj_inv);
            const errorMagnitude = GF19.divide(numerator, sigma_prime_at_Xj_inv);
            errorMagnitudes.push(errorMagnitude);
        }
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

        for (let i = 0; i < errorLocationsIndices.length; i++) {
            const pos = errorLocationsIndices[i];
            const mag = errorMagnitudes[i];
            if (pos >= correctedCodeword.length) throw new Error(`CorrectErrors: Error location ${pos} is out of bounds for codeword length ${correctedCodeword.length}.`);
            correctedCodeword[pos] = GF19.subtract(correctedCodeword[pos], mag);
        }
        return correctedCodeword;
    }

    /**
     * Decodes a received codeword block.
     * Implements the full Peterson-Gorenstein-Zierler algorithm variant (Syndromes, BM, Chien, Forney).
     * @param {Array<Number>} receivedCodewordSymbols - The received codeword (n symbols, [c0, c1, ...]).
     * @returns {Array<Number>|null} The decoded message block (k symbols), or null if errors are uncorrectable.
     */
    decode(receivedCodewordSymbols) {
        if (receivedCodewordSymbols.length !== this.codewordLengthN) {
            console.error(`RS Decode Error: Received block length ${receivedCodewordSymbols.length} != expected N ${this.codewordLengthN}`);
            return null;
        }
        const receivedPoly = [...receivedCodewordSymbols];

        const syndromes = this.calculateSyndromes(receivedPoly);
        if (syndromes.every(s => s === 0)) {
            return receivedPoly.slice(0, this.codewordLengthN - this.numParitySymbols);
        }

        try {
            const sigma = this.berlekampMassey(syndromes);
            const errorLocations = this.chienSearch(sigma);
            const numErrorsFound = errorLocations.length;
            const sigmaDegree = PolynomialGF19.degree(sigma);

            if (numErrorsFound === 0 && !syndromes.every(s=>s===0) ) { console.warn("RS Decode Warning: Non-zero syndromes but no error locations found by Chien Search."); return null; }
            if (numErrorsFound !== sigmaDegree && sigmaDegree > 0) { console.warn("RS Decode Warning: Number of error locations found by Chien Search does not match degree of sigma."); return null; }

            const maxCorrectableErrors = Math.floor(this.numParitySymbols / 2);
            if (numErrorsFound > maxCorrectableErrors) { console.warn(`RS Decode Warning: Found ${numErrorsFound} errors, but can only correct up to ${maxCorrectableErrors}.`); return null; }

            const omega = this.calculateErrorEvaluatorPolynomialOmega(syndromes, sigma);
            const errorMagnitudes = this.forneysAlgorithm(sigma, omega, errorLocations);

            const correctedCodeword = this.correctErrors(receivedPoly, errorLocations, errorMagnitudes);

            const finalSyndromes = this.calculateSyndromes(correctedCodeword);
            if (!finalSyndromes.every(s => s === 0)) {
                console.warn("RS Decode Warning: Corrected codeword still has non-zero syndromes. Correction likely failed.");
                return null;
            }

            return correctedCodeword.slice(0, this.codewordLengthN - this.numParitySymbols);
        } catch (e) {
            console.error("RS Decode Exception:", e.message, e.stack ? e.stack.substring(0,300):"");
            return null;
        }
    }
}

// Test function (from previous step, ensures it's part of the file)
function testReedSolomon() {
    if (typeof GF19 === 'undefined' || typeof PolynomialGF19 === 'undefined') { console.error("ReedSolomon tests: Dependencies not loaded!"); return false; }
    console.log("--- Reed-Solomon Encoder/Decoder Tests ---"); let pass = true;
    const check = (desc, actual, expected) => { const success = JSON.stringify(actual) === JSON.stringify(expected); console.log(`RS Test: ${desc} - Expected: ${expected}, Got: ${actual}. ${success ? 'PASS' : 'FAIL'}`); if (!success) pass = false; };
    const numParity = 4; const N = 18; const K = N - numParity;
    const rsEncoder = new ReedSolomonEncoderGF19(numParity); const rsDecoder = new ReedSolomonDecoderGF19(numParity, N);
    const expectedGenPoly = [7, 13, 13, 4, 1]; check("Encoder g(x)", rsEncoder.generatorPolynomial, expectedGenPoly);
    const testMessage = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    const encodedMessage = rsEncoder.encode(testMessage); check("Encoded message length", encodedMessage.length, N);
    let rootsEvalZero = true; const encodedPolyForEval = [...encodedMessage];
    for (let i = 0; i < numParity; i++) { const root = GF19.power(GF19.primitiveElement, i); if (PolynomialGF19.evaluate(encodedPolyForEval, root) !== 0) { rootsEvalZero = false; break; } }
    check("Codeword evaluation at g(x) roots", rootsEvalZero, true);
    let decodedMessage = rsDecoder.decode([...encodedMessage]); check("Decode (no error)", decodedMessage, testMessage);
    const encodedWithError1 = [...encodedMessage]; encodedWithError1[0] = GF19.add(encodedWithError1[0], 5);
    decodedMessage = rsDecoder.decode(encodedWithError1); check("Decode (1 error at pos 0)", decodedMessage, testMessage);
    const encodedWithError2 = [...encodedMessage]; encodedWithError2[0] = GF19.add(encodedWithError2[0], 5); encodedWithError2[5] = GF19.add(encodedWithError2[5], 3);
    decodedMessage = rsDecoder.decode(encodedWithError2); check("Decode (2 errors at pos 0, 5)", decodedMessage, testMessage);
    const encodedWithError3 = [...encodedMessage]; encodedWithError3[0] = GF19.add(encodedWithError3[0], 5); encodedWithError3[5] = GF19.add(encodedWithError3[5], 3); encodedWithError3[10] = GF19.add(encodedWithError3[10], 7);
    decodedMessage = rsDecoder.decode(encodedWithError3); check("Decode (3 errors - uncorrectable)", decodedMessage, null);
    console.log(`ReedSolomon All Tests Result: ${pass ? 'PASS' : 'FAIL'}`); return pass;
}
