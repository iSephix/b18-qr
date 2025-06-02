// js/polynomial.js
// Depends on GF19 object from gf19.js

/**
 * @fileoverview Provides polynomial arithmetic operations over GF(19).
 * Polynomials are represented as arrays of coefficients, e.g., [c0, c1, c2] for c0 + c1*x + c2*x^2.
 * Depends on GF19 object from gf19.js.
 * @namespace PolynomialGF19
 */
const PolynomialGF19 = {
    /**
     * Normalizes a polynomial by removing leading zero coefficients.
     * Ensures the zero polynomial is represented as [0].
     * @param {Array<Number>} p - Coefficients of the polynomial.
     * @returns {Array<Number>} Normalized polynomial coefficients.
     */
    normalize: function(p) {
        if (!Array.isArray(p)) {
            console.error("PolynomialGF19.normalize: Invalid input type. Expected array, got:", typeof p, p);
            return [0]; // Default to zero polynomial for invalid input type
        }
        if (p.length === 0) return [0]; // Empty array is zero polynomial

        let d = p.length - 1;
        while (d > 0 && p[d] === 0) {
            d--;
        }

        // If all coefficients were zero (e.g., [0,0,0]) or it was just [0]
        if (d === 0 && p[0] === 0) return [0];

        return p.slice(0, d + 1);
    },

    /**
     * Calculates the degree of a polynomial.
     * @param {Array<Number>} p - Coefficients of the polynomial.
     * @returns {Number} Degree of the polynomial (-1 for zero polynomial).
     */
    degree: function(p) {
        const norm_p = this.normalize(p);
        if (norm_p.length === 1 && norm_p[0] === 0) return -1;
        return norm_p.length - 1;
    },

    /** Adds two polynomials. @param {Array<Number>} p1 @param {Array<Number>} p2 @returns {Array<Number>} Resultant polynomial. */
    add: function(p1, p2) {
        const normP1 = this.normalize(p1); const normP2 = this.normalize(p2);
        const len1 = normP1.length; const len2 = normP2.length; const maxLen = Math.max(len1, len2);
        const result = new Array(maxLen).fill(0);
        for (let i = 0; i < maxLen; i++) {
            const c1 = i < len1 ? normP1[i] : 0; const c2 = i < len2 ? normP2[i] : 0;
            result[i] = GF19.add(c1, c2);
        }
        return this.normalize(result); // Final normalize, though intermediate should be fine
    },

    /** Subtracts polynomial p2 from p1. @param {Array<Number>} p1 @param {Array<Number>} p2 @returns {Array<Number>} Resultant polynomial. */
    subtract: function(p1, p2) {
        const normP1 = this.normalize(p1); const normP2 = this.normalize(p2);
        const len1 = normP1.length; const len2 = normP2.length; const maxLen = Math.max(len1, len2);
        const result = new Array(maxLen).fill(0);
        for (let i = 0; i < maxLen; i++) {
            const c1 = i < len1 ? normP1[i] : 0; const c2 = i < len2 ? normP2[i] : 0;
            result[i] = GF19.subtract(c1, c2);
        }
        return this.normalize(result);
    },

    /** Multiplies a polynomial by a scalar. @param {Array<Number>} p @param {Number} scalar @returns {Array<Number>} Resultant polynomial. */
    multiplyByScalar: function(p, scalar) {
        const normP = this.normalize(p);
        if (scalar === 0 || this.degree(normP) === -1) return [0];

        const result = normP.map(coeff => GF19.multiply(coeff, scalar));
        // No need to normalize again if normP was normalized and scalar != 0
        return result;
    },

    /** Multiplies two polynomials. @param {Array<Number>} p1 @param {Array<Number>} p2 @returns {Array<Number>} Resultant polynomial. */
    multiply: function(p1, p2) {
        const normP1 = this.normalize(p1); const normP2 = this.normalize(p2);
        const deg1 = this.degree(normP1); const deg2 = this.degree(normP2);
        if (deg1 === -1 || deg2 === -1) return [0];

        const resultDegree = deg1 + deg2;
        const result = new Array(resultDegree + 1).fill(0);
        for (let i = 0; i <= deg1; i++) {
            if(normP1[i] === 0) continue;
            for (let j = 0; j <= deg2; j++) {
                if(normP2[j] === 0) continue;
                const termProduct = GF19.multiply(normP1[i], normP2[j]);
                result[i + j] = GF19.add(result[i + j], termProduct);
            }
        }
        return this.normalize(result); // Should be normal, but good for safety
    },

    /** Evaluates a polynomial at a given point x. @param {Array<Number>} polyCoeffs @param {Number} x @returns {Number} Result of evaluation. */
    evaluate: function(polyCoeffs, x) {
    let result = 0;
    // Horner's method: P(x) = c0 + x(c1 + x(c2 + ... x(cN)))
    // polyCoeffs is [c0, c1, ..., cN]. Loop cN down to c0.
    // result_old = 0
    // i = N: result = polyCoeffs[N] + result_old*x = cN
    // i = N-1: result = polyCoeffs[N-1] + result_old*x = c(N-1) + cN*x
    // i = 0: result = polyCoeffs[0] + result_old*x = c0 + c1*x + ... + cN*x^N
    // The implementation is: result = 0; for (let i = polyCoeffs.length - 1; i >= 0; i--) { result = GF19.add(GF19.multiply(result, x), polyCoeffs[i]); }
    // Let poly = [c0, c1, c2]. length = 3. N=2.
    // i=2 (coeff c2): result = GF19.add(GF19.multiply(0,x), c2) = c2.
    // i=1 (coeff c1): result = GF19.add(GF19.multiply(c2,x), c1) = c2*x + c1.
    // i=0 (coeff c0): result = GF19.add(GF19.multiply(c2*x+c1,x), c0) = c2*x^2 + c1*x + c0.
    // This is correct.

    // Note: Original file had this.normalize(polyCoeffs) here. User's new code omits it.
    // The S0 debug log below uses polyCoeffs.length, which is consistent with omitting normalization here.
    for (let i = polyCoeffs.length - 1; i >= 0; i--) {
        result = GF19.add(GF19.multiply(result, x), polyCoeffs[i]);
    }

    // Ensure GF19 context is available for logging
    if (typeof GF19 !== 'undefined' && GF19.add) {
        // Specific log for the failing case's S0 calculation (codeword length 18, x=1 for S0)
        if (x === 1 && polyCoeffs.length === 18) {
            let manual_sum_for_debug = 0;
            for(let k_debug = 0; k_debug < polyCoeffs.length; k_debug++) {
                manual_sum_for_debug = GF19.add(manual_sum_for_debug, polyCoeffs[k_debug]);
            }
            // Previous S0 log lines were commented out by prior diff. This is the new active one.
            console.log("DEBUG S0: poly=[" + polyCoeffs.join(',') + "]");
            console.log("DEBUG S0: PolynomialGF19.evaluate (x=1), Horner result: " + result + ", Manual sum (direct): " + manual_sum_for_debug);
        }
    }
    return result;
},

    /** Divides polynomial dividend by divisor. @param {Array<Number>} dividend @param {Array<Number>} divisor @returns {Object} {quotient, remainder}. */
    divide: function(dividend, divisor) {
        // console.log(`PolyDivide Start: D=${JSON.stringify(dividend)}, d=${JSON.stringify(divisor)}`);
        let currentDividend = this.normalize([...dividend]);
        const normDivisor = this.normalize([...divisor]);

        if (this.degree(normDivisor) === -1) {
            throw new Error("PolynomialGF19.divide: Division by zero polynomial.");
        }

        let degCurrentDividend = this.degree(currentDividend);
        const degDivisor = this.degree(normDivisor);

        if (degCurrentDividend < degDivisor) {
            // console.log(`PolyDivide: degD (${degCurrentDividend}) < degd (${degDivisor}). R=${JSON.stringify(currentDividend)}`);
            return { quotient: [0], remainder: currentDividend };
        }

        const quotient = new Array(degCurrentDividend - degDivisor + 1).fill(0);
        const leadDivisorCoeff = normDivisor[degDivisor];
        const invLeadDivisorCoeff = GF19.inverse(leadDivisorCoeff);

        // console.log(`PolyDivide Init: normD=${JSON.stringify(currentDividend)}, normd=${JSON.stringify(normDivisor)}, degD=${degCurrentDividend}, degd=${degDivisor}, Q_len=${quotient.length}`);

        while (degCurrentDividend >= degDivisor) {
            const currentLeadCoeff = currentDividend[degCurrentDividend];
            const scale = GF19.multiply(currentLeadCoeff, invLeadDivisorCoeff);
            const currentQuotientDegree = degCurrentDividend - degDivisor;
            quotient[currentQuotientDegree] = scale;

            // console.log(`PolyDivide Loop: degD=${degCurrentDividend}, leadDCoeff=${currentLeadCoeff}, scale=${scale}, currQDeg=${currentQuotientDegree}, Q_interm=${JSON.stringify(quotient)}`);

            for (let i = 0; i <= degDivisor; i++) {
                if (normDivisor[i] === 0 && scale === 0) continue; // Optimization
                const termToSubtract = GF19.multiply(scale, normDivisor[i]);
                const dividendIdx = currentQuotientDegree + i;
                currentDividend[dividendIdx] = GF19.subtract(currentDividend[dividendIdx] || 0, termToSubtract);
            }

            // Crucial: After subtraction, the leading terms of currentDividend might have become zero.
            // We MUST use a normalized version to get the correct degree for the next iteration's check and leading coefficient.
            currentDividend = this.normalize(currentDividend);
            degCurrentDividend = this.degree(currentDividend);
            // console.log(`PolyDivide Loop End: currentD_after_sub_norm=${JSON.stringify(currentDividend)}, new_degD=${degCurrentDividend}`);
        }
        const finalQuotient = this.normalize(quotient);
        const finalRemainder = currentDividend; // Already normalized from the loop's end
        // console.log(`PolyDivide Result: Q=${JSON.stringify(finalQuotient)}, R=${JSON.stringify(finalRemainder)}`);
        return { quotient: finalQuotient, remainder: finalRemainder };
    },

    /** Multiplies a polynomial by x^m. @param {Array<Number>} p @param {Number} m - Power, non-negative. @returns {Array<Number>} Resultant polynomial. */
    multiplyByXPowerM: function(p, m) {
        if (m < 0) throw new Error("PolynomialGF19.multiplyByXPowerM: Power m must be non-negative.");
        const normP = this.normalize(p);
        if (this.degree(normP) === -1) return [0];
        if (m === 0) return normP;

        const result = new Array(m).fill(0);
        result.push(...normP);
        return result;
    },

    /** Calculates the derivative of a polynomial. @param {Array<Number>} p @returns {Array<Number>} Derivative polynomial. */
    derivative: function(p) {
        const normP = this.normalize(p);
        if (this.degree(normP) <= 0) return [0];
        const derivCoeffs = [];
        for (let i = 1; i < normP.length; i++) {
            derivCoeffs.push(GF19.multiply(normP[i], i));
        }
        return this.normalize(derivCoeffs);
    }
};

// Test function for PolynomialGF19
function testPolynomialGF19() {
    if (typeof GF19 === 'undefined') { console.error("PolynomialGF19 tests: GF19 not loaded!"); return false; }
    console.log("--- PolynomialGF19 Tests (Corrected with specific division test) ---");
    let pass = true;

    const checkValue = (desc, actual, expected) => {
        const success = JSON.stringify(actual) === JSON.stringify(expected);
        console.log(`PolyGF19 Test (Value): ${desc} - Expected: ${expected}, Got: ${actual}. ${success ? 'PASS' : 'FAIL'}`);
        if (!success) pass = false;
    };
    const checkPoly = (desc, actual, expected) => {
        const actualNorm = Array.isArray(actual) ? PolynomialGF19.normalize(actual) : actual;
        const expectedNorm = Array.isArray(expected) ? PolynomialGF19.normalize(expected) : expected;
        const success = JSON.stringify(actualNorm) === JSON.stringify(expectedNorm);
        console.log(`PolyGF19 Test (Poly): ${desc} - Expected: ${JSON.stringify(expectedNorm)}, Got: ${JSON.stringify(actualNorm)}. ${success ? 'PASS' : 'FAIL'}`);
        if (!success) pass = false;
    };

    const p1 = [1, 2]; // 1 + 2x
    const p2 = [3, 0, 1]; // 3 + x^2
    const pZero = [0];

    checkValue("degree(p1)", PolynomialGF19.degree(p1), 1);
    checkValue("degree(p2)", PolynomialGF19.degree(p2), 2);
    checkValue("degree([0,0,0])", PolynomialGF19.degree([0,0,0]), -1);
    checkValue("degree([5])", PolynomialGF19.degree([5]), 0);
    checkValue("degree([])", PolynomialGF19.degree([]), -1);


    checkPoly("add(p1,p2)", PolynomialGF19.add(p1,p2), [4,2,1]);
    checkPoly("subtract(p1,p2)", PolynomialGF19.subtract(p1,p2), [17,2,18]);
    checkPoly("multiplyByScalar(p1,3)", PolynomialGF19.multiplyByScalar(p1,3), [3,6]);
    checkPoly("multiplyByScalar(p1,0)", PolynomialGF19.multiplyByScalar(p1,0), [0]);
    checkPoly("multiplyByScalar(pZero,5)", PolynomialGF19.multiplyByScalar(pZero,5), [0]);
    checkPoly("multiply(p1,p2)", PolynomialGF19.multiply(p1,p2), [3,6,1,2]);
    checkPoly("multiply(p1,pZero)", PolynomialGF19.multiply(p1,pZero), [0]);
    checkPoly("multiply(pZero,p2)", PolynomialGF19.multiply(pZero,p2), [0]);

    checkValue("evaluate(p1,5)", PolynomialGF19.evaluate(p1,5), 11);
    checkValue("evaluate(p2,3)", PolynomialGF19.evaluate(p2,3), 12);

    const divRes1 = PolynomialGF19.divide([3,6,1,2], [1,2]);
    checkPoly("divide_quotient1", divRes1.quotient, [3,0,1]);
    checkPoly("divide_remainder1", divRes1.remainder, [0]);

    const divDividend2 = [18, 0, 1]; // x^2 + 18
    const divDivisor2  = [18, 1];   // x + 18
    const divRes2 = PolynomialGF19.divide(divDividend2, divDivisor2);
    checkPoly("divide_quotient2 (x^2+18)/(x+18)", divRes2.quotient, [1, 1]); // Expected: x + 1
    checkPoly("divide_remainder2 (x^2+18)/(x+18)", divRes2.remainder, [0]);

    const divRes3 = PolynomialGF19.divide([1,1,1], [1,1]); // (x^2+x+1)/(x+1)
    checkPoly("divide_quotient3", divRes3.quotient, [0,1]); // Expected: x
    checkPoly("divide_remainder3", divRes3.remainder, [1]); // Expected: 1

    const divRes4 = PolynomialGF19.divide([1,2,3,4,5], [1,2,3]); // (5x^4+4x^3+3x^2+2x+1)/(3x^2+2x+1)
    // 5x^4+4x^3+3x^2+2x+1 = (3x^2+2x+1)( (5/3)x^2 + (2/9)x + (2/27) ) + ( (28/27)x + (25/27) )
    // In GF(19): 5/3 = 5*13=65=8. 2/9=2*17=34=15. 2/27=2*17*13=15*13=195=5.
    // Q_gf19 = 8x^2+15x+5 => [5,15,8]
    // R_gf19: 28/27 = (28%19) / (27%19) = 9 / 8 = 9*12=108=13. (13x)
    //         25/27 = (25%19) / (27%19) = 6 / 8  = 6*12=72=15. (+15)
    // R_gf19 = 13x+15 => [15,13]
    const divRes4_expected_Q = [5,15,8];
    const divRes4_expected_R = [15,13];
    const divRes4_actual = PolynomialGF19.divide([1,2,3,4,5], [1,2,3]);
    checkPoly("divide_quotient4", divRes4_actual.quotient, divRes4_expected_Q);
    checkPoly("divide_remainder4", divRes4_actual.remainder, divRes4_expected_R);


    checkPoly("derivative([1,2,3])", PolynomialGF19.derivative([1,2,3]), [2,6]);
    checkPoly("derivative([5])", PolynomialGF19.derivative([5]), [0]);
    checkPoly("derivative([0])", PolynomialGF19.derivative([0]), [0]);
    checkPoly("derivative([1,0,0,5])", PolynomialGF19.derivative([1,0,0,5]), [0,0,15]);

    checkPoly("normalize([0,0,1,0])", PolynomialGF19.normalize([0,0,1,0]), [0,0,1]);
    checkPoly("normalize([0,0,0])", PolynomialGF19.normalize([0,0,0]), [0]);
    checkPoly("normalize([])", PolynomialGF19.normalize([]), [0]);
    checkPoly("normalize([1])", PolynomialGF19.normalize([1]), [1]);
    checkPoly("normalize([0,1])", PolynomialGF19.normalize([0,1]), [0,1]);

    console.log(`PolynomialGF19 All Tests Result: ${pass ? 'PASS' : 'FAIL'}`);
    return pass;
}

// Appending JSDoc (already present from previous step)
/**
 * @fileoverview Provides polynomial arithmetic operations over GF(19).
 * Depends on GF19 object.
 * @object PolynomialGF19
 */
