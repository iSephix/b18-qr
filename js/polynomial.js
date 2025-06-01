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
            // This case should ideally not be reached if internal functions use it correctly.
            // It was primarily an issue with how the test utility called it.
            console.error("PolynomialGF19.normalize: Input was not an array!", p);
            return [0]; // Or throw error, but for safety return zero polynomial.
        }
        if (p.length === 0) return [0]; // Empty array is zero polynomial

        let d = p.length - 1;
        while (d > 0 && p[d] === 0) {
            d--;
        }

        if (d === 0 && p[0] === 0) return [0]; // All coefficients were zero or single [0]

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
        const len1 = p1.length; const len2 = p2.length; const maxLen = Math.max(len1, len2);
        const result = new Array(maxLen).fill(0);
        for (let i = 0; i < maxLen; i++) {
            const c1 = i < len1 ? p1[i] : 0; const c2 = i < len2 ? p2[i] : 0;
            result[i] = GF19.add(c1, c2);
        }
        return this.normalize(result);
    },

    /** Subtracts polynomial p2 from p1. @param {Array<Number>} p1 @param {Array<Number>} p2 @returns {Array<Number>} Resultant polynomial. */
    subtract: function(p1, p2) {
        const len1 = p1.length; const len2 = p2.length; const maxLen = Math.max(len1, len2);
        const result = new Array(maxLen).fill(0);
        for (let i = 0; i < maxLen; i++) {
            const c1 = i < len1 ? p1[i] : 0; const c2 = i < len2 ? p2[i] : 0;
            result[i] = GF19.subtract(c1, c2);
        }
        return this.normalize(result);
    },

    /** Multiplies a polynomial by a scalar. @param {Array<Number>} p @param {Number} scalar @returns {Array<Number>} Resultant polynomial. */
    multiplyByScalar: function(p, scalar) {
        if (scalar === 0) return [0]; // Multiplying by zero scalar results in zero polynomial
        const norm_p = this.normalize(p);
        if(this.degree(norm_p) === -1) return [0]; // Multiplying zero polynomial by non-zero scalar

        const result = norm_p.map(coeff => GF19.multiply(coeff, scalar));
        return this.normalize(result); // Should be normalized if norm_p was, but good practice.
    },

    /** Multiplies two polynomials. @param {Array<Number>} p1 @param {Array<Number>} p2 @returns {Array<Number>} Resultant polynomial. */
    multiply: function(p1, p2) {
        const norm_p1 = this.normalize(p1);
        const norm_p2 = this.normalize(p2);

        const deg1 = this.degree(norm_p1);
        const deg2 = this.degree(norm_p2);

        if (deg1 === -1 || deg2 === -1) return [0]; // If either is zero polynomial, product is zero

        const resultDegree = deg1 + deg2;
        const result = new Array(resultDegree + 1).fill(0);
        for (let i = 0; i <= deg1; i++) {
            if(norm_p1[i] === 0) continue; // Optimization
            for (let j = 0; j <= deg2; j++) {
                if(norm_p2[j] === 0) continue; // Optimization
                const termProduct = GF19.multiply(norm_p1[i], norm_p2[j]);
                result[i + j] = GF19.add(result[i + j], termProduct);
            }
        }
        return this.normalize(result); // Normalize, though it should be if inputs were normal
    },

    /** Evaluates a polynomial at a given point x. @param {Array<Number>} p @param {Number} x @returns {Number} Result of evaluation. */
    evaluate: function(p, x) {
        let result = 0;
        // Horner's method, assumes p is [c0, c1, ..., cn] where cn is highest degree
        for (let i = p.length - 1; i >= 0; i--) {
            result = GF19.add(GF19.multiply(result, x), p[i]);
        }
        return result;
    },

    /** Divides polynomial dividend by divisor. @param {Array<Number>} dividend @param {Array<Number>} divisor @returns {Object} {quotient, remainder}. */
    divide: function(dividend, divisor) {
        let normDividend = this.normalize([...dividend]); // Use a copy
        const normDivisor = this.normalize([...divisor]);
        if (this.degree(normDivisor) === -1) throw new Error("PolynomialGF19.divide: Division by zero polynomial.");

        let degDividend = this.degree(normDividend);
        const degDivisor = this.degree(normDivisor);

        if (degDividend < degDivisor) return { quotient: [0], remainder: normDividend };

        const quotient = new Array(degDividend - degDivisor + 1).fill(0);
        const leadDivisorCoeff = normDivisor[degDivisor];
        const invLeadDivisorCoeff = GF19.inverse(leadDivisorCoeff);

        while (degDividend >= degDivisor && this.degree(normDividend) !== -1) {
            const currentLeadDividendCoeff = normDividend[degDividend];
            const scale = GF19.multiply(currentLeadDividendCoeff, invLeadDivisorCoeff);
            const currentQuotientDegree = degDividend - degDivisor;
            quotient[currentQuotientDegree] = scale;

            for (let i = 0; i <= degDivisor; i++) {
                if (normDivisor[i] === 0 && scale === 0) continue;
                const termToSubtract = GF19.multiply(scale, normDivisor[i]);
                normDividend[currentQuotientDegree + i] = GF19.subtract(normDividend[currentQuotientDegree + i], termToSubtract);
            }
            // Efficiently update degree by finding new highest non-zero coefficient
            let newDeg = normDividend.length -1;
            while(newDeg >= 0 && normDividend[newDeg] === 0) newDeg--;
            if(newDeg < 0) { // all coefficients became zero
                normDividend = [0];
            } else {
                normDividend.length = newDeg + 1; // truncate trailing zeros
            }
            degDividend = this.degree(normDividend); // Re-calculate degree based on potentially shortened array
        }
        return { quotient: this.normalize(quotient), remainder: normDividend };
    },

    /** Multiplies a polynomial by x^m. @param {Array<Number>} p @param {Number} m - Power, non-negative. @returns {Array<Number>} Resultant polynomial. */
    multiplyByXPowerM: function(p, m) {
        if (m < 0) throw new Error("PolynomialGF19.multiplyByXPowerM: Power m must be non-negative.");
        const normP = this.normalize(p);
        if (this.degree(normP) === -1) return [0]; // 0 * x^m = 0
        if (m === 0) return normP;

        const result = new Array(m).fill(0);
        result.push(...normP);
        return result; // Already normalized if normP was normalized and non-zero
    },

    /** Calculates the derivative of a polynomial. @param {Array<Number>} p @returns {Array<Number>} Derivative polynomial. */
    derivative: function(p) {
        const normP = this.normalize(p);
        if (this.degree(normP) <= 0) return [0]; // Derivative of constant or zero is zero
        const derivCoeffs = [];
        for (let i = 1; i < normP.length; i++) {
            derivCoeffs.push(GF19.multiply(normP[i], i));
        }
        return this.normalize(derivCoeffs); // Normalize, though it should be
    }
};

// Test function for PolynomialGF19
function testPolynomialGF19() {
    if (typeof GF19 === 'undefined') { console.error("PolynomialGF19 tests: GF19 not loaded!"); return false; }
    console.log("--- PolynomialGF19 Tests (Corrected) ---");
    let pass = true;

    const checkValue = (desc, actual, expected) => {
        const success = JSON.stringify(actual) === JSON.stringify(expected);
        console.log(`PolyGF19 Test (Value): ${desc} - Expected: ${expected}, Got: ${actual}. ${success ? 'PASS' : 'FAIL'}`);
        if (!success) pass = false;
    };

    const checkPoly = (desc, actual, expected) => {
        const actualNorm = Array.isArray(actual) ? PolynomialGF19.normalize(actual) : actual; // Normalize if it's an array
        const expectedNorm = PolynomialGF19.normalize(expected); // Expected is always an array

        const success = JSON.stringify(actualNorm) === JSON.stringify(expectedNorm);
        console.log(`PolyGF19 Test (Poly): ${desc} - Expected: ${expectedNorm}, Got: ${actualNorm}. ${success ? 'PASS' : 'FAIL'}`);
        if (!success) pass = false;
    };

    const p1 = [1, 2]; // 1 + 2x
    const p2 = [3, 0, 1]; // 3 + x^2
    const pZero = [0];

    checkValue("degree(p1)", PolynomialGF19.degree(p1), 1);
    checkValue("degree(p2)", PolynomialGF19.degree(p2), 2);
    checkValue("degree([0,0,0])", PolynomialGF19.degree([0,0,0]), -1);
    checkValue("degree([5])", PolynomialGF19.degree([5]), 0);
    checkValue("degree([])", PolynomialGF19.degree([]), -1); // Empty array should be zero poly


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

    const divRes = PolynomialGF19.divide([3,6,1,2], [1,2]);
    checkPoly("divide_quotient", divRes.quotient, [3,0,1]);
    checkPoly("divide_remainder", divRes.remainder, [0]);

    const divRes2 = PolynomialGF19.divide([1,2,3], [1,1]); // (x^2+2x+1)/(x+1) = x+1
    checkPoly("divide_quotient2", divRes2.quotient, [1,1]); // x+1
    checkPoly("divide_remainder2", divRes2.remainder, [0]);


    checkPoly("derivative([1,2,3])", PolynomialGF19.derivative([1,2,3]), [2,6]);
    checkPoly("derivative([5])", PolynomialGF19.derivative([5]), [0]);
    checkPoly("derivative([0])", PolynomialGF19.derivative([0]), [0]);
    checkPoly("derivative([1,0,0,5])", PolynomialGF19.derivative([1,0,0,5]), [0,0,GF19.multiply(5,3)]); // 15x^2 = [0,0,15]

    checkPoly("normalize([0,0,1,0])", PolynomialGF19.normalize([0,0,1,0]), [0,0,1]);
    checkPoly("normalize([0,0,0])", PolynomialGF19.normalize([0,0,0]), [0]);
    checkPoly("normalize([])", PolynomialGF19.normalize([]), [0]);
    checkPoly("normalize([1])", PolynomialGF19.normalize([1]), [1]);
    checkPoly("normalize([0,1])", PolynomialGF19.normalize([0,1]), [0,1]);


    console.log(`PolynomialGF19 All Tests Result: ${pass ? 'PASS' : 'FAIL'}`);
    return pass;
}

/**
 * @fileoverview Provides polynomial arithmetic operations over GF(19).
 * Polynomials are represented as arrays of coefficients, e.g., [c0, c1, c2] for c0 + c1*x + c2*x^2.
 * Depends on GF19 object from gf19.js.
 * @namespace PolynomialGF19
 */
