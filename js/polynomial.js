// js/polynomial.js
// Depends on GF19 object from gf19.js

const PolynomialGF19 = {
    normalize: function(p) {
        if (!p || p.length === 0) return [0];
        let d = p.length - 1;
        while (d > 0 && p[d] === 0) { d--; }
        if (d === 0 && p[0] === 0) return [0];
        return p.slice(0, d + 1);
    },
    degree: function(p) {
        const norm_p = this.normalize(p);
        if (norm_p.length === 1 && norm_p[0] === 0) return -1; // Degree of zero polynomial
        return norm_p.length - 1;
    },
    add: function(p1, p2) {
        const norm_p1 = this.normalize(p1); const norm_p2 = this.normalize(p2);
        const len1 = norm_p1.length; const len2 = norm_p2.length; const maxLen = Math.max(len1, len2);
        const result = new Array(maxLen).fill(0);
        for (let i = 0; i < maxLen; i++) {
            const c1 = i < len1 ? norm_p1[i] : 0; const c2 = i < len2 ? norm_p2[i] : 0;
            result[i] = GF19.add(c1, c2);
        }
        return this.normalize(result);
    },
    subtract: function(p1, p2) {
        const norm_p1 = this.normalize(p1); const norm_p2 = this.normalize(p2);
        const len1 = norm_p1.length; const len2 = norm_p2.length; const maxLen = Math.max(len1, len2);
        const result = new Array(maxLen).fill(0);
        for (let i = 0; i < maxLen; i++) {
            const c1 = i < len1 ? norm_p1[i] : 0; const c2 = i < len2 ? norm_p2[i] : 0;
            result[i] = GF19.subtract(c1, c2);
        }
        return this.normalize(result);
    },
    multiplyByScalar: function(p, scalar) {
        const norm_p = this.normalize(p);
        if (scalar === 0 || (norm_p.length === 1 && norm_p[0] === 0)) return [0];
        const result = norm_p.map(coeff => GF19.multiply(coeff, scalar));
        return this.normalize(result); // Should already be normal if p was normal and scalar != 0
    },
    multiply: function(p1, p2) {
        const norm_p1 = this.normalize(p1); const norm_p2 = this.normalize(p2);
        if ((norm_p1.length === 1 && norm_p1[0] === 0) || (norm_p2.length === 1 && norm_p2[0] === 0)) return [0];
        const deg1 = this.degree(norm_p1); const deg2 = this.degree(norm_p2);
        const resultDegree = deg1 + deg2; const result = new Array(resultDegree + 1).fill(0);
        for (let i = 0; i <= deg1; i++) {
            for (let j = 0; j <= deg2; j++) {
                if (norm_p1[i] === 0 || norm_p2[j] === 0) continue; // Optimization
                const termProduct = GF19.multiply(norm_p1[i], norm_p2[j]);
                result[i + j] = GF19.add(result[i + j], termProduct);
            }
        }
        return this.normalize(result);
    },
    evaluate: function(p, x) {
        const norm_p = this.normalize(p);
        let result = 0;
        // Evaluate using Horner's method (from highest degree to lowest for array indexing)
        for (let i = norm_p.length - 1; i >= 0; i--) {
            result = GF19.add(GF19.multiply(result, x), norm_p[i]);
        }
        return result;
    },
    divide: function(dividend, divisor) {
        let normDividend = this.normalize([...dividend]); // Use a copy for mutation
        const normDivisor = this.normalize([...divisor]);
        if (normDivisor.length === 1 && normDivisor[0] === 0) throw new Error("PolynomialGF19.divide: Division by zero polynomial.");

        let degDividend = this.degree(normDividend);
        const degDivisor = this.degree(normDivisor);

        if (degDividend < degDivisor) return { quotient: [0], remainder: normDividend };

        const quotient = new Array(degDividend - degDivisor + 1).fill(0);
        const leadDivisorCoeff = normDivisor[degDivisor];
        const invLeadDivisorCoeff = GF19.inverse(leadDivisorCoeff);

        while (degDividend >= degDivisor && !(normDividend.length === 1 && normDividend[0] === 0) ) {
            const currentLeadDividendCoeff = normDividend[degDividend];
            const scale = GF19.multiply(currentLeadDividendCoeff, invLeadDivisorCoeff);
            const currentQuotientDegree = degDividend - degDivisor;
            quotient[currentQuotientDegree] = scale;

            for (let i = 0; i <= degDivisor; i++) {
                if (normDivisor[i] === 0 && scale === 0) continue; // Optimization
                const termToSubtract = GF19.multiply(scale, normDivisor[i]);
                normDividend[currentQuotientDegree + i] = GF19.subtract(normDividend[currentQuotientDegree + i], termToSubtract);
            }
            // Re-normalize dividend and update its degree
            // Avoid creating new array in loop: find new degree by scanning from high terms
            let newDeg = normDividend.length - 1;
            while(newDeg >=0 && normDividend[newDeg] === 0) newDeg--;
            if(newDeg < 0) normDividend = [0]; // All coeffs became zero
            else normDividend.length = newDeg + 1; // Truncate trailing zeros
            degDividend = this.degree(normDividend);
        }
        return { quotient: this.normalize(quotient), remainder: normDividend };
    },
    multiplyByXPowerM: function(p, m) {
        if (m < 0) throw new Error("PolynomialGF19.multiplyByXPowerM: Power m must be non-negative.");
        if (m === 0) return this.normalize([...p]);
        const normP = this.normalize(p);
        if (normP.length === 1 && normP[0] === 0) return [0];
        const result = new Array(m).fill(0);
        result.push(...normP);
        return result; // Already normalized if normP was normalized and non-zero
    },
    derivative: function(p) {
        const normP = this.normalize(p);
        if (normP.length <= 1) return [0];
        const derivCoeffs = [];
        for (let i = 1; i < normP.length; i++) {
            derivCoeffs.push(GF19.multiply(normP[i], i));
        }
        return this.normalize(derivCoeffs);
    }
};
// Appending tests to js/polynomial.js

function testPolynomialGF19() {
    if (typeof GF19 === 'undefined') { console.error("PolynomialGF19 tests: GF19 not loaded!"); return false; }
    console.log("--- PolynomialGF19 Tests ---");
    let pass = true;
    const checkPoly = (desc, actual, expected) => {
        const actualNorm = PolynomialGF19.normalize(actual);
        const expectedNorm = PolynomialGF19.normalize(expected);
        const success = JSON.stringify(actualNorm) === JSON.stringify(expectedNorm);
        console.log(`PolyGF19 Test: ${desc} - Expected: ${expectedNorm}, Got: ${actualNorm}. ${success ? 'PASS' : 'FAIL'}`);
        if (!success) pass = false;
    };

    const p1 = [1, 2]; // 1 + 2x
    const p2 = [3, 0, 1]; // 3 + x^2
    const pZero = [0];
    checkPoly("degree(p1)", PolynomialGF19.degree(p1), 1);
    checkPoly("add(p1,p2)", PolynomialGF19.add(p1,p2), [4,2,1]); // 4+2x+x^2
    checkPoly("subtract(p1,p2)", PolynomialGF19.subtract(p1,p2), [17,2,18]); // (1-3) + 2x - x^2 = 17+2x+18x^2
    checkPoly("multiplyByScalar(p1,3)", PolynomialGF19.multiplyByScalar(p1,3), [3,6]);
    checkPoly("multiply(p1,p2)", PolynomialGF19.multiply(p1,p2), [3,6,1,2]); // (1+2x)(3+x^2)=3+6x+x^2+2x^3
    checkPoly("evaluate(p1,5)", PolynomialGF19.evaluate(p1,5), 11); // 1+2*5 = 11
    const divRes = PolynomialGF19.divide([3,6,1,2], [1,2]); // (3+6x+x^2+2x^3)/(1+2x) = x^2+3
    checkPoly("divide_quotient", divRes.quotient, [3,0,1]);
    checkPoly("divide_remainder", divRes.remainder, [0]);
    checkPoly("derivative([1,2,3])", PolynomialGF19.derivative([1,2,3]), [2,6]); // d/dx (1+2x+3x^2) = 2+6x
    console.log(`PolynomialGF19 All Tests Result: ${pass ? 'PASS' : 'FAIL'}`);
    return pass;
}
// Appending JSDoc to js/polynomial.js

/**
 * @fileoverview Provides polynomial arithmetic operations over GF(19).
 * Polynomials are represented as arrays of coefficients, e.g., [c0, c1, c2] for c0 + c1*x + c2*x^2.
 * Depends on GF19 object from gf19.js.
 * @namespace PolynomialGF19
 */
