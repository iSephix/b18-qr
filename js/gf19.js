// js/gf19.js
const GF19 = {
    FIELD_SIZE: 19, // Added for clarity
    add: function(a, b) { return (a + b) % 19; },
    subtract: function(a, b) { return (a - b + 19) % 19; },
    multiply: function(a, b) { return (a * b) % 19; },
    inverse: function(a) {
        const a_mod = (a % 19 + 19) % 19;
        if (a_mod === 0) throw new Error("GF19.inverse: Inverse of zero requested.");
        for (let x = 1; x < 19; x++) { if ((a_mod * x) % 19 === 1) return x; }
        throw new Error("GF19.inverse: Inverse not found (should not happen for non-zero in prime field).");
    },
    divide: function(a, b) {
        const b_mod = (b % 19 + 19) % 19;
        if (b_mod === 0) throw new Error("GF19.divide: Division by zero.");
        return this.multiply(a, this.inverse(b_mod));
    },
    power: function(base, exp) {
        let res = 1;
        base = (base % 19 + 19) % 19;
        if (base === 0 && exp === 0) return 1; // Convention 0^0 = 1 for polynomial fields
        if (base === 0 && exp < 0) throw new Error("GF19.power: 0 cannot be raised to a negative power.");
        if (exp < 0) { base = this.inverse(base); exp = -exp; }
        while (exp > 0) {
            if (exp % 2 === 1) res = this.multiply(res, base);
            base = this.multiply(base, base);
            exp = Math.floor(exp / 2);
        }
        return res;
    },
    primitiveElement: 2,
    log: function(value, primitive = GF19.primitiveElement) {
        value = (value % 19 + 19) % 19;
        if (value === 0) throw new Error("GF19.log: Log of zero is undefined.");
        let current = 1;
        for (let i = 0; i < (GF19.FIELD_SIZE -1) ; i++) { // Max exponent is FIELD_SIZE - 2 for log, up to FIELD_SIZE - 1 for full cycle
            if (current === value) return i;
            current = GF19.multiply(current, primitive);
        }
        throw new Error(`GF19.log: Value ${value} not found in powers of primitive ${primitive}.`);
    },
    exp: function(power, primitive = GF19.primitiveElement) {
        // power is modulo FIELD_SIZE - 1 for non-zero results
        return GF19.power(primitive, power % (GF19.FIELD_SIZE - 1));
    }
};
// Appending tests to js/gf19.js

function testGF19() {
    console.log("--- GF(19) Tests ---");
    let pass = true;
    const check = (desc, actual, expected) => {
        const success = JSON.stringify(actual) === JSON.stringify(expected);
        console.log(`GF19 Test: ${desc} - Expected: ${expected}, Got: ${actual}. ${success ? 'PASS' : 'FAIL'}`);
        if (!success) pass = false;
    };
    check("add(10, 15)", GF19.add(10, 15), 6);
    check("subtract(5, 10)", GF19.subtract(5, 10), 14);
    check("multiply(7, 8)", GF19.multiply(7, 8), 18);
    check("inverse(3)", GF19.inverse(3), 13);
    check("divide(18, 3)", GF19.divide(18, 3), 6);
    check("divide(2, 5)", GF19.divide(2, 5), 8); // inv(5)=4, 2*4=8
    check("power(2,3)", GF19.power(2,3), 8);
    check("power(2,18)", GF19.power(2,18), 1); // Fermat's Little Theorem a^(p-1) = 1 mod p
    check("log(8,2)", GF19.log(8,2), 3); // 2^3 = 8
    check("exp(3,2)", GF19.exp(3,2), 8);
    try { GF19.inverse(0); check("inverse(0)", "Error not thrown", "Error thrown");} catch(e) {check("inverse(0)", "Error thrown", "Error thrown");}
    console.log(`GF19 All Tests Result: ${pass ? 'PASS' : 'FAIL'}`);
    return pass;
}
// Appending JSDoc to js/gf19.js

/**
 * @fileoverview Provides arithmetic operations for Galois Field GF(19).
 * All operations are modulo 19.
 * @namespace GF19
 */
// JSDoc for GF19 object methods would typically go above each method.
// For example:
// /** Adds two numbers in GF(19). @param {Number} a Operand a. @param {Number} b Operand b. @returns {Number} (a+b)%19. */
// GF19.add = function(a,b) { ... }
