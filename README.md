# Custom Visual Code - Pure JavaScript Tool

This project implements a custom 2D visual code generation and reading system entirely in client-side JavaScript. It is a JavaScript port of an original Python-based system.

## Features

- **Client-Side Encoding:** Converts input text into a custom visual code image.
  - Supports plain text.
  - Special tags for embedding structured data:
    - `<linear>equation</linear>`: For linear equations (e.g., `2x+3=7`).
    - `<power>equation</power>`: For power equations (e.g., `x^2=9`).
    - `<base64>data</base64>`: For Base64 encoded data.
  - AES Encryption: Encrypt segments of text using:
    - `<encrypt_method>AES</encrypt_method>`
    - `<encrypt_key>your-key</encrypt_key>`
    - `<encrypt>secret message</encrypt>`
- **Client-Side Decoding:** Upload an image of the custom visual code to decode its content.
  - Uses OpenCV.js for image processing (marker detection, perspective correction, symbol identification).
  - Implements Reed-Solomon error correction over GF(19) to handle errors.
- **Interactive Web UI:** `index.html` provides an interface for encoding text, uploading images for decoding, and viewing results.
- **Internal Tests:** Includes a suite of JavaScript unit tests for core components (GF arithmetic, polynomials, Reed-Solomon, data codecs) runnable from the UI.

## How to Use

1.  Clone or download this repository.
2.  Open the `index.html` file in a modern web browser that supports JavaScript and WebAssembly (for OpenCV.js).
3.  **To Encode:**
    - Enter text into the "Text to Encode" area.
    - Click the "Encode to Visual Code" button.
    - The generated visual code will appear on the canvas, and a download link will be provided.
4.  **To Decode:**
    - Click the "Upload Image File" input and select an image of the custom visual code.
    - The image will be previewed.
    - Click the "Decode Visual Code" button.
    - The decoded (and processed) string will appear in the results area.
5.  **To Run Tests:**
    - Click the "Run Internal JS Tests" button.
    - Results will be logged to the browser's developer console, with a summary in the decode results area.

## File Structure

- `index.html`: The main web application page.
- `js/`: Directory containing all JavaScript modules:
  - `gf19.js`: Galois Field GF(19) arithmetic.
  - `polynomial.js`: Polynomial arithmetic over GF(19).
  - `reedsolomon.js`: Reed-Solomon encoder and decoder classes.
  - `custom_codec.js`: Handles data conversion (text <-> bytes <-> symbols), AES, and orchestrates encoding/decoding pipelines.
  - `image_processing.js`: Core image analysis logic using OpenCV.js (marker detection, perspective warp, symbol identification).
  - `ui.js`: User interface event handling and DOM manipulation.
  - `main.js`: Main script for initialization, OpenCV loading checks, and test runner.

## Dependencies (Loaded via CDN)

- **OpenCV.js:** Used for image processing tasks in decoding.
- **Math.js:** Used for client-side evaluation of `<linear>` and `<power>` tags in decoded strings.

## Visual Code Specifics (Brief)

- **Symbols:** Based on combinations of colors (6) and shapes (3: square, circle, triangle).
- **Markers:** Custom 3x3 patterns at three corners of the code for orientation and perspective correction.
- **Data Encoding:** Text is converted to bytes, then to pairs of base-18 symbols. Special tags are mapped to specific byte values.
- **Error Correction:** Reed-Solomon codes over GF(19) are used (N=18, K=14, so up to 2 symbol errors correctable per block).
EOF
