# app.py

from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import galois
from PIL import Image, ImageDraw
import math
import re
import cv2
import base64
# import sympy # Sympy removed
# from sympy import Eq, solve # Sympy removed
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import io
import os

app = Flask(__name__)

# ECC Functions
def encode_base18_rs(message_base18, n, k):
    """
    Encodes a message of base-18 symbols using Reed-Solomon codes over GF(19).
    message_base18: list of integers in base-18 (0-17).
    n: codeword length.
    k: message length.
    Returns: Encoded codeword as a GF(19) array.
    """
    GF = galois.GF(19)
    RS = galois.ReedSolomon(n, k, field=GF)
    message = GF(message_base18)
    codeword = RS.encode(message)
    return codeword

def decode_base18_rs(received_codeword, n, k):
    """
    Decodes a received codeword using Reed-Solomon codes over GF(19).
    Corrects up to (n - k) // 2 symbol errors.
    received_codeword: list of integers in base-18 (0-17).
    Returns: Decoded message as a list of base-18 symbols.
    """
    GF = galois.GF(19)
    RS = galois.ReedSolomon(n, k, field=GF)
    received = GF(received_codeword)
    try:
        message = RS.decode(received)
        return message.tolist()
    except galois.ReedSolomonError as e:
        app.logger.error(f"Decoding failed: {e}") # Changed print to app.logger.error
        return None

# Function to generate the symbols list consistently
def generate_symbols():
    colors = ['black', 'white', 'blue', 'green', 'yellow', 'red']
    shapes = ['square', 'circle', 'triangle']
    symbols = [(color, shape) for color in colors for shape in shapes]
    return symbols

# Generate the symbols list
symbol_list = generate_symbols()

# Define the size of each symbol (cell) and margin
CELL_SIZE = 10        # Size of each symbol (cell)
MARGIN_SIZE = 3       # Margin between symbols

# Define the special symbol to fill extra cells
special_symbol = ('gray', 'square')

# Map color names to RGB values
color_rgb_map = {
    'black': (0, 0, 0),
    'white': (255, 0, 255),  # Changed to magenta to distinguish from background
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (255, 255, 0),
    'red': (255, 0, 0),
    'gray': (128, 128, 128),  # Special symbol color
    'background': (255, 255, 255)
}

def encode_string(input_string):
    """
    Encode a string into a list of symbol pairs and return the symbol pairs.
    """
    # Define the marker byte map
    marker_byte_map = {
        '<linear>': 257,
        '</linear>': 258,
        '<power>': 259,
        '</power>': 260,
        '<base64>': 281,
        '</base64>': 282,
        '<encrypt_method>': 285,
        '</encrypt_method>': 286,
        '<encrypt_key>': 287,
        '</encrypt_key>': 288,
        '<encrypt>': 289,
        '</encrypt>': 290,
    }

    # Build a regex pattern to match the markers
    marker_pattern = re.compile('|'.join(re.escape(k) for k in marker_byte_map.keys()))

    # Initialize the byte list
    byte_list = []

    # Split the input string into tokens
    tokens = marker_pattern.split(input_string)
    markers = marker_pattern.findall(input_string)

    # Variables to store encryption method and key
    stored_encrypt_method = None
    stored_encrypt_key = None
    is_encrypting = False

    # Now process the tokens and markers
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if index < len(markers):
            marker = markers[index]
        else:
            marker = None

        if is_encrypting:
            # Encrypt the token
            if stored_encrypt_method == 'AES' and stored_encrypt_key is not None:
                encrypted_data = encrypt_data(token.encode('utf-8'), stored_encrypt_key)
                # Add encrypted data to byte_list
                byte_list.extend(encrypted_data)
            else:
                raise ValueError("Encryption method or key not specified properly.")
        else:
            # Encode the token to bytes and append to byte_list
            byte_array = token.encode('utf-8')
            byte_list.extend(byte_array)

        if marker:
            marker_byte = marker_byte_map[marker]
            byte_list.append(marker_byte)

            if marker == '<encrypt_method>':
                index += 1
                if index < len(tokens):
                    method_token = tokens[index]
                    stored_encrypt_method = method_token.strip()
                    # No need to encrypt method token
                    method_bytes = method_token.encode('utf-8')
                    byte_list.extend(method_bytes)
                    # Append end marker
                    end_marker_byte = marker_byte_map['</encrypt_method>']
                    byte_list.append(end_marker_byte)
            elif marker == '<encrypt_key>':
                index += 1
                if index < len(tokens):
                    key_token = tokens[index]
                    stored_encrypt_key = key_token.strip()
                    # No need to encrypt key token
                    key_bytes = key_token.encode('utf-8')
                    byte_list.extend(key_bytes)
                    # Append end marker
                    end_marker_byte = marker_byte_map['</encrypt_key>']
                    byte_list.append(end_marker_byte)
            elif marker == '<encrypt>':
                is_encrypting = True
            elif marker == '</encrypt>':
                is_encrypting = False

        index += 1

    # Now, process the byte_list
    # Adjust the max byte value check
    for byte in byte_list:
        if not 0 <= byte <= 323:
            raise ValueError("Byte value out of range (0-323)")

    # Map bytes to symbol indices
    symbol_indices = []
    for byte in byte_list:
        symbol1_index = byte // 18
        symbol2_index = byte % 18
        symbol_indices.extend([symbol1_index, symbol2_index])
        # Debug statements
        app.logger.info(f"Encoding byte {byte}: symbol1_index={symbol1_index}, symbol2_index={symbol2_index}") # Changed print to app.logger.info

    # ECC parameters
    n = 18  # Codeword length (must divide q - 1, where q = 19)
    k = 14  # Message length
    t = (n - k) // 2  # Error correction capability

    # Pad symbol_indices to a multiple of k
    if len(symbol_indices) % k != 0:
        padding_length = k - (len(symbol_indices) % k)
        symbol_indices.extend([0]*padding_length)  # Padding with zeros (can be any value 0-17)

    # Encode using ECC
    codeword_indices = []
    for i in range(0, len(symbol_indices), k):
        message_block = symbol_indices[i:i+k]
        codeword_block = encode_base18_rs(message_block, n, k)
        codeword_indices.extend(codeword_block.tolist())

    # Map codeword indices to symbols (including special_symbol for index 18)
    symbols_list_extended = symbol_list + [special_symbol]  # Indices 0-18
    symbol_list_final = []
    for idx in codeword_indices:
        if idx < len(symbols_list_extended):
            symbol = symbols_list_extended[idx]
        else:
            symbol = special_symbol  # Fallback to special_symbol if index is out of range
        symbol_list_final.append(symbol)

    # Ensure even number of symbols for pairing
    if len(symbol_list_final) % 2 != 0:
        symbol_list_final.append(special_symbol)

    # Create symbol pairs
    symbol_pairs = []
    for i in range(0, len(symbol_list_final), 2):
        pair = (symbol_list_final[i], symbol_list_final[i+1])
        symbol_pairs.append(pair)

    return symbol_pairs

def get_marker_pattern():
    """
    Returns a 3x3 grid representing the marker pattern.
    """
    marker_symbol_outer = ('black', 'square')
    marker_symbol_inner = ('white', 'square')
    marker_pattern = [
        [marker_symbol_outer, marker_symbol_outer, marker_symbol_outer],
        [marker_symbol_outer, marker_symbol_inner, marker_symbol_outer],
        [marker_symbol_outer, marker_symbol_outer, marker_symbol_outer]
    ]
    return marker_pattern

def create_image(symbol_pairs, output_image=None):
    """
    Create an image from the list of symbol pairs, including markers.
    If output_image is None, return the image object instead of saving.
    """
    # Define marker pattern
    marker_pattern = get_marker_pattern()
    marker_size = len(marker_pattern)  # Assuming square marker

    # Flatten the list of symbol pairs into a list of symbols
    symbols_flat = [symbol for pair in symbol_pairs for symbol in pair]
    num_data_symbols = len(symbols_flat)

    # Calculate the minimal data area size
    data_area_size = math.ceil(math.sqrt(num_data_symbols))

    # Adjust grid size to include markers
    grid_width = data_area_size + marker_size + 1  # +1 for spacing between markers and data
    grid_height = data_area_size + marker_size + 1

    # Initialize the grid
    grid = [[special_symbol for _ in range(grid_width)] for _ in range(grid_height)]

    # Place markers at the corners
    # Top-left corner
    for i in range(marker_size):
        for j in range(marker_size):
            grid[i][j] = marker_pattern[i][j]
    # Top-right corner
    for i in range(marker_size):
        for j in range(marker_size):
            grid[i][grid_width - marker_size + j] = marker_pattern[i][j]
    # Bottom-left corner
    for i in range(marker_size):
        for j in range(marker_size):
            grid[grid_height - marker_size + i][j] = marker_pattern[i][j]

    # Place data symbols into the grid
    data_index = 0
    for i in range(grid_height):
        for j in range(grid_width):
            # Skip marker areas
            in_marker_area = (
                (i < marker_size and j < marker_size) or  # Top-left
                (i < marker_size and j >= grid_width - marker_size) or  # Top-right
                (i >= grid_height - marker_size and j < marker_size)  # Bottom-left
            )
            if in_marker_area:
                continue
            if data_index < num_data_symbols:
                grid[i][j] = symbols_flat[data_index]
                data_index += 1
            else:
                # Fill remaining with special symbol
                grid[i][j] = special_symbol

    # Now, create the image from the grid
    img_width = grid_width * CELL_SIZE + (grid_width + 1) * MARGIN_SIZE
    img_height = grid_height * CELL_SIZE + (grid_height + 1) * MARGIN_SIZE

    # Create an RGB image with our background color
    image = Image.new('RGB', (img_width, img_height), color=color_rgb_map['background'])
    draw = ImageDraw.Draw(image)

    for row in range(grid_height):
        for col in range(grid_width):
            symbol = grid[row][col]
            color_name, shape = symbol
            color_rgb = color_rgb_map[color_name]

            x = MARGIN_SIZE + col * (CELL_SIZE + MARGIN_SIZE)
            y = MARGIN_SIZE + row * (CELL_SIZE + MARGIN_SIZE)
            top_left = (x, y)
            bottom_right = (x + CELL_SIZE - 1, y + CELL_SIZE - 1)

            if shape == 'square':
                draw.rectangle([top_left, bottom_right], fill=color_rgb)
            elif shape == 'circle':
                draw.ellipse([top_left, bottom_right], fill=color_rgb)
            elif shape == 'triangle':
                points = [
                    (x + CELL_SIZE // 2, y),            # Top center
                    (x, y + CELL_SIZE - 1),             # Bottom left
                    (x + CELL_SIZE - 1, y + CELL_SIZE - 1)  # Bottom right
                ]
                draw.polygon(points, fill=color_rgb)

    if output_image:
        # If output_image is a string, save to file
        if isinstance(output_image, str):
            image.save(output_image)
            app.logger.info(f"Image saved as {output_image}") # Changed print to app.logger.info
        else:
            # Assume output_image is a BytesIO object
            image.save(output_image, format='PNG')
    else:
        return image

def encrypt_data(data, key):
    """
    Encrypt data using AES encryption.
    """
    # Ensure key is 16 bytes
    key_bytes = key.encode('utf-8')
    key_padded = pad(key_bytes, 16)[:16]
    cipher = AES.new(key_padded, AES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(data, 16))
    app.logger.info(f"CYPHER: {list(ciphertext)}") # Changed print to app.logger.info
    return list(ciphertext)

def decrypt_data(ciphertext, key):
    """
    Decrypt data using AES decryption.
    """
    # Ensure key is 16 bytes
    key_bytes = key.encode('utf-8')
    key_padded = pad(key_bytes, 16)[:16]
    cipher = AES.new(key_padded, AES.MODE_ECB)
    decrypted_data = unpad(cipher.decrypt(bytes(ciphertext)), 16)
    return decrypted_data

# def process_linear_function(equation_str): # Removed - Sympy dependency
#     """
#     Solve a linear equation provided as a string.
#     """
#     # x = sympy.symbols('x')
#     # # Remove spaces
#     # equation_str = equation_str.replace(' ', '')
#     # try:
#     #     # Split the equation into left and right parts
#     #     lhs_str, rhs_str = equation_str.split('=')
#     #     lhs = sympy.sympify(lhs_str)
#     #     rhs = sympy.sympify(rhs_str)
#     #     equation = sympy.Eq(lhs, rhs)
#     #     # Solve the equation
#     #     solution = sympy.solve(equation, x)
#     #     return solution
#     # except Exception as e:
#     #     app.logger.error(f"Error solving linear equation: {e}")
#     #     return []
#     pass


# def process_power_function(content_str): # Removed - Sympy dependency
#     """
#     Solve a power equation provided as a string.
#     """
#     # x = sympy.symbols('x')
#     # # Replace '^' with '**'
#     # content_str = content_str.replace('^', '**')
#     # # Remove spaces
#     # content_str = content_str.replace(' ', '')
#     # try:
#     #     # Split the equation into left and right parts
#     #     lhs_str, rhs_str = content_str.split('=')
#     #     lhs = sympy.sympify(lhs_str)
#     #     rhs = sympy.sympify(rhs_str)
#     #     equation = sympy.Eq(lhs, rhs)
#     #     # Solve the equation
#     #     solution = sympy.solve(equation, x)
#     #     return solution
#     # except Exception as e:
#     #     app.logger.error(f"Error solving power function: {e}")
#     #     return []
#     pass

def process_base64_data(content_str):
    """
    Decode base64 data and save it as an image.
    """
    try:
        decoded_data = base64.b64decode(content_str)
        # In a serverless environment like Vercel, writing to the local filesystem is ephemeral
        # and generally not recommended for persistent storage or user-facing file delivery.
        # If the goal was to allow users to download this image, it would typically be
        # returned as a response from a dedicated endpoint, e.g., with a Content-Type of image/png.
        app.logger.info("Base64 data was processed. In a serverless environment, it is not saved to a file.")
    except Exception as e:
        app.logger.error(f"Error decoding base64 data: {e}") # Changed print to app.logger.error

def identify_symbol(cell_image):
    """
    Identify the color and shape of the symbol in the cell image using OpenCV.
    """

    # Ensure the image is in 'RGB' mode
    cell_image = cell_image.convert('RGB')
    np_image = np.array(cell_image)

    # Get the background color
    background_color = np.array(color_rgb_map['background'], dtype=np.uint8)

    # Allow a small tolerance when comparing background color
    tolerance = 60  # Adjust this value as needed
    diff = np.abs(np_image.astype(int) - background_color.astype(int))
    non_background_pixels = np.any(diff > tolerance, axis=-1).astype(np.uint8) * 255

    # If no non-background pixels are found, return None
    if not np.any(non_background_pixels):
        return None

    # Get the RGB values of non-background pixels
    colors_in_image = np_image[non_background_pixels == 255]

    if colors_in_image.shape[0] == 0: # No non-background pixels found
        app.logger.debug("identify_symbol: No non-background pixels found in the cell.")
        return None # Or return special_symbol if that's more appropriate fallback

    # Compute the median color of the symbol for robustness
    # Calculate median for each channel (R, G, B) independently
    median_r = np.median(colors_in_image[:, 0])
    median_g = np.median(colors_in_image[:, 1])
    median_b = np.median(colors_in_image[:, 2])
    median_color = np.array([median_r, median_g, median_b])

    # Find the closest predefined color using Euclidean distance
    min_distance = float('inf')
    detected_color = None
    # app.logger.debug(f"Median color: {median_color}") # For debugging
    for color_name, rgb_tuple in color_rgb_map.items():
        if color_name == 'background':
            continue

        # Ensure rgb_tuple is a numpy array for distance calculation
        rgb_array = np.array(rgb_tuple)
        distance = np.linalg.norm(median_color - rgb_array)
        # app.logger.debug(f"Comparing with {color_name} ({rgb_tuple}), distance: {distance}") # For debugging

        if distance < min_distance:
            min_distance = distance
            detected_color = color_name

    # Add a threshold for minimum distance? If min_distance is still very large,
    # it means no color in the map is a good match.
    # For now, closest match is chosen. Max possible distance is sqrt(3 * 255^2) approx 441.
    # If min_distance > (some_threshold * 441), then maybe it's an unknown color.
    # Example threshold: if distance is greater than, say, 100, it's a poor match.
    # Let's set a loose threshold for now. A tighter one might be needed after testing.
    MAX_ACCEPTABLE_COLOR_DISTANCE = 150 # Heuristic value
    if min_distance > MAX_ACCEPTABLE_COLOR_DISTANCE:
        app.logger.debug(f"identify_symbol: Median color {median_color} was too far (dist: {min_distance}) from any known color. Detected as '{detected_color}'. Overriding to None.")
        # This helps prevent misclassification of truly off-colors.
        # Depending on requirements, could return special_symbol or None.
        # Returning None implies the cell is unidentifiable.
        return None


    if detected_color == 'gray':
        # Special symbol
        return special_symbol

    # Now, determine the shape using OpenCV contours
    # Convert non-background pixels to binary image
    binary_image = non_background_pixels

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    num_vertices = len(approx)

    # Shape identification based on number of vertices
    if num_vertices == 3:
        detected_shape = 'triangle'
    elif num_vertices == 4:
        detected_shape = 'square'
    else:
        detected_shape = 'circle'

    # Debug statement
    app.logger.info(f"Identified symbol: Color={detected_color}, Shape={detected_shape}") # Changed print to app.logger.info

    return (detected_color, detected_shape)

def decode_image_pil(image):
    """
    Decode a PIL Image back into the original string using ECC.
    """
    # Define the marker pattern
    marker_pattern = get_marker_pattern()
    marker_size = len(marker_pattern)

    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(image.convert('RGB'))
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

    # Find markers using computer vision techniques
    raw_found_markers_rects = find_markers_cv(binary_image, get_marker_pattern())
    app.logger.debug(f"Found {len(raw_found_markers_rects) if raw_found_markers_rects else 0} raw markers.")

    if not raw_found_markers_rects or len(raw_found_markers_rects) < 3:
        app.logger.error(f"Not enough raw markers found for perspective correction.")
        return None

    # Select the three corner markers (TL, TR, BL)
    corner_markers = select_corner_markers(raw_found_markers_rects, open_cv_image.shape[:2])

    if not corner_markers:
        app.logger.error("Failed to select three distinct corner markers.")
        return None

    tl_rect, tr_rect, bl_rect = corner_markers['TL'], corner_markers['TR'], corner_markers['BL']
    app.logger.debug(f"Selected corner markers: TL={tl_rect}, TR={tr_rect}, BL={bl_rect}")

    # Define source points for perspective transform (outer corners of markers)
    # TL marker: (x,y)
    # TR marker: (x+w,y)
    # BL marker: (x,y+h)
    src_pt_tl = (tl_rect[0], tl_rect[1])
    src_pt_tr = (tr_rect[0] + tr_rect[2], tr_rect[1])
    src_pt_bl = (bl_rect[0], bl_rect[1] + bl_rect[3])

    # Estimate the 4th source point (BR of the QR code area)
    # P4_src = P_tl + (P_tr - P_tl) + (P_bl - P_tl)
    # This is vector addition on the corner points.
    src_pt_br = (
        src_pt_tl[0] + (src_pt_tr[0] - src_pt_tl[0]) + (src_pt_bl[0] - src_pt_tl[0]),
        src_pt_tl[1] + (src_pt_tr[1] - src_pt_tl[1]) + (src_pt_bl[1] - src_pt_tl[1])
    )
    src_pts = np.float32([src_pt_tl, src_pt_tr, src_pt_bl, src_pt_br])
    app.logger.debug(f"Source points for transform: {src_pts.tolist()}")

    # Define destination points for a canonical square image
    WARPED_SIZE = 300 # Standard size for the warped image
    dst_pts = np.float32([
        [0, 0],                           # TL
        [WARPED_SIZE - 1, 0],             # TR
        [0, WARPED_SIZE - 1],             # BL
        [WARPED_SIZE - 1, WARPED_SIZE - 1]  # BR
    ])
    app.logger.debug(f"Destination points for transform: {dst_pts.tolist()}")

    # Calculate perspective transformation matrix and warp the image
    try:
        perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        app.logger.debug(f"Perspective matrix: {perspective_matrix.tolist() if perspective_matrix is not None else 'None'}")
    except cv2.error as e:
        app.logger.error(f"cv2.getPerspectiveTransform failed: {e}. Source points: {src_pts.tolist()}")
        points_for_check = [src_pt_tl, src_pt_tr, src_pt_bl] # Check TL, TR, BL for collinearity
        # Using a simplified collinearity check based on the determinant/area:
        # (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
        area_check = points_for_check[0][0]*(points_for_check[1][1]-points_for_check[2][1]) + \
                     points_for_check[1][0]*(points_for_check[2][1]-points_for_check[0][1]) + \
                     points_for_check[2][0]*(points_for_check[0][1]-points_for_check[1][1])
        if abs(area_check) < 1.0: # If area is very close to zero
            app.logger.error(f"Source points TL, TR, BL are likely collinear (area_check: {area_check}). Cannot perform perspective transform.")
        return None

    if perspective_matrix is None: # Should be caught by try-except, but as a safeguard
        app.logger.error("Perspective matrix is None after getPerspectiveTransform call.")
        return None

    warped_image_cv = cv2.warpPerspective(open_cv_image, perspective_matrix, (WARPED_SIZE, WARPED_SIZE))
    app.logger.debug(f"Warped image created with size {WARPED_SIZE}x{WARPED_SIZE}.")

    # --- Data Extraction from Warped Image (Trial-and-Error for Grid Dimension) ---

    # This is where it gets tricky. We need to know the total number of cells
    # in the grid (N x N) that the `warped_image_cv` now represents.
    # The `create_image` function uses `grid_width = data_area_size + marker_size + 1`.
    # `data_area_size` depends on the number of symbols.
    # For now, let's make a working assumption or estimate.
    # If we assume standard QR codes, marker_size is 3 cells.
    # A typical small QR might be 21x21 or 25x25 cells.
    # Let's assume ESTIMATED_TOTAL_CELLS_DIM is known or can be estimated.
    # For this custom format, let's assume the markers are 3 cells each.
    # The `marker_size` variable (from get_marker_pattern) is the size in cells.

    # Placeholder: This needs a robust way to determine or estimate.
    # One way: analyze distances between selected corner markers in pixels,
    # divide by average marker width in pixels. This gives marker size in "pixel-blocks".
    # Then, knowing marker is 3 "cells", one "cell" = (marker_width_px / 3).
    # Total image width in cells = WARPED_SIZE / cell_size_px.
    # Example: TL rect (x,y,w,h). w is width of marker in pixels.
    # If marker is 3 cells, then one cell_in_warped_image_px = (w_of_tl_marker_in_warped_image / marker_size_cells)
    # This is complex because w_of_tl_marker_in_warped_image is not simply tl_rect[2] after warping.
    # After warping, the TL marker occupies the top-left of warped_image_cv.
    # Its size would be (marker_size_cells / ESTIMATED_TOTAL_CELLS_DIM) * WARPED_SIZE.

    # Let's assume, for now, we can estimate the total dimension.
    # For example, if the original image had `num_data_symbols`, then:
    # `data_area_size = math.ceil(math.sqrt(num_data_symbols))`
    # `grid_cells_total_dim = data_area_size + marker_pattern_size_cells + 1`
    # This requires `num_data_symbols` which we don't have at decode time until *after* decoding. Chicken and egg.

    # Simplified assumption for now: Try a common size like 21 or 25.
    # Let's try to derive it from the original image creation logic if possible,
    # assuming the density of symbols is somewhat consistent.
    # The number of symbols in `symbols_flat` is before ECC.
    # The number of symbols after ECC is `codeword_indices`.
    # `num_data_symbols = len(symbols_flat)` in `create_image`.
    # This is unknown here.

    # Fallback: If we cannot determine total cells, we cannot reliably extract.
    # For now, let's define a plausible default total grid dimension for testing.
    # A typical QR code version 1 is 21x21 cells.
    # A version 2 is 25x25 cells.
    # The `get_marker_pattern()` returns a 3x3 pattern, so `marker_size_in_cells = 3`.

    # TODO: This ESTIMATED_TOTAL_CELLS_DIM needs to be determined more robustly.
    ESTIMATED_TOTAL_CELLS_DIM = 25 # Example: like a Version 2 QR code (25x25)

    single_cell_size_px = WARPED_SIZE / ESTIMATED_TOTAL_CELLS_DIM
    marker_size_in_cells = len(get_marker_pattern()) # Should be 3

    symbols_flat_from_warped = []
    for r_cell_idx in range(ESTIMATED_TOTAL_CELLS_DIM):
        for c_cell_idx in range(ESTIMATED_TOTAL_CELLS_DIM):
            # Check if current cell is part of a marker
            is_in_tl_marker = (r_cell_idx < marker_size_in_cells and c_cell_idx < marker_size_in_cells)
            is_in_tr_marker = (r_cell_idx < marker_size_in_cells and c_cell_idx >= ESTIMATED_TOTAL_CELLS_DIM - marker_size_in_cells)
            is_in_bl_marker = (r_cell_idx >= ESTIMATED_TOTAL_CELLS_DIM - marker_size_in_cells and c_cell_idx < marker_size_in_cells)

            if is_in_tl_marker or is_in_tr_marker or is_in_bl_marker:
                continue # Skip marker cells

            # Calculate cell bounds in the warped image
            x_start = int(c_cell_idx * single_cell_size_px)
            y_start = int(r_cell_idx * single_cell_size_px)
            x_end = int((c_cell_idx + 1) * single_cell_size_px)
            y_end = int((r_cell_idx + 1) * single_cell_size_px)

            # Add a small margin/inset to avoid cell boundaries, similar to original MARGIN_SIZE logic
            # This might be important for identify_symbol if it's sensitive to edges.
            inset_px = int(single_cell_size_px * 0.1) # 10% inset

            x_start_inset = min(x_start + inset_px, x_end -1) # Ensure x_start_inset < x_end
            y_start_inset = min(y_start + inset_px, y_end -1) # Ensure y_start_inset < y_end
            x_end_inset = max(x_end - inset_px, x_start_inset + 1) # Ensure x_end_inset > x_start_inset
            y_end_inset = max(y_end - inset_px, y_start_inset + 1) # Ensure y_end_inset > y_start_inset

            if x_start_inset >= x_end_inset or y_start_inset >= y_end_inset: # Cell too small after inset
                app.logger.debug(f"Cell [{r_cell_idx},{c_cell_idx}] too small after inset, skipping.")
                cell_symbol = special_symbol # Or handle as error / None
            else:
                cell_cv_sub_image = warped_image_cv[y_start_inset:y_end_inset, x_start_inset:x_end_inset]

                if cell_cv_sub_image.size == 0:
                    app.logger.warning(f"Empty cell image at [{r_cell_idx}][{c_cell_idx}] from warped image. Using special_symbol.")
                    cell_symbol = special_symbol
                else:
                    # Convert OpenCV cell image (BGR) to PIL Image (RGB) for identify_symbol
                    cell_pil_image = Image.fromarray(cv2.cvtColor(cell_cv_sub_image, cv2.COLOR_BGR2RGB))
                    cell_symbol = identify_symbol(cell_pil_image)
                    if cell_symbol is None:
                        # app.logger.debug(f"identify_symbol returned None for cell [{r_cell_idx},{c_cell_idx}]. Using special_symbol.")
                        # For debugging, save this cell:
                        # cell_pil_image.save(f"debug_cell_{r_cell_idx}_{c_cell_idx}.png")
                        cell_symbol = special_symbol

            symbols_flat_from_warped.append(cell_symbol)

    if not symbols_flat_from_warped:
        app.logger.error("No symbols extracted from the warped image.")
        return None

    app.logger.info(f"Extracted {len(symbols_flat_from_warped)} symbols from warped image.")

    # --- Rest of the decoding process (ECC, byte conversion, string reconstruction) ---
    # This part uses `symbols_flat_from_warped`

    # Helper function for trying a specific grid dimension
    def _try_decode_grid(flat_symbols, n_ecc, k_ecc, grid_dim_attempt):
        symbols_list_extended_local = symbol_list + [special_symbol]
        local_symbol_indices = []
        for symbol_count, symbol_val in enumerate(flat_symbols):
            try:
                idx = symbols_list_extended_local.index(symbol_val)
                local_symbol_indices.append(idx)
            except ValueError:
                app.logger.debug(f"Grid {grid_dim_attempt}: Symbol {symbol_val} at flat index {symbol_count} not found. Skipping.")
                continue

        app.logger.debug(f"Grid {grid_dim_attempt}: Converted {len(flat_symbols)} flat symbols to {len(local_symbol_indices)} indices.")
        if not local_symbol_indices:
            app.logger.debug(f"Grid {grid_dim_attempt}: No valid symbol indices for this grid attempt.")
            return None

        decoded_message_symbols = []
        has_ecc_errors = False
        for i in range(0, len(local_symbol_indices), n_ecc):
            codeword_block = local_symbol_indices[i:i+n_ecc]
            if len(codeword_block) < n_ecc:
                codeword_block.extend([symbols_list_extended_local.index(special_symbol)] * (n_ecc - len(codeword_block)))

            # app.logger.debug(f"Grid {grid_dim_attempt}: Decoding block {i//n_ecc} with {len(codeword_block)} indices.")
            message_block = decode_base18_rs(codeword_block, n_ecc, k_ecc) # n_ecc, k_ecc are RS params
            if message_block is not None:
                decoded_message_symbols.extend(message_block)
            else:
                app.logger.debug(f"Grid {grid_dim_attempt}: RS decoding failed for a block (n={n_ecc}, k={k_ecc}).")
                has_ecc_errors = True
                break

        if has_ecc_errors or not decoded_message_symbols:
            app.logger.debug(f"Grid {grid_dim_attempt}: ECC decoding failed for one or more blocks or no symbols after ECC.")
            return None
        app.logger.debug(f"Grid {grid_dim_attempt}: ECC decoding successful, {len(decoded_message_symbols)} message symbols obtained.")

        byte_list = []
        idx_sym = 0
        while idx_sym < len(decoded_message_symbols) - 1:
            symbol1_index = decoded_message_symbols[idx_sym]
            symbol2_index = decoded_message_symbols[idx_sym+1]
            if symbol1_index >= 18 or symbol2_index >= 18:
                idx_sym += 1
                continue
            byte_value = symbol1_index * 18 + symbol2_index
            if 0 <= byte_value <= 323:
                byte_list.append(byte_value)
            else:
                app.logger.debug(f"Grid {grid_dim_attempt}: Invalid byte value {byte_value} from symbols at indices {idx_sym}, {idx_sym+1}")
            idx_sym += 2

        app.logger.debug(f"Grid {grid_dim_attempt}: Converted message symbols to {len(byte_list)} bytes.")
        if not byte_list:
            app.logger.debug(f"Grid {grid_dim_attempt}: No valid bytes extracted from decoded symbols.")
            return None

        # Reconstruct the original string
        # Define reverse marker byte map
        marker_byte_map_local = {
            '<linear>': 257, '</linear>': 258, '<power>': 259, '</power>': 260,
            '<base64>': 281, '</base64>': 282, '<encrypt_method>': 285, '</encrypt_method>': 286,
            '<encrypt_key>': 287, '</encrypt_key>': 288, '<encrypt>': 289, '</encrypt>': 290,
        }
        # marker_names_local = {v: k for k, v in marker_byte_map_local.items()} # This is not what original code did
        # Original code had specific names for start markers
        marker_names_local = {
            257: 'linear', 259: 'power', 281: 'base64', 285: 'encrypt_method', 287: 'encrypt_key', 289: 'encrypt',
        }
        marker_pairs_local = { # Start to End
            257: 258, 259: 260, 281: 282, 285: 286, 287: 288, 289: 290,
        }

        output_string_final = ''
        idx_byte_stream = 0
        current_marker_type_final = None
        content_buffer_final = []
        stored_encrypt_method_final = None
        stored_encrypt_key_final = None
        is_encrypting_final = False

        while idx_byte_stream < len(byte_list):
            byte_val_stream = byte_list[idx_byte_stream]

            if byte_val_stream in marker_names_local:
                if content_buffer_final and not current_marker_type_final and not is_encrypting_final:
                     output_string_final += bytes(content_buffer_final).decode('utf-8', errors='replace')
                     content_buffer_final = []
                current_marker_type_final = marker_names_local[byte_val_stream]
                if current_marker_type_final == 'encrypt_method':
                    idx_byte_stream += 1
                    method_bytes_final = []
                    while idx_byte_stream < len(byte_list) and byte_list[idx_byte_stream] != marker_pairs_local[byte_val_stream]:
                        method_bytes_final.append(byte_list[idx_byte_stream])
                        idx_byte_stream += 1
                    stored_encrypt_method_final = bytes(method_bytes_final).decode('utf-8', errors='replace').strip()
                elif current_marker_type_final == 'encrypt_key':
                    idx_byte_stream += 1
                    key_bytes_final = []
                    while idx_byte_stream < len(byte_list) and byte_list[idx_byte_stream] != marker_pairs_local[byte_val_stream]:
                        key_bytes_final.append(byte_list[idx_byte_stream])
                        idx_byte_stream += 1
                    stored_encrypt_key_final = bytes(key_bytes_final).decode('utf-8', errors='replace').strip()
                elif current_marker_type_final == 'encrypt':
                    is_encrypting_final = True
                    content_buffer_final = []
                else:
                    content_buffer_final = []
            elif byte_val_stream in marker_pairs_local.values():
                expected_start_marker_byte_final = None
                for start_b, end_b in marker_pairs_local.items():
                    if end_b == byte_val_stream: expected_start_marker_byte_final = start_b; break
                expected_marker_type_final = marker_names_local.get(expected_start_marker_byte_final)

                if is_encrypting_final and expected_marker_type_final == 'encrypt':
                    if stored_encrypt_method_final == 'AES' and stored_encrypt_key_final is not None:
                        try:
                            decrypted_data_final = decrypt_data(content_buffer_final, stored_encrypt_key_final)
                            output_string_final += decrypted_data_final.decode('utf-8', errors='replace')
                        except Exception as e:
                            app.logger.error(f"Grid {grid_dim_attempt}: AES decryption failed: {e}")
                            return None
                    else:
                        app.logger.error(f"Grid {grid_dim_attempt}: Encryption method/key missing for decrypt.")
                        return None
                    is_encrypting_final = False; content_buffer_final = []; current_marker_type_final = None
                elif current_marker_type_final and expected_marker_type_final == current_marker_type_final:
                    content_str_final = bytes(content_buffer_final).decode('utf-8', errors='replace')
                    start_marker_tag_final = f"<{current_marker_type_final}>"
                    end_marker_tag_final = f"</{current_marker_type_final}>"
                    if current_marker_type_final in ['linear', 'power']:
                        output_string_final += start_marker_tag_final + content_str_final + end_marker_tag_final
                    elif current_marker_type_final == 'base64':
                        process_base64_data(content_str_final)
                        output_string_final += f"{start_marker_tag_final}{content_str_final}{end_marker_tag_final}[Base64 processed by server]"
                    content_buffer_final = []; current_marker_type_final = None
                else: pass # Mismatched or unexpected end marker
            else:
                if is_encrypting_final or current_marker_type_final:
                    content_buffer_final.append(byte_val_stream)
                else:
                    output_string_final += chr(byte_val_stream)
            idx_byte_stream += 1

        if is_encrypting_final or current_marker_type_final:
            app.logger.warning(f"Grid {grid_dim_attempt}: Decoding finished while in marker '{current_marker_type_final}' or encrypting state.")
            if content_buffer_final and not is_encrypting_final and not current_marker_type_final : # only if it's plain text leftover
                 output_string_final += bytes(content_buffer_final).decode('utf-8', errors='replace')

        if not output_string_final.strip():
            app.logger.debug(f"Grid {grid_dim_attempt}: Resulting output string is empty.")
            return None

        app.logger.info(f"Grid {grid_dim_attempt}: Successfully decoded string: '{output_string_final[:50]}...'")
        return output_string_final

    # --- Main part of decode_image_pil resumes ---
    POSSIBLE_GRID_DIMS = [21, 25, 29, 17, 33]
    n_ecc, k_ecc = 18, 14

    for grid_dim_to_try in POSSIBLE_GRID_DIMS:
        app.logger.info(f"Attempting decode with grid dimension: {grid_dim_to_try}x{grid_dim_to_try}")

        single_cell_px = WARPED_SIZE / grid_dim_to_try
        # marker_size_in_cells is from get_marker_pattern(), typically 3

        current_symbols_flat = []
        for r_idx in range(grid_dim_to_try):
            for c_idx in range(grid_dim_to_try):
                # Skip marker areas
                is_tl_marker_area = (r_idx < marker_size_in_cells and c_idx < marker_size_in_cells)
                is_tr_marker_area = (r_idx < marker_size_in_cells and c_idx >= grid_dim_to_try - marker_size_in_cells)
                is_bl_marker_area = (r_idx >= grid_dim_to_try - marker_size_in_cells and c_idx < marker_size_in_cells)
                if is_tl_marker_area or is_tr_marker_area or is_bl_marker_area:
                    continue

                cell_x_start = int(c_idx * single_cell_px)
                cell_y_start = int(r_idx * single_cell_px)
                cell_x_end = int((c_idx + 1) * single_cell_px)
                cell_y_end = int((r_idx + 1) * single_cell_px)

                inset = int(single_cell_px * 0.15) # Increased inset slightly to 15%

                xs_inset = min(cell_x_start + inset, cell_x_end - 1)
                ys_inset = min(cell_y_start + inset, cell_y_end - 1)
                xe_inset = max(cell_x_end - inset, xs_inset + 1)
                ye_inset = max(cell_y_end - inset, ys_inset + 1)

                symbol_in_cell = special_symbol # Default to special if issues
                if xs_inset < xe_inset and ys_inset < ye_inset:
                    cv_cell_img = warped_image_cv[ys_inset:ye_inset, xs_inset:xe_inset]
                    if cv_cell_img.size > 0:
                        pil_cell_img = Image.fromarray(cv2.cvtColor(cv_cell_img, cv2.COLOR_BGR2RGB))
                        identified = identify_symbol(pil_cell_img)
                        if identified:
                            symbol_in_cell = identified
                        # else: app.logger.debug(f"Grid {grid_dim_to_try}: Cell ({r_idx},{c_idx}) identify_symbol failed.")
                    # else: app.logger.debug(f"Grid {grid_dim_to_try}: Cell ({r_idx},{c_idx}) empty after crop.")
                # else: app.logger.debug(f"Grid {grid_dim_to_try}: Cell ({r_idx},{c_idx}) too small after inset.")
                current_symbols_flat.append(symbol_in_cell)

        app.logger.debug(f"Grid {grid_dim_to_try}: Extracted {len(current_symbols_flat)} flat symbols.")
        if not current_symbols_flat:
            app.logger.debug(f"Grid {grid_dim_to_try}: No symbols extracted.")
            continue # Try next grid dimension

        # Pass grid_dim_to_try for logging purposes within _try_decode_grid
        final_decoded_string = _try_decode_grid(current_symbols_flat, n_ecc, k_ecc, grid_dim_to_try)
        if final_decoded_string is not None:
            app.logger.info(f"Successfully decoded with grid dimension {grid_dim_to_try}x{grid_dim_to_try}.")
            return final_decoded_string

    app.logger.error("Failed to decode image with any of the attempted grid dimensions.")
    return None
    # for symbol_count, symbol in enumerate(symbols_flat_from_warped):
    #     try:
    #         idx = symbols_list_extended.index(symbol)
    #         symbol_indices.append(idx)
    #     except ValueError:
    #         # app.logger.warning(f"Symbol {symbol} at flat index {symbol_count} not found in symbols_list_extended. Skipping.")
    #         continue
    #
    # if not symbol_indices:
    #     app.logger.error("No valid symbol indices obtained from warped image symbols.")
    #     return None

    # ECC parameters
    # n = 18  # Codeword length (must match encoding)
    # k = 14  # Message length
    #
    # # Process symbol_indices in blocks of length n
    # decoded_message_symbols = []
    # for i in range(0, len(symbol_indices), n):
    #     codeword_block = symbol_indices[i:i+n]
    #     if len(codeword_block) < n:
    #         # Incomplete block, pad with special_symbol index
    #         codeword_block.extend([symbols_list_extended.index(special_symbol)] * (n - len(codeword_block)))
    #     # Apply ECC decoding
    #     message_block = decode_base18_rs(codeword_block, n, k)
    #     if message_block is not None:
    #         decoded_message_symbols.extend(message_block)
    #     else:
    #         # Decoding failed, skip this block or handle accordingly
    #         app.logger.warning(f"RS decoding failed for block starting at index {i}")
    #         continue
    #
    # # Convert message symbols back into bytes
    # byte_list = []
    # i = 0
    # while i < len(decoded_message_symbols) - 1:
    #     symbol1_index = decoded_message_symbols[i]
    #     symbol2_index = decoded_message_symbols[i+1]
    #     if symbol1_index >= 18 or symbol2_index >= 18:
    #         # Skip pairs with special symbols
    #         i += 1
    #         continue
    #     byte_value = symbol1_index * 18 + symbol2_index
    #     if 0 < byte_value <= 323:
    #         byte_list.append(byte_value)
    #     else:
    #         # Invalid byte value, skip
    #         app.logger.warning(f"Invalid byte value {byte_value} from symbols at indices {i}, {i+1}")
    #     i += 2
    #
    # # Define reverse marker byte map
    # marker_byte_map = {
    #     '<linear>': 257,
    #     '</linear>': 258,
    #     '<power>': 259,
    #     '</power>': 260,
    #     '<base64>': 281,
    #     '</base64>': 282,
    #     '<encrypt_method>': 285,
    #     '</encrypt_method>': 286,
    #     '<encrypt_key>': 287,
    #     '</encrypt_key>': 288,
    #     '<encrypt>': 289,
    #     '</encrypt>': 290,
    # }
    # reverse_marker_byte_map = {v: k for k, v in marker_byte_map.items()}
    #
    # # Define marker pairs and names
    # marker_pairs = {
    #     257: 258,  # '<linear>' to '</linear>'
    #     259: 260,  # '<power>' to '</power>'
    #     281: 282,  # '<base64>' to '</base64>'
    #     285: 286,  # '<encrypt_method>' to '</encrypt_method>'
    #     287: 288,  # '<encrypt_key>' to '</encrypt_key>'
    #     289: 290,  # '<encrypt>' to '</encrypt>'
    # }
    # marker_names = {
    #     257: 'linear',
    #     259: 'power',
    #     281: 'base64',
    #     285: 'encrypt_method',
    #     287: 'encrypt_key',
    #     289: 'encrypt',
    # }
    #
    # # Reconstruct the original string, processing markers
    # output_string = ''
    # i = 0
    # current_marker = None
    # content_buffer = []
    # stored_encrypt_method = None
    # stored_encrypt_key = None
    # is_encrypting = False
    #
    # while i < len(byte_list):
    #     byte_value = byte_list[i]
    #     if byte_value in marker_names:
    #         # Start marker
    #         marker_name = marker_names[byte_value]
    #         if marker_name == 'encrypt_method':
    #             # Collect the method name until end marker
    #             method_bytes = []
    #             i += 1
    #             while i < len(byte_list) and byte_list[i] != marker_byte_map['</encrypt_method>']:
    #                 method_bytes.append(byte_list[i])
    #                 i += 1
    #             stored_encrypt_method = bytes(method_bytes).decode('utf-8', errors='replace').strip()
    #         elif marker_name == 'encrypt_key':
    #             # Collect the key until end marker
    #             key_bytes = []
    #             i += 1
    #             while i < len(byte_list) and byte_list[i] != marker_byte_map['</encrypt_key>']:
    #                 key_bytes.append(byte_list[i])
    #                 i += 1
    #             stored_encrypt_key = bytes(key_bytes).decode('utf-8', errors='replace').strip()
    #         elif marker_name == 'encrypt':
    #             is_encrypting = True
    #             content_buffer = []
    #         else:
    #             current_marker = marker_name
    #             content_buffer = []
    #     elif byte_value in marker_pairs.values():
    #         # End marker
    #         if is_encrypting and byte_value == marker_byte_map['</encrypt>']:
    #             if stored_encrypt_method == 'AES' and stored_encrypt_key is not None:
    #                 decrypted_data = decrypt_data(content_buffer, stored_encrypt_key)
    #                 output_string += decrypted_data.decode('utf-8', errors='replace')
    #             else:
    #                 raise ValueError("Encryption method or key not specified properly.")
    #             is_encrypting = False
    #             content_buffer = []
    #         elif current_marker is not None:
    #             # Process the content buffer
    #             content_bytes = bytes(content_buffer)
    #             content_str = content_bytes.decode('utf-8', errors='replace')
    #
    #             original_tag_name = current_marker # 'linear', 'power', or 'base64'
    #             start_marker_str = f"<{original_tag_name}>"
    #             end_marker_str = f"</{original_tag_name}>"
    #
    #             if current_marker == 'linear' or current_marker == 'power':
    #                 # Append the original tag and its content, for client-side processing by math.js
    #                 output_string += start_marker_str + content_str + end_marker_str
    #                 app.logger.info(f"Passing <{current_marker}> tag to client: {start_marker_str}{content_str}{end_marker_str}")
    #             elif current_marker == 'base64':
    #                 process_base64_data(content_str) # This logs internally for Vercel
    #                 # Append the original tag and content, plus a server-side processing note
    #                 output_string += f"{start_marker_str}{content_str}{end_marker_str}[Base64 data processed by server and not displayed here]"
    #
    #             current_marker = None
    #             content_buffer = []
    #     elif is_encrypting:
    #         # Collect encrypted content bytes
    #         content_buffer.append(byte_value)
    #     elif current_marker is not None:
    #         # Collect content bytes for markers like <linear>
    #         content_buffer.append(byte_value)
    #     else:
    #         # Normal byte
    #         output_string += chr(byte_value)
    #     i += 1
    #
    # return output_string
    symbols_list_extended = symbol_list + [special_symbol]  # Indices 0-18
    symbol_indices = []
    for symbol in symbols_flat:
        try:
            idx = symbols_list_extended.index(symbol)
            symbol_indices.append(idx)
        except ValueError:
            # Symbol not found, skip
            continue

    # ECC parameters
    n = 18  # Codeword length (must match encoding)
    k = 14  # Message length

    # Process symbol_indices in blocks of length n
    decoded_message_symbols = []
    for i in range(0, len(symbol_indices), n):
        codeword_block = symbol_indices[i:i+n]
        if len(codeword_block) < n:
            # Incomplete block, pad with special_symbol index
            codeword_block.extend([symbols_list_extended.index(special_symbol)] * (n - len(codeword_block)))
        # Apply ECC decoding
        message_block = decode_base18_rs(codeword_block, n, k)
        if message_block is not None:
            decoded_message_symbols.extend(message_block)
        else:
            # Decoding failed, skip this block or handle accordingly
            app.logger.warning(f"RS decoding failed for block starting at index {i}") # Changed print to app.logger.warning
            continue

    # Convert message symbols back into bytes
    byte_list = []
    i = 0
    while i < len(decoded_message_symbols) - 1:
        symbol1_index = decoded_message_symbols[i]
        symbol2_index = decoded_message_symbols[i+1]
        if symbol1_index >= 18 or symbol2_index >= 18:
            # Skip pairs with special symbols
            i += 1
            continue
        byte_value = symbol1_index * 18 + symbol2_index
        if 0 < byte_value <= 323:
            byte_list.append(byte_value)
        else:
            # Invalid byte value, skip
            app.logger.warning(f"Invalid byte value {byte_value} from symbols at indices {i}, {i+1}") # Changed print to app.logger.warning
        i += 2

    # Define reverse marker byte map
    marker_byte_map = {
        '<linear>': 257,
        '</linear>': 258,
        '<power>': 259,
        '</power>': 260,
        '<base64>': 281,
        '</base64>': 282,
        '<encrypt_method>': 285,
        '</encrypt_method>': 286,
        '<encrypt_key>': 287,
        '</encrypt_key>': 288,
        '<encrypt>': 289,
        '</encrypt>': 290,
    }
    reverse_marker_byte_map = {v: k for k, v in marker_byte_map.items()}

    # Define marker pairs and names
    marker_pairs = {
        257: 258,  # '<linear>' to '</linear>'
        259: 260,  # '<power>' to '</power>'
        281: 282,  # '<base64>' to '</base64>'
        285: 286,  # '<encrypt_method>' to '</encrypt_method>'
        287: 288,  # '<encrypt_key>' to '</encrypt_key>'
        289: 290,  # '<encrypt>' to '</encrypt>'
    }
    marker_names = {
        257: 'linear',
        259: 'power',
        281: 'base64',
        285: 'encrypt_method',
        287: 'encrypt_key',
        289: 'encrypt',
    }

    # Reconstruct the original string, processing markers
    output_string = ''
    i = 0
    current_marker = None
    content_buffer = []
    stored_encrypt_method = None
    stored_encrypt_key = None
    is_encrypting = False

    while i < len(byte_list):
        byte_value = byte_list[i]
        if byte_value in marker_names:
            # Start marker
            marker_name = marker_names[byte_value]
            if marker_name == 'encrypt_method':
                # Collect the method name until end marker
                method_bytes = []
                i += 1
                while i < len(byte_list) and byte_list[i] != marker_byte_map['</encrypt_method>']:
                    method_bytes.append(byte_list[i])
                    i += 1
                stored_encrypt_method = bytes(method_bytes).decode('utf-8', errors='replace').strip()
            elif marker_name == 'encrypt_key':
                # Collect the key until end marker
                key_bytes = []
                i += 1
                while i < len(byte_list) and byte_list[i] != marker_byte_map['</encrypt_key>']:
                    key_bytes.append(byte_list[i])
                    i += 1
                stored_encrypt_key = bytes(key_bytes).decode('utf-8', errors='replace').strip()
            elif marker_name == 'encrypt':
                is_encrypting = True
                content_buffer = []
            else:
                current_marker = marker_name
                content_buffer = []
        elif byte_value in marker_pairs.values():
            # End marker
            if is_encrypting and byte_value == marker_byte_map['</encrypt>']:
                if stored_encrypt_method == 'AES' and stored_encrypt_key is not None:
                    decrypted_data = decrypt_data(content_buffer, stored_encrypt_key)
                    output_string += decrypted_data.decode('utf-8', errors='replace')
                else:
                    raise ValueError("Encryption method or key not specified properly.")
                is_encrypting = False
                content_buffer = []
            elif current_marker is not None:
                # Process the content buffer
                content_bytes = bytes(content_buffer)
                content_str = content_bytes.decode('utf-8', errors='replace')

                original_tag_name = current_marker # 'linear', 'power', or 'base64'
                start_marker_str = f"<{original_tag_name}>"
                end_marker_str = f"</{original_tag_name}>"

                if current_marker == 'linear' or current_marker == 'power':
                    # Append the original tag and its content, for client-side processing by math.js
                    output_string += start_marker_str + content_str + end_marker_str
                    app.logger.info(f"Passing <{current_marker}> tag to client: {start_marker_str}{content_str}{end_marker_str}")
                elif current_marker == 'base64':
                    process_base64_data(content_str) # This logs internally for Vercel
                    # Append the original tag and content, plus a server-side processing note
                    output_string += f"{start_marker_str}{content_str}{end_marker_str}[Base64 data processed by server and not displayed here]"

                current_marker = None
                content_buffer = []
        elif is_encrypting:
            # Collect encrypted content bytes
            content_buffer.append(byte_value)
        elif current_marker is not None:
            # Collect content bytes for markers like <linear>
            content_buffer.append(byte_value)
        else:
            # Normal byte
            output_string += chr(byte_value)
        i += 1

    return output_string

def verify_marker_pattern_cv(marker_candidate_image, expected_pattern_symbols):
    """
    Verifies if a candidate image region matches the 3x3 marker pattern.
    marker_candidate_image: A binary OpenCV image of the candidate region.
    expected_pattern_symbols: The 3x3 marker pattern from get_marker_pattern().
    Returns: True if the pattern matches, False otherwise.
    """
    height, width = marker_candidate_image.shape
    if height == 0 or width == 0:
        app.logger.debug("verify_marker_pattern_cv: Candidate image is empty.")
        return False

    cell_h, cell_w = height // 3, width // 3
    if cell_h == 0 or cell_w == 0:
        app.logger.debug(f"verify_marker_pattern_cv: Candidate image too small (h={height}, w={width}) to form 3x3 grid.")
        return False

    expected_binary_pattern = []
    for row_idx in range(3):
        current_row = []
        for col_idx in range(3):
            color_name, _ = expected_pattern_symbols[row_idx][col_idx]
            if color_name == 'black':
                current_row.append(255) # Expected to be white (object) in adaptive binary image
            elif color_name == 'white':
                current_row.append(0)   # Expected to be black (background) in adaptive binary image
            else:
                app.logger.warning(f"Unexpected color in marker pattern: {color_name}")
                return False
        expected_binary_pattern.append(current_row)

    observed_pattern = []
    for r_cell in range(3):
        observed_row = []
        for c_cell in range(3):
            cell = marker_candidate_image[r_cell*cell_h:(r_cell+1)*cell_h, c_cell*cell_w:(c_cell+1)*cell_w]
            if cell.size == 0:
                app.logger.debug(f"verify_marker_pattern_cv: Cell at [{r_cell}][{c_cell}] is empty.")
                return False

            mean_intensity = np.mean(cell)
            observed_value = 255 if mean_intensity > 127 else 0 # Threshold mean to binary
            observed_row.append(observed_value)
        observed_pattern.append(observed_row)

    if not np.array_equal(observed_pattern, expected_binary_pattern):
        # app.logger.debug(f"Pattern mismatch: Expected {expected_binary_pattern}, Observed {observed_pattern}")
        # To save images for debugging (ensure cv2 is available and path is writable if uncommented):
        # cv2.imwrite(f"debug_marker_candidate_observed.png", marker_candidate_image)
        # temp_expected_img = np.zeros((height, width), dtype=np.uint8)
        # for r_idx in range(3):
        #     for c_idx in range(3):
        #         val = expected_binary_pattern[r_idx][c_idx]
        #         temp_expected_img[r_idx*cell_h:(r_idx+1)*cell_h, c_idx*cell_w:(c_idx+1)*cell_w] = val
        # cv2.imwrite(f"debug_marker_candidate_expected.png", temp_expected_img)
        return False

    return True


def find_markers_cv(binary_image, expected_pattern):
    """
    Finds all 3x3 markers in a binary image.
    binary_image: An OpenCV binary image (result of thresholding).
    expected_pattern: The 3x3 marker pattern from get_marker_pattern().
    Returns: A list of merged marker bounding boxes [(x, y, w, h), ...].
    """
    found_markers_coords = []

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None or len(hierarchy) == 0:
        app.logger.info("No contours or hierarchy found.")
        return []

    # Iterate through contours to find marker candidates (outermost squares)
    for i, contour in enumerate(contours):
        # Approximate contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) == 4: # Is it a quadrilateral?
            x, y, w, h = cv2.boundingRect(approx)

            aspect_ratio = float(w) / h
            if not (0.75 < aspect_ratio < 1.25): # More tolerant aspect ratio
                continue

            # Area filter: adaptive to image size, min_area in pixels
            min_marker_abs_size = 20 # e.g., marker should be at least 20x20 pixels
            # Marker should not be excessively large (e.g., > 50% of image dimension)
            if not (min_marker_abs_size < w < binary_image.shape[1] * 0.5 and \
                    min_marker_abs_size < h < binary_image.shape[0] * 0.5):
                continue

            if not cv2.isContourConvex(approx):
                continue

            # Check for the 3-level nested structure typical of QR finder patterns
            # hierarchy structure: [Next, Previous, First_Child, Parent]
            # Current contour `i` is a candidate for the outermost square (c0)

            # c0 must have a child (c1 - the white ring)
            child_idx = hierarchy[0][i][2]
            if child_idx == -1: # No child
                continue

            # c1 must not have a next or prev sibling within c0 for typical finder patterns
            # This means c1 is the *only* contour directly inside c0
            if hierarchy[0][child_idx][0] != -1 or hierarchy[0][child_idx][1] != -1:
                 # If c1 has siblings, it might be a more complex structure, not a simple finder pattern.
                 # This check can be strict. For now, let's allow it and rely on pattern verification.
                 pass


            # c1 must have a child (c2 - the inner black square)
            grandchild_idx = hierarchy[0][child_idx][2]
            if grandchild_idx == -1: # No grandchild
                continue

            # c2 (grandchild) should ideally not have further children for a standard finder pattern.
            if hierarchy[0][grandchild_idx][2] != -1: # Grandchild has a child
                # This could be noise or a non-standard marker.
                # For robustness, we might still consider it if the pattern matches.
                pass

            # Now, verify the 3x3 pattern using the ROI of the outermost contour `i`
            marker_roi = binary_image[y:y+h, x:x+w]
            if marker_roi.size == 0:
                continue

            # Resize to a standard size for pattern verification to handle small variations
            # Ensure size is a multiple of 3 for grid division.
            std_size = 3 * 15 # e.g., 45x45. Increased from 3*7.
            resized_roi = cv2.resize(marker_roi, (std_size, std_size), interpolation=cv2.INTER_NEAREST)

            if verify_marker_pattern_cv(resized_roi, expected_pattern):
                found_markers_coords.append({'rect': (x, y, w, h), 'contour': approx})
                # cv2.drawContours(binary_image, [approx], -1, (127), 2) # For debugging
            # else:
                # app.logger.debug(f"Nested contour at ({x},{y}) failed pattern verification.")
                # cv2.imwrite(f"debug_failed_verify_{x}_{y}.png", resized_roi)


    if not found_markers_coords:
        app.logger.info("No markers passed all filters.")
        return []

    app.logger.info(f"Found {len(found_markers_coords)} raw marker candidates after hierarchy and pattern check.")

    # Non-Maximum Suppression (NMS) to merge/filter overlapping boxes
    # Sort by area (descending) or confidence (if available)
    # For now, a simpler overlap-based merging.
    # We use the bounding boxes for NMS.

    if not found_markers_coords:
        return []

    # Extracting rects for NMS
    rects = [m['rect'] for m in found_markers_coords]

    # Simple NMS: if a box is heavily overlapped by a larger box, suppress it.
    # A better NMS would use confidence scores, but we don't have that directly.
    # We can use area as a proxy for "strength".
    # Sort by area descending, so larger boxes are processed first.
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)

    merged_boxes = []
    iou_threshold = 0.3 # Intersection over Union threshold

    while len(rects) > 0:
        current_rect = rects.pop(0) # Get the largest remaining rect
        merged_boxes.append(current_rect)

        # Store rects to keep
        remaining_rects = []
        for r_idx in range(len(rects)):
            other_rect = rects[r_idx]

            # Calculate Intersection over Union (IoU)
            x1_inter = max(current_rect[0], other_rect[0])
            y1_inter = max(current_rect[1], other_rect[1])
            x2_inter = min(current_rect[0] + current_rect[2], other_rect[0] + other_rect[2])
            y2_inter = min(current_rect[1] + current_rect[3], other_rect[1] + other_rect[3])

            inter_width = max(0, x2_inter - x1_inter)
            inter_height = max(0, y2_inter - y1_inter)
            inter_area = inter_width * inter_height

            current_rect_area = current_rect[2] * current_rect[3]
            other_rect_area = other_rect[2] * other_rect[3]

            union_area = current_rect_area + other_rect_area - inter_area

            if union_area == 0: # Avoid division by zero
                iou = 0
            else:
                iou = inter_area / union_area

            if iou < iou_threshold:
                remaining_rects.append(other_rect)

        rects = remaining_rects

    app.logger.info(f"find_markers_cv returning {len(merged_boxes)} merged markers: {merged_boxes}")
    return merged_boxes


def get_marker_center(marker_rect):
    x, y, w, h = marker_rect
    return (x + w // 2, y + h // 2)

def select_corner_markers(marker_rects, image_shape_for_sorting_heuristic):
    """
    Selects the top-left (TL), top-right (TR), and bottom-left (BL) markers
    from a list of detected marker bounding boxes.
    marker_rects: List of (x, y, w, h) for detected markers.
    image_shape_for_sorting_heuristic: tuple (height, width) of the image, used for heuristics.
    Returns: A dictionary {'TL': rect_tl, 'TR': rect_tr, 'BL': rect_bl} or None.
    """
    num_markers = len(marker_rects)
    if num_markers < 3:
        app.logger.warning(f"Not enough markers detected ({num_markers}) to select corners.")
        return None

    # Calculate centers for all markers
    centers = [get_marker_center(rect) for rect in marker_rects]

    # Pair rects with their centers for easier manipulation
    markers_with_centers = list(zip(marker_rects, centers))

    # Heuristic to find Top-Left (TL):
    # Sort by distance from origin (0,0) or sum of coordinates.
    # Smallest sum of (center_x + center_y) is likely TL.
    markers_with_centers.sort(key=lambda mc: mc[1][0] + mc[1][1])
    tl_marker_rect, tl_center = markers_with_centers[0]

    remaining_markers = []
    for r, c in markers_with_centers[1:]:
        if r != tl_marker_rect: # Ensure not to include TL again if multiple markers are at same coord sum
             remaining_markers.append(({'rect': r, 'center': c}))

    if len(remaining_markers) < 2:
        app.logger.warning(f"Could not find enough remaining markers ({len(remaining_markers)}) after selecting TL.")
        return None

    best_tr = None
    best_bl = None

    # Iterate through all unique pairs of remaining markers to find TR and BL
    # This is more robust than sequential greedy selection.
    min_combined_metric = float('inf') # Lower is better

    for i in range(len(remaining_markers)):
        for j in range(len(remaining_markers)):
            if i == j:
                continue

            p1_data = remaining_markers[i] # Candidate for TR or BL
            p2_data = remaining_markers[j] # Candidate for the other role

            v_tl_p1 = (p1_data['center'][0] - tl_center[0], p1_data['center'][1] - tl_center[1])
            v_tl_p2 = (p2_data['center'][0] - tl_center[0], p2_data['center'][1] - tl_center[1])

            # Check basic geometric properties
            # P1 should be to the right of TL, P2 should be below TL (or vice-versa)
            # Vector TL->P1 should be somewhat horizontal, TL->P2 somewhat vertical.
            # Or use cross product to determine orientation (TL, P1, P2) - should form a right-hand turn.
            # z_cross = v_tl_p1[0] * v_tl_p2[1] - v_tl_p1[1] * v_tl_p2[0]

            # We expect P1 to be TR and P2 to be BL, or P1 to be BL and P2 to be TR.
            # Case 1: P1 is TR, P2 is BL
            # v_tl_p1_x > 0, v_tl_p1_y is smallish
            # v_tl_p2_y > 0, v_tl_p2_x is smallish

            # Calculate dot product of (TL->P1) and (TL->P2) to check for perpendicularity
            len_v_tl_p1 = math.sqrt(v_tl_p1[0]**2 + v_tl_p1[1]**2)
            len_v_tl_p2 = math.sqrt(v_tl_p2[0]**2 + v_tl_p2[1]**2)

            if len_v_tl_p1 == 0 or len_v_tl_p2 == 0: continue

            cos_angle_between_p1_p2 = (v_tl_p1[0]*v_tl_p2[0] + v_tl_p1[1]*v_tl_p2[1]) / (len_v_tl_p1 * len_v_tl_p2)
            angle_diff_from_90 = abs(abs(math.degrees(math.acos(cos_angle_between_p1_p2))) - 90)

            # Metric: how close is the angle to 90 degrees? Lower is better.
            # Add distances: prefer markers that form a larger area (further apart)
            # This combined metric can be tuned.
            # Penalize if P1 is not clearly right-ish or P2 is not clearly down-ish (or vice versa)

            # Let's try to assign roles: P_tr_cand, P_bl_cand
            # If v_tl_p1 is more horizontal and v_tl_p2 is more vertical:
            if abs(v_tl_p1[0]) > abs(v_tl_p1[1]) and abs(v_tl_p2[1]) > abs(v_tl_p2[0]):
                # P1 is potential TR, P2 is potential BL
                # Ensure P1 is to the right of TL, P2 is below TL
                if v_tl_p1[0] > 0 and v_tl_p2[1] > 0:
                    P_tr_cand, P_bl_cand = p1_data, p2_data
                else: continue # Wrong orientation
            # If v_tl_p1 is more vertical and v_tl_p2 is more horizontal:
            elif abs(v_tl_p1[1]) > abs(v_tl_p1[0]) and abs(v_tl_p2[0]) > abs(v_tl_p2[1]):
                # P1 is potential BL, P2 is potential TR
                # Ensure P1 is below TL, P2 is to the right of TL
                if v_tl_p1[1] > 0 and v_tl_p2[0] > 0:
                    P_tr_cand, P_bl_cand = p2_data, p1_data
                else: continue # Wrong orientation
            else:
                continue # Not clearly TR/BL candidates in terms of axis alignment

            # Check if already found a better pair or if this pair is valid
            # metric = angle_diff_from_90 - (len_v_tl_p1 + len_v_tl_p2) * 0.01 # Prioritize angle, then distance
            metric = angle_diff_from_90 # Simpler metric for now

            if metric < min_combined_metric:
                 # Verify cross product sign for orientation TL -> P_tr_cand -> P_bl_cand
                 # If P_tr_cand is (x1,y1) and P_bl_cand is (x2,y2) relative to TL
                vec_tr = (P_tr_cand['center'][0] - tl_center[0], P_tr_cand['center'][1] - tl_center[1])
                vec_bl = (P_bl_cand['center'][0] - tl_center[0], P_bl_cand['center'][1] - tl_center[1])

                # Z component of cross product: vec_tr_x * vec_bl_y - vec_tr_y * vec_bl_x
                # For image coords (y down), if TL->TR is along positive X and TL->BL is along positive Y,
                # cross product should be positive.
                z_cross = vec_tr[0] * vec_bl[1] - vec_tr[1] * vec_bl[0]
                if z_cross > 0: # Correct "L" shape orientation (TL-TR then TL-BL is counter-clockwise turn if Y is up, or clockwise if Y is down)
                                # With Y pointing down, TL->TR (positive X like) and TL->BL (positive Y like)
                                # means (x, small_y) and (small_x, y).
                                # (x*y) - (small_y*small_x) should be positive if x,y are large positive.
                    min_combined_metric = metric
                    best_tr = P_tr_cand
                    best_bl = P_bl_cand
                # else:
                    # app.logger.debug(f"Pair ({P_tr_cand['rect']}, {P_bl_cand['rect']}) rejected due to z_cross {z_cross}")


    if not best_tr or not best_bl:
        app.logger.warning(f"Could not identify distinct TR and BL markers using pair iteration. TR: {best_tr is not None}, BL: {best_bl is not None}")
        return None

    if best_tr['rect'] == best_bl['rect']: # Should not happen if i != j
        app.logger.error("Critical error: TR and BL markers are identical in pair iteration.") # Should be caught by i!=j
        return None

    app.logger.info(f"Selected TL: {tl_marker_rect}, TR: {best_tr['rect']}, BL: {best_bl['rect']}")
    return {'TL': tl_marker_rect, 'TR': best_tr['rect'], 'BL': best_bl['rect']}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/decode', methods=['POST'])
def decode_route():
    if 'qr_image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['qr_image']

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        # Open the image using PIL
        image = Image.open(file.stream)

        # Decode the image
        decoded_str = decode_image_pil(image)

        if decoded_str is None:
            return jsonify({'error': 'Failed to decode the image.'}), 400

        return jsonify({'decoded_string': decoded_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/encode', methods=['POST'])
def encode_route():
    data = request.form.get('data')
    if not data:
        return jsonify({'error': 'No data provided to encode.'}), 400

    try:
        # Encode the string and create the image in memory
        symbol_pairs = encode_string(data)

        # Create image in memory using BytesIO
        img_io = io.BytesIO()
        create_image(symbol_pairs, output_image=img_io)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png', as_attachment=True, attachment_filename='encoded_qr.png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
