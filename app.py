# app.py

from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import galois
from PIL import Image, ImageDraw
import math
import re
import cv2
import base64
import sympy
from sympy import Eq, solve
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
    # Reed-Solomon codes are defined over a Galois Field (GF).
    # GF(19) means the symbols are integers from 0 to 18.
    GF = galois.GF(19)
    # n: total symbols in a codeword (message + parity).
    # k: message symbols in a codeword.
    RS = galois.ReedSolomon(n, k, field=GF)
    message = GF(message_base18)  # Convert message symbols to GF(19) elements
    codeword = RS.encode(message)
    return codeword

def decode_base18_rs(received_codeword, n, k):
    """
    Decodes a received codeword using Reed-Solomon codes over GF(19).
    Corrects up to (n - k) // 2 symbol errors.
    received_codeword: list of integers in base-18 (0-17).
    Returns: Decoded message as a list of base-18 symbols.
    """
    # GF(19) indicates symbols are 0-18.
    GF = galois.GF(19)
    # n: total symbols in the received codeword.
    # k: expected message symbols in the codeword.
    RS = galois.ReedSolomon(n, k, field=GF)
    received = GF(received_codeword)  # Convert received symbols to GF(19) elements
    try:
        # Attempt to decode the message. This can correct up to (n-k)/2 errors.
        message = RS.decode(received)
        return message.tolist()  # Return as a list of integers
    except galois.ReedSolomonError as e:
        # This error occurs if the codeword has too many errors to be corrected.
        app.logger.error(f"Decoding failed: {e}")
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
    Encodes a string into a list of symbol pairs for image generation.
    The process involves:
    1. Tokenizing the input string based on special tags (e.g., <linear>, <encrypt>).
    2. Converting text segments and special tags into a sequence of bytes.
       - Regular text is UTF-8 encoded.
       - Special tags are mapped to predefined byte values.
       - Content within <encrypt_method> and <encrypt_key> tags is stored directly.
       - Content within <encrypt> tags is encrypted using AES.
    3. Validating that all byte values are within the allowed range (0-323).
    4. Mapping each byte to two base-18 symbols (indices 0-17).
    5. Applying Reed-Solomon forward error correction to the symbol indices.
       - Symbols are padded with '0' to form complete blocks of k symbols.
       - Each block of k symbols is encoded into n symbols.
    6. Converting the encoded symbol indices back to (color, shape) tuples.
    7. Grouping symbols into pairs for image cell representation.
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
                    app.logger.info(f"Encoding <encrypt_method>: {stored_encrypt_method}")
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
                    app.logger.info(f"Encoding <encrypt_key>: {stored_encrypt_key[:2]}...{stored_encrypt_key[-2:]}") # Log partial key for security
                    # No need to encrypt key token
                    key_bytes = key_token.encode('utf-8')
                    byte_list.extend(key_bytes)
                    # Append end marker
                    end_marker_byte = marker_byte_map['</encrypt_key>']
                    byte_list.append(end_marker_byte)
            elif marker == '<encrypt>':
                app.logger.info("Starting <encrypt> block")
                is_encrypting = True
            elif marker == '</encrypt>':
                app.logger.info("Ending </encrypt> block")
                is_encrypting = False

        index += 1

    # Now, process the byte_list
    # Adjust the max byte value check
    for byte in byte_list:
        if not 0 <= byte <= 323:
            raise ValueError("Byte value out of range (0-323)")

    # Map bytes to symbol indices
    # Each byte (0-323) is mapped to two base-18 symbols (0-17).
    symbol_indices = []
    for byte in byte_list:
        symbol1_index = byte // 18
        symbol2_index = byte % 18
        symbol_indices.extend([symbol1_index, symbol2_index])
        # Debug statements
        app.logger.info(f"Encoding byte {byte}: symbol1_index={symbol1_index}, symbol2_index={symbol2_index}")

    # --- Reed-Solomon Encoding ---
    # ECC parameters for GF(19):
    n = 18  # Codeword length (total symbols after encoding: message + parity)
            # For GF(q), n must be <= q-1. Here, 18 <= 19-1.
    k = 14  # Message length (number of data symbols per block)
    # t = (n - k) // 2  # Error correction capability: (18-14)/2 = 2 symbols per block.

    # Pad symbol_indices to be a multiple of k (message length for RS encoding).
    # This ensures that the data can be broken into complete blocks of k symbols.
    # Padding uses '0', which is a valid symbol in GF(19).
    if len(symbol_indices) % k != 0:
        padding_length = k - (len(symbol_indices) % k)
        symbol_indices.extend([0]*padding_length)
        app.logger.info(f"Padded symbol_indices with {padding_length} zeros for RS encoding.")

    # Encode using Reed-Solomon in blocks.
    # Each block of k message symbols is encoded into a codeword of n symbols.
    codeword_indices = []
    for i in range(0, len(symbol_indices), k):
        message_block = symbol_indices[i:i+k] # Get a block of k symbols
        # Encode the block using RS(n,k) over GF(19)
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
    # --- Grid Setup ---
    # Define marker pattern (3x3) and its size.
    marker_pattern = get_marker_pattern()
    marker_size = len(marker_pattern)  # marker_size is 3 for a 3x3 pattern.

    # Flatten the list of symbol pairs into a single list of symbols.
    symbols_flat = [symbol for pair in symbol_pairs for symbol in pair]
    num_data_symbols = len(symbols_flat)

    # Calculate the minimal square grid size required for data symbols.
    data_area_size = math.ceil(math.sqrt(num_data_symbols))

    # Calculate total grid dimensions:
    # Add marker_size for one side (e.g., top markers).
    # Add 1 for spacing between the marker area and the data area.
    # The data_area_size itself.
    # Example: If data_area_size=10, marker_size=3, then grid_width = 10 (data) + 3 (marker) + 1 (space) = 14.
    # This calculation seems to place markers only along two combined edges (e.g. top & left of data)
    # or implies data area is distinct from marker columns/rows.
    # Based on marker placement logic, markers are at 3 corners, so data fills around them.
    # A more accurate way for grid_width/height calculation considering corner markers:
    # The central data area is `data_area_size` x `data_area_size`.
    # Markers are placed at corners. If we assume a simple bounding box approach:
    # grid_width = data_area_size (for data) + marker_size (for one side of markers, e.g. left)
    # Let's re-evaluate based on current marker placement.
    # The current placement puts markers in TL, TR, BL corners.
    # The data area is effectively between these.
    # Grid width needs to accommodate data_area_size symbols plus width of markers on sides.
    # If data_area_size is the side length of the square data region:
    # Width: marker_size (left) + data_area_size + marker_size (right) - this is not how it's done.
    # The current logic: `grid_width = data_area_size + marker_size + 1`
    # This suggests `data_area_size` is the width of the data region,
    # and markers are added along one side (e.g., left) and one row on top, plus a space.
    # Let's trace the placement:
    # Markers are at (0,0), (0, grid_width-marker_size), (grid_height-marker_size, 0).
    # Data is placed by iterating i,j over grid_height, grid_width and skipping marker areas.
    # So, data_area_size is the dimension of the main data block.
    # The total grid size is `data_area_size` plus space for markers around it.
    # A common pattern is: marker_size + data_area_size + marker_size.
    # The current `grid_width = data_area_size + marker_size + 1` implies that
    # one marker dimension and a space are added to the data area size.
    # This seems to be an estimation for a layout where data is mostly contiguous.
    # For a robust layout with corner markers, if `data_area_size` is the dimension of the data region,
    # and markers (size `m`) are at corners, the total grid width might be closer to `m + data_area_size` if data slots into the space
    # defined by markers, or `data_area_size` itself if it's an outer bound.
    # Given the existing code, we assume `data_area_size` refers to the side length of the data symbol region,
    # and `grid_width`/`grid_height` are calculated to fit this data plus the corner markers and some spacing.
    # The `+1` likely ensures a separating row/column between data and markers or edge.

    grid_width = data_area_size + marker_size + 1
    grid_height = data_area_size + marker_size + 1
    app.logger.info(f"Calculated grid dimensions: {grid_width}x{grid_height} for {num_data_symbols} data symbols (data area: {data_area_size}x{data_area_size})")

    # Initialize the grid with a special_symbol (e.g., gray square) for empty cells.
    grid = [[special_symbol for _ in range(grid_width)] for _ in range(grid_height)]

    # --- Place Corner Markers ---
    # Markers are placed at Top-Left, Top-Right, and Bottom-Left corners.
    # Top-left corner marker placement:
    for i in range(marker_size):
        for j in range(marker_size):
            grid[i][j] = marker_pattern[i][j]
    # Top-right corner marker placement:
    for i in range(marker_size):
        for j in range(marker_size):
            grid[i][grid_width - marker_size + j] = marker_pattern[i][j]
    # Bottom-left corner marker placement:
    for i in range(marker_size):
        for j in range(marker_size):
            grid[grid_height - marker_size + i][j] = marker_pattern[i][j]

    # --- Place Data Symbols ---
    # Data symbols are filled into the grid, row by row, skipping areas occupied by markers.
    data_index = 0
    for i in range(grid_height):
        for j in range(grid_width):
            # Check if the current cell (i, j) is part of any marker area.
            is_top_left_marker = (i < marker_size and j < marker_size)
            is_top_right_marker = (i < marker_size and j >= grid_width - marker_size)
            is_bottom_left_marker = (i >= grid_height - marker_size and j < marker_size)

            in_marker_area = is_top_left_marker or is_top_right_marker or is_bottom_left_marker

            if in_marker_area:
                continue  # Skip if it's a marker cell.

            # If not a marker cell, place a data symbol if available.
            if data_index < num_data_symbols:
                grid[i][j] = symbols_flat[data_index]
                data_index += 1
            else:
                # If all data symbols are placed, fill remaining cells with the special_symbol.
                grid[i][j] = special_symbol

    # --- Image Rendering ---
    # Calculate final image dimensions based on grid size, cell size, and margin size.
    # Total width = (number of cells horizontally * cell width) + (number of margins horizontally * margin width)
    img_width = grid_width * CELL_SIZE + (grid_width + 1) * MARGIN_SIZE
    img_height = grid_height * CELL_SIZE + (grid_height + 1) * MARGIN_SIZE

    # Create an RGB image with the specified background color.
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
            app.logger.info(f"Image saved as {output_image}")
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
    app.logger.info(f"CYPHER: {list(ciphertext)}")
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

def process_linear_function(equation_str):
    """
    Solve a linear equation provided as a string.
    """
    x = sympy.symbols('x')
    # Remove spaces
    equation_str = equation_str.replace(' ', '')
    try:
        # Split the equation into left and right parts
        lhs_str, rhs_str = equation_str.split('=')
        lhs = sympy.sympify(lhs_str)
        rhs = sympy.sympify(rhs_str)
        equation = sympy.Eq(lhs, rhs)
        # Solve the equation
        solution = sympy.solve(equation, x)
        return solution
    except Exception as e:
        app.logger.error(f"Error solving linear equation: {e}")
        return []

def process_power_function(content_str):
    """
    Solve a power equation provided as a string.
    """
    x = sympy.symbols('x')
    # Replace '^' with '**'
    content_str = content_str.replace('^', '**')
    # Remove spaces
    content_str = content_str.replace(' ', '')
    try:
        # Split the equation into left and right parts
        lhs_str, rhs_str = content_str.split('=')
        lhs = sympy.sympify(lhs_str)
        rhs = sympy.sympify(rhs_str)
        equation = sympy.Eq(lhs, rhs)
        # Solve the equation
        solution = sympy.solve(equation, x)
        return solution
    except Exception as e:
        app.logger.error(f"Error solving power function: {e}")
        return []

def process_base64_data(content_str):
    """
    Decode base64 data and save it as an image.
    """
    try:
        decoded_data = base64.b64decode(content_str)
        # Save to a file
        with open('decoded_image.png', 'wb') as f:
            f.write(decoded_data)
        app.logger.info("Base64 data decoded and saved as 'decoded_image.png'")
    except Exception as e:
        app.logger.error(f"Error decoding base64 data: {e}")

def identify_symbol(cell_image):
    """
    Identify the color and shape of the symbol in the cell image using OpenCV.
    """

    # Ensure the image is in 'RGB' mode
    cell_image = cell_image.convert('RGB')
    np_image = np.array(cell_image)

    # --- Background Subtraction ---
    # Identify non-background pixels to isolate the symbol.
    # The background color is defined in `color_rgb_map`.
    background_color = np.array(color_rgb_map['background'], dtype=np.uint8)

    # `tolerance`: A pixel channel's value must differ from the background's
    # corresponding channel by more than this tolerance to be considered part of the symbol.
    # This helps ignore minor artifacts or slight color variations in the background.
    tolerance = 60
    # Calculate the absolute difference between the image and the background color.
    # If any color channel (R, G, or B) difference exceeds the tolerance, mark it as non-background.
    diff = np.abs(np_image.astype(int) - background_color.astype(int))
    non_background_pixels = np.any(diff > tolerance, axis=-1).astype(np.uint8) * 255

    # If no non-background pixels are found (e.g., an empty or pure background cell),
    # then no symbol can be identified.
    if not np.any(non_background_pixels):
        return None

    # --- Color Identification ---
    # Get the RGB values of all identified non-background (symbol) pixels.
    colors_in_image = np_image[non_background_pixels == 255]

    # Compute the average color of these symbol pixels. This average RGB value
    # represents the dominant color of the symbol.
    avg_color = np.mean(colors_in_image, axis=0)

    # Find the closest predefined color from `color_rgb_map` using Euclidean distance
    # in the RGB color space. This determines the symbol's color.
    min_distance = float('inf')
    detected_color = None
    for color_name, rgb in color_rgb_map.items():
        if color_name == 'background':
            continue
        distance = np.linalg.norm(avg_color - np.array(rgb))
        if distance < min_distance:
            min_distance = distance
            detected_color = color_name

    if detected_color == 'gray':
        # If the detected color is 'gray', it's treated as a special symbol
        # used for padding or undefined areas, as defined by `special_symbol`.
        return special_symbol

    # --- Shape Identification using OpenCV Contours ---
    # Convert the non-background pixels (symbol) into a binary image (black and white)
    # for contour detection.
    binary_image = non_background_pixels

    # Find contours in the binary image.
    # `cv2.RETR_EXTERNAL`: Retrieves only the extreme outer contours.
    # `cv2.CHAIN_APPROX_SIMPLE`: Compresses horizontal, vertical, and diagonal segments
    #                            and leaves only their end points.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # If no contours are found, the shape cannot be determined.
        return None

    # Assume the largest contour found corresponds to the symbol's shape.
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a simpler polygon.
    # `epsilon`: Maximum distance between the original contour and its approximation.
    #            It's calculated as a percentage (4% in this case) of the contour's arc length.
    #            A smaller epsilon means a closer approximation; a larger epsilon means more simplification.
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # The number of vertices in the approximated polygon helps identify the shape.
    num_vertices = len(approx)

    # Shape identification based on the number of vertices:
    # 3 vertices: Triangle
    # 4 vertices: Square (or rectangle, but symbols are designed as squares)
    # Others:     Circle (approximated by a polygon with more vertices)
    if num_vertices == 3:
        detected_shape = 'triangle'
    elif num_vertices == 4:
        detected_shape = 'square'
    else:
        # Includes shapes that are more complex or round, approximated as circles.
        detected_shape = 'circle'

    # Log the identified symbol's color and shape.
    app.logger.info(f"Identified symbol: Color={detected_color}, Shape={detected_shape}")

    return (detected_color, detected_shape)

def decode_image_pil(image):
    """
    Decodes a PIL Image containing the symbol grid back into the original string.
    The process involves:
    1. Defining grid parameters based on image size, cell size, and margins.
    2. Verifying the presence and correctness of corner markers. If markers are incorrect, decoding fails.
    3. Reading symbols from the grid:
       - Each cell in the image is cropped.
       - `identify_symbol()` is called to determine the (color, shape) of the symbol in the cell.
       - Areas occupied by markers are skipped. Unidentifiable symbols are replaced by `special_symbol`.
    4. Mapping the identified (color, shape) symbols to their corresponding numerical indices (0-18).
    5. Applying Reed-Solomon error correction decoding:
       - Symbol indices are processed in blocks of n symbols.
       - Incomplete blocks are padded with the `special_symbol` index (18).
       - Each block is decoded from n symbols to k symbols. Failed decodes are logged, and the process continues (partial recovery).
    6. Converting the decoded symbol indices back into bytes (each pair of symbols forms one byte).
    7. Reconstructing the original string from the byte list:
       - This involves parsing special marker bytes (e.g., for <linear>, <encrypt>) and handling their content.
       - Encrypted content is decrypted using AES.
       - Content for tags like <linear>, <power>, <base64> is processed by respective helper functions.
    """
    # Define the marker pattern
    marker_pattern = get_marker_pattern()
    marker_size = len(marker_pattern)

    # Define grid dimensions based on image size
    img_width, img_height = image.size
    grid_width = (img_width - MARGIN_SIZE) // (CELL_SIZE + MARGIN_SIZE)
    grid_height = (img_height - MARGIN_SIZE) // (CELL_SIZE + MARGIN_SIZE)

    # Initialize the grid
    grid = [[special_symbol for _ in range(grid_width)] for _ in range(grid_height)]

    # Identify markers and their positions
    marker_positions = [
        (0, 0),  # Top-left
        (0, grid_width - marker_size),  # Top-right
        (grid_height - marker_size, 0),  # Bottom-left
    ]

    # Verify markers
    for (row_offset, col_offset) in marker_positions:
        marker_verified = True
        for i in range(marker_size):
            for j in range(marker_size):
                x = MARGIN_SIZE + (col_offset + j) * (CELL_SIZE + MARGIN_SIZE)
                y = MARGIN_SIZE + (row_offset + i) * (CELL_SIZE + MARGIN_SIZE)
                cell_box = (x, y, x + CELL_SIZE, y + CELL_SIZE)
                cell_image = image.crop(cell_box)
                symbol = identify_symbol(cell_image)
                expected_symbol = marker_pattern[i][j]
                if symbol != expected_symbol:
                    marker_verified = False
                    break
            if not marker_verified:
                app.logger.error(f"Marker verification failed at position ({row_offset}, {col_offset})")
                return None  # Or handle accordingly

    # Read the grid symbols
    symbols_flat = []
    for row in range(grid_height):
        for col in range(grid_width):
            # Skip marker areas
            in_marker_area = (
                (row < marker_size and col < marker_size) or  # Top-left
                (row < marker_size and col >= grid_width - marker_size) or  # Top-right
                (row >= grid_height - marker_size and col < marker_size)  # Bottom-left
            )
            if in_marker_area:
                continue
            x = MARGIN_SIZE + col * (CELL_SIZE + MARGIN_SIZE)
            y = MARGIN_SIZE + row * (CELL_SIZE + MARGIN_SIZE)
            # Crop the cell image
            cell_box = (x, y, x + CELL_SIZE, y + CELL_SIZE)
            cell_image = image.crop(cell_box)
            # Identify the symbol
            symbol = identify_symbol(cell_image)
            if symbol is not None:
                symbols_flat.append(symbol)
            else:
                symbols_flat.append(special_symbol)  # If symbol is None, use special_symbol

    # Map symbols to indices (including special_symbol at index 18)
    symbols_list_extended = symbol_list + [special_symbol]  # Indices 0-18
    symbol_indices = []
    for symbol in symbols_flat:
        try:
            idx = symbols_list_extended.index(symbol)
            symbol_indices.append(idx)
        except ValueError:
            # Symbol not found, skip
            continue

    # --- Reed-Solomon Decoding ---
    # ECC parameters used during encoding (must match):
    n = 18  # Codeword length (total symbols in each block that was encoded)
    k = 14  # Message length (number of data symbols expected after decoding each block)

    # Process symbol_indices (read from image) in blocks of length n for RS decoding.
    decoded_message_symbols = []
    for i in range(0, len(symbol_indices), n):
        codeword_block = symbol_indices[i:i+n] # Get a block of n symbols

        # If the last block is shorter than n, pad it.
        # Padding uses the index of `special_symbol` (18), which is a valid element in GF(19).
        # This is important because RS decoding expects blocks of fixed length n.
        if len(codeword_block) < n:
            padding_needed = n - len(codeword_block)
            codeword_block.extend([symbols_list_extended.index(special_symbol)] * padding_needed)
            app.logger.info(f"Padded last RS codeword block with {padding_needed} special_symbol indices.")

        # Attempt to decode the block using RS(n,k) over GF(19).
        message_block = decode_base18_rs(codeword_block, n, k)
        if message_block is not None:
            # If decoding is successful, add the k message symbols to the list.
            decoded_message_symbols.extend(message_block)
        else:
            # If decoding fails for a block (too many errors), log a warning and continue.
            # This allows for partial data recovery if other blocks are decodable.
            app.logger.warning(f"RS decoding failed for block starting at image symbol index {i}")
            # Note: Depending on requirements, one might choose to halt on first error,
            # or collect information about failed blocks. Here, we attempt to recover as much as possible.
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
            app.logger.warning(f"Invalid byte value {byte_value} from symbols at indices {i}, {i+1}")
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
            app.logger.info(f"Found start marker for <{marker_name}>")
            if marker_name == 'encrypt_method':
                # Collect the method name until end marker
                method_bytes = []
                i += 1
                while i < len(byte_list) and byte_list[i] != marker_byte_map['</encrypt_method>']:
                    method_bytes.append(byte_list[i])
                    i += 1
                stored_encrypt_method = bytes(method_bytes).decode('utf-8', errors='replace').strip()
                app.logger.info(f"Decoded <encrypt_method>: {stored_encrypt_method}")
            elif marker_name == 'encrypt_key':
                # Collect the key until end marker
                key_bytes = []
                i += 1
                while i < len(byte_list) and byte_list[i] != marker_byte_map['</encrypt_key>']:
                    key_bytes.append(byte_list[i])
                    i += 1
                stored_encrypt_key = bytes(key_bytes).decode('utf-8', errors='replace').strip()
                app.logger.info(f"Decoded <encrypt_key>: {stored_encrypt_key[:2]}...{stored_encrypt_key[-2:]}")
            elif marker_name == 'encrypt':
                is_encrypting = True
                content_buffer = []
            else: # 'linear', 'power', 'base64'
                current_marker = marker_name
                content_buffer = []
        elif byte_value in marker_pairs.values():
            # End marker
            end_marker_tag = reverse_marker_byte_map[byte_value]
            app.logger.info(f"Found end marker {end_marker_tag}")
            if is_encrypting and byte_value == marker_byte_map['</encrypt>']:
                if stored_encrypt_method == 'AES' and stored_encrypt_key is not None:
                    try:
                        decrypted_data_bytes = decrypt_data(content_buffer, stored_encrypt_key)
                        output_string += decrypted_data_bytes.decode('utf-8', errors='replace')
                        app.logger.info("Successfully decrypted and decoded AES content.")
                    except (ValueError, UnicodeDecodeError) as e:
                        app.logger.error(f"Decryption/Decode error for encrypted content: {e}")
                        output_string += "[Decryption Error: unable to process content]"
                else:
                    app.logger.error("Encryption method/key not specified for <encrypt> block.")
                    # This case will also be caught by the route's general exception handler if not handled here.
                    # For consistency, appending an error message to output_string might be better than raising an error mid-decode.
                    output_string += "[Encryption Error: method or key not specified]"
                    # raise ValueError("Encryption method or key not specified properly.") # Original behavior
                is_encrypting = False
                content_buffer = []
            elif current_marker is not None and byte_value == marker_pairs.get(marker_byte_map.get(f'<{current_marker}>')):
                app.logger.info(f"Processing content for <{current_marker}>")
                # Process the content buffer
                content_bytes = bytes(content_buffer)
                if current_marker == 'linear':
                    # Process linear function
                    content_str = content_bytes.decode('utf-8', errors='replace')
                    solution = process_linear_function(content_str)
                    output_string += f"Linear Function Solution: {solution}\n"
                elif current_marker == 'power':
                    # Process power function
                    content_str = content_bytes.decode('utf-8', errors='replace')
                    solution = process_power_function(content_str)
                    output_string += f"Power Function Solution: {solution}\n"
                elif current_marker == 'base64':
                    # Process base64 data
                    content_str = content_bytes.decode('utf-8', errors='replace')
                    process_base64_data(content_str) # This logs internally
                    output_string += f"[Base64 data processed]\n"
                # Reset current_marker
                current_marker = None
                content_buffer = []
            # Consider what happens if an end marker is found that doesn't match current_marker (e.g. </linear> inside <power>)
            # Current logic: it would be treated as a normal byte if not the expected end marker. This is acceptable.
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
