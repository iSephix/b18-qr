from flask import Flask, request, jsonify, render_template
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
        print(f"Decoding failed: {e}")
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

def encode_string(input_string, output_image='output.png'):
    """
    Encode a string into a list of symbol pairs and create an image with ECC.
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
        print(f"Encoding byte {byte}: symbol1_index={symbol1_index}, symbol2_index={symbol2_index}")

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
        symbol = symbols_list_extended[idx]
        symbol_list_final.append(symbol)

    # Ensure even number of symbols for pairing
    if len(symbol_list_final) % 2 != 0:
        symbol_list_final.append(special_symbol)

    # Create symbol pairs
    symbol_pairs = []
    for i in range(0, len(symbol_list_final), 2):
        pair = (symbol_list_final[i], symbol_list_final[i+1])
        symbol_pairs.append(pair)

    # Create the image from the symbol pairs
    create_image(symbol_pairs, filename=output_image)
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

def create_image(symbol_pairs, filename='output.png'):
    """
    Create an image from the list of symbol pairs, including markers.
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

    # Save the image
    image.save(filename)
    print(f"Image saved as {filename}")

def encrypt_data(data, key):
    """
    Encrypt data using AES encryption.
    """
    # Ensure key is 16 bytes
    key_bytes = key.encode('utf-8')
    key_padded = pad(key_bytes, 16)[:16]
    cipher = AES.new(key_padded, AES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(data, 16))
    print("CYPHER:    ", ciphertext)
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
        print(f"Error solving linear equation: {e}")
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
        print(f"Error solving power function: {e}")
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
        print("Base64 data decoded and saved as 'decoded_image.png'")
    except Exception as e:
        print(f"Error decoding base64 data: {e}")

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

    # Compute the average color of the symbol
    avg_color = np.mean(colors_in_image, axis=0)

    # Find the closest predefined color using Euclidean distance
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
    print(f"Identified symbol: Color={detected_color}, Shape={detected_shape}")

    return (detected_color, detected_shape)

# Modify the decode_image function to accept PIL Image instead of filename
def decode_image_pil(image):
    """
    Decode a PIL Image back into the original string using ECC.
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
                print(f"Marker verification failed at position ({row_offset}, {col_offset})")
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
            print(f"Decoding failed for block starting at index {i}")
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
            print(f"Invalid byte value {byte_value} from symbols at indices {i}, {i+1}")
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
                    decrypted_data = decrypt_data(bytes(content_buffer), stored_encrypt_key)
                    output_string += decrypted_data.decode('utf-8', errors='replace')
                else:
                    raise ValueError("Encryption method or key not specified properly.")
                is_encrypting = False
                content_buffer = []
            elif current_marker is not None:
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
                    process_base64_data(content_str)
                    output_string += f"[Base64 data processed]\n"
                # Reset current_marker
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/decode', methods=['POST'])
def decode():
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

# Optionally, you can add an encoding endpoint
@app.route('/encode', methods=['POST'])
def encode():
    data = request.form.get('data')
    if not data:
        return jsonify({'error': 'No data provided to encode.'}), 400

    try:
        # Encode the string and create the image in memory
        symbol_pairs = encode_string(data, output_image=None)  # Modify encode_string to handle in-memory images

        # Instead of saving to a file, save to a BytesIO object
        img_io = io.BytesIO()
        create_image(symbol_pairs, filename=img_io)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png', as_attachment=True, attachment_filename='encoded_qr.png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
