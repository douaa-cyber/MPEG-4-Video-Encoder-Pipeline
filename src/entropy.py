import numpy as np
import cv2 as cv
import heapq
import pickle
import struct

# ======================
# ZIGZAG
# ======================
def zigzag_scan(block):
    h, w = block.shape
    result = []

    for s in range(h + w - 1):
        if s % 2 == 0:
            for i in range(s + 1):
                j = s - i
                if i < h and j < w:
                    result.append(int(block[i][j]))
        else:
            for i in range(s + 1):
                j = s - i
                if j < h and i < w:
                    result.append(int(block[j][i]))
    return result


def inverse_zigzag(arr, size=8):
    block = np.zeros((size, size))
    idx = 0

    for s in range(2 * size - 1):
        if s % 2 == 0:
            for i in range(s + 1):
                j = s - i
                if i < size and j < size:
                    block[i][j] = arr[idx]
                    idx += 1
        else:
            for i in range(s + 1):
                j = s - i
                if j < size and i < size:
                    block[j][i] = arr[idx]
                    idx += 1

    return block


# ======================
# RLE
# ======================
def rle_encode(arr):
    encoded = []
    count = 0

    for val in arr:
        if val == 0:
            count += 1
        else:
            encoded.append((count, val))
            count = 0

    encoded.append((0, 0))  # End Of Block
    return encoded


def rle_decode(encoded):
    arr = []

    for zeros, val in encoded:
        if (zeros, val) == (0, 0):
            break
        arr.extend([0] * zeros)
        arr.append(val)

    while len(arr) < 64:
        arr.append(0)

    return arr


# ======================
# HUFFMAN
# ======================
class Node:
    def __init__(self, value=None, freq=0):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data):
    freq = {}
    for item in data:
        freq[item] = freq.get(item, 0) + 1

    heap = [Node(value=k, freq=v) for k, v in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        root = Node()
        root.left = heap[0]
        return root

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)

        merged = Node(freq=n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heapq.heappush(heap, merged)

    return heap[0]


def build_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}

    if node is None:
        return codebook

    if node.value is not None:
        codebook[node.value] = prefix if prefix != "" else "0"

    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)

    return codebook


def huffman_encode(data, codebook):
    return "".join(codebook[item] for item in data)


def huffman_decode(bitstring, root):
    decoded = []
    node = root

    for bit in bitstring:
        node = node.left if bit == "0" else node.right

        if node.value is not None:
            decoded.append(node.value)
            node = root

    return decoded


# ======================
# BITSTREAM UTILS
# ======================
def bits_to_bytes(bitstring):
    b = bytearray()
    for i in range(0, len(bitstring), 8):
        byte = bitstring[i:i+8]
        b.append(int(byte.ljust(8, '0'), 2))
    return b


def bytes_to_bits(byte_data):
    return ''.join(f'{byte:08b}' for byte in byte_data)


# ======================
# SINGLE FILE SAVE/LOAD
# Format: [4 bytes tree size][tree bytes][bitstream bytes]
# ======================
def save_to_bin(bitstring, tree, filename="video.bin"):
    tree_bytes = pickle.dumps(tree)
    tree_size = len(tree_bytes)
    bit_bytes = bits_to_bytes(bitstring)

    with open(filename, "wb") as f:
        f.write(struct.pack(">I", tree_size))  # 4-byte header = tree size
        f.write(tree_bytes)
        f.write(bit_bytes)

    total = 4 + tree_size + len(bit_bytes)
    print(f"Saved to {filename} ({total} bytes total)")


def load_from_bin(filename="video.bin"):
    with open(filename, "rb") as f:
        tree_size = struct.unpack(">I", f.read(4))[0]
        tree = pickle.loads(f.read(tree_size))
        bit_bytes = f.read()

    bitstring = bytes_to_bits(bit_bytes)
    return bitstring, tree


# ======================
# FLATTEN STREAM
# ======================
def flatten_stream(encoded_stream):
    flat = []

    for frame_type, content in encoded_stream:

        flat.append(frame_type)

        if frame_type == "I":
            # content = (rle_blocks, original_shape)
            rle_blocks, original_shape = content
            for block in rle_blocks:
                for pair in block:
                    flat.append(pair)

        elif frame_type == "P":
            mv, residuals = content

            # Flatten motion vectors
            for pos, v in mv:
                flat.append(tuple(v))  # (dy, dx)

            # Flatten residual RLE sub-blocks
            for pos, rle_sub_blocks in residuals:
                for block in rle_sub_blocks:
                    for pair in block:
                        flat.append(pair)

    return flat

# À METTRE À LA FIN DE src/entropy.py

def unflatten_stream(flat_data, width, height):
    """
    Prend la liste plate décodée par Huffman et reconstruit la structure 
    attendue par les fonctions de décodage I et P.
    """
    encoded_stream = []
    iterator = iter(flat_data)
    
    # Calcul du nombre de blocs 8x8 pour les I-frames
    # et de macroblocs 16x16 pour les P-frames
    h_pad = (8 - height % 8) % 8 + height
    w_pad = (8 - width % 8) % 8 + width
    num_blocks_i = (h_pad // 8) * (w_pad // 8)
    
    num_blocks_p_x = width // 16
    num_blocks_p_y = height // 16
    num_macroblocks_p = num_blocks_p_x * num_blocks_p_y

    try:
        while True:
            # 1. On lit le type de frame ('I' ou 'P')
            frame_type = next(iterator)
            
            if frame_type == "I":
                rle_blocks = []
                # On sait qu'une I-frame contient un certain nombre de blocs 8x8
                for _ in range(num_blocks_i):
                    block_rle = []
                    # Chaque bloc RLE se termine par un marqueur de fin (0,0) ou le compte total de coefficients
                    # Dans ton cas, on extrait les paires (count, value) jusqu'à ce que le bloc soit complet
                    # Une approche simple si tu as stocké des listes de tuples :
                    pair = next(iterator)
                    # On boucle tant que la paire appartient au bloc (dépend de comment flatten l'a écrit)
                    # Pour coller à ton flatten_stream actuel qui met tout à la suite :
                    # Ton code met des tuples (run, val) directement dans la liste plate.
                    
                    # Attention : lire les éléments du RLE. Comme chaque bloc a un nombre variable de paires,
                    # une astuce consiste à reconstruire selon la logique de ton flatten.
                    pass
                    
    except StopIteration:
        pass
        
    return encoded_stream