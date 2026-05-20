import numpy as np

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
            encoded.append((count, int(val)))
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
# FLATTEN / UNFLATTEN HELPERS
# ======================
def _flatten_rle_blocks(flat, rle_blocks):
    """Append a list of RLE blocks (each block = list of (zeros, val) pairs)."""
    flat.append(len(rle_blocks))
    for block in rle_blocks:
        flat.append(len(block))
        for pair in block:
            flat.append(pair)


def _flatten_residuals(flat, residuals):
    """Append residual_blocks structure: list of (pos, [rle_block, ...])."""
    flat.append(len(residuals))
    for _, rle_sub_blocks in residuals:
        flat.append(len(rle_sub_blocks))
        for block in rle_sub_blocks:
            flat.append(len(block))
            for pair in block:
                flat.append(pair)



# ======================
# FLATTEN STREAM
# ======================
def flatten_stream(encoded_stream):
    """
    Convert structured stream → flat list.

    P-frame content is now (mv, res_y, res_cb, res_cr).
    """
    flat = []

    for frame_type, content in encoded_stream:
        flat.append(frame_type)

        if frame_type == "I":
            rle_blocks, shape = content
            flat.append(shape)
            flat.append(len(rle_blocks))
            for channel_blocks in rle_blocks:
                _flatten_rle_blocks(flat, channel_blocks)

        elif frame_type == "P":
            mv, res_y, res_cb, res_cr = content

            # Motion vectors (positions + vectors)
            flat.append(len(mv))
            for _, v in mv:
                flat.append(tuple(v))

            # Residuals for each channel
            _flatten_residuals(flat, res_y)
            _flatten_residuals(flat, res_cb)
            _flatten_residuals(flat, res_cr)

    return flat
