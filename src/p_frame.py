import numpy as np
import cv2 as cv
from src.entropy import zigzag_scan, rle_encode, rle_decode, inverse_zigzag


def motion_estimation(block, ref, i, j, S=4):
    """
    Find best matching 16x16 block in reference frame using MSE.
    Returns motion vector (dy, dx) and predicted block.
    """
    N = 16
    H, W = ref.shape

    best_mse = float("inf")
    best_mv = (0, 0)
    best_block = np.zeros((16, 16), dtype=np.float32)

    for dy in range(-S, S + 1):
        for dx in range(-S, S + 1):
            y = i + dy
            x = j + dx

            if y < 0 or x < 0 or y + N > H or x + N > W:
                continue

            candidate = ref[y:y + N, x:x + N]
            mse = np.mean((block - candidate) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_mv = (dy, dx)
                best_block = candidate

    return best_mv, best_block


def encode_p_frame(current_frame, reference_frame, fq=10, S=4):
    """
    Encode a P-frame using:
    - 16x16 macroblocks
    - motion estimation in ±S window
    - residual DCT + quantization + zigzag + RLE
    """
    if len(current_frame.shape) == 3:
        current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2YCrCb)[:, :, 0]

    if len(reference_frame.shape) == 3:
        reference_frame = cv.cvtColor(reference_frame, cv.COLOR_BGR2YCrCb)[:, :, 0]

    H, W = current_frame.shape
    Q = np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=int)

    motion_vectors = []
    residual_blocks = []

    for i in range(0, H, 16):
        for j in range(0, W, 16):
            block = current_frame[i:i + 16, j:j + 16]

            if block.shape != (16, 16):
                continue

            # 1. Motion estimation
            mv, pred_block = motion_estimation(block, reference_frame, i, j, S)

            # 2. Residual
            residual = np.float32(block) - np.float32(pred_block)

            # 3. DCT + Quantization + Zigzag + RLE on each 8x8 sub-block
            rle_sub_blocks = []

            for x in range(0, 16, 8):
                for y in range(0, 16, 8):
                    sub = np.float32(residual[x:x+8, y:y+8])
                    dct = cv.dct(sub)
                    quant = np.floor(dct / Q)

                    zz = zigzag_scan(quant)
                    rle = rle_encode(zz)
                    rle_sub_blocks.append(rle)

            motion_vectors.append(((i, j), mv))
            residual_blocks.append(((i, j), rle_sub_blocks))

    return motion_vectors, residual_blocks


def decode_p_frame(reference_frame, motion_vectors, residual_blocks, fq=10):
    """
    Reconstruct P-frame using motion compensation + residual decoding.
    """
    if len(reference_frame.shape) == 3:
        reference_frame = cv.cvtColor(reference_frame, cv.COLOR_BGR2YCrCb)[:, :, 0]

    H, W = reference_frame.shape
    result = reference_frame.astype(np.float32).copy()

    Q = np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=int)

    for (i, j), rle_sub_blocks in residual_blocks:

        # Get motion vector
        mv = next(m for pos, m in motion_vectors if pos == (i, j))
        dy, dx = mv

        # Prediction from reference frame
        pred = reference_frame[i + dy:i + dy + 16, j + dx:j + dx + 16]

        if pred.shape != (16, 16):
            continue

        # Reconstruct residual from RLE sub-blocks
        residual = np.zeros((16, 16), dtype=np.float32)

        for k, (x, y) in enumerate([(0, 0), (0, 8), (8, 0), (8, 8)]):
            zz = rle_decode(rle_sub_blocks[k])
            block = inverse_zigzag(zz)
            dequant = block * Q
            idct = cv.idct(np.float32(dequant))
            residual[x:x+8, y:y+8] = idct

        # Final reconstruction
        result[i:i+16, j:j+16] = pred + residual

    return np.uint8(np.clip(result, 0, 255))