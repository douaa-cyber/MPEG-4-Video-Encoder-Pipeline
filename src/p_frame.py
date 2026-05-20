import numpy as np
import cv2 as cv
from src.entropy import zigzag_scan, rle_encode, rle_decode, inverse_zigzag


def motion_estimation(block, ref, i, j, S=4):
    N = 16
    H, W = ref.shape

    best_mv    = (0, 0)
    best_mse   = np.mean((block - ref[i:i+N, j:j+N].astype(np.float32)) ** 2)
    best_block = ref[i:i+N, j:j+N]

    step = S
    cy, cx = i, j

    while step >= 1:
        for dy in [-step, 0, step]:
            for dx in [-step, 0, step]:
                y, x = cy + dy, cx + dx
                if y < 0 or x < 0 or y + N > H or x + N > W:
                    continue

                # ← Contrainte : vecteur total ne dépasse pas S
                if abs(y - i) > S or abs(x - j) > S:
                    continue

                cand = ref[y:y+N, x:x+N].astype(np.float32)
                mse  = np.mean((block - cand) ** 2)
                if mse < best_mse:
                    best_mse   = mse
                    cy, cx     = y, x
                    best_block = cand

        step //= 2

    return (cy - i, cx - j), best_block

def encode_channel_residuals(current_ch, reference_ch, Q, block_size=16):
    """
    Encode residuals (DCT + quantization + zigzag + RLE) for one channel.
    Returns list of (position, rle_sub_blocks).
    """
    H, W = current_ch.shape
    residual_blocks = []

    for i in range(0, H - block_size + 1, block_size):
        for j in range(0, W - block_size + 1, block_size):
            block = current_ch[i:i + block_size, j:j + block_size]
            pred  = reference_ch[i:i + block_size, j:j + block_size]

            residual = np.float32(block) - np.float32(pred)

            rle_sub_blocks = []
            for x in range(0, block_size, 8):
                for y in range(0, block_size, 8):
                    sub   = np.float32(residual[x:x+8, y:y+8])
                    dct   = cv.dct(sub)
                    quant = np.floor(dct / Q)
                    zz    = zigzag_scan(quant)
                    rle   = rle_encode(zz)
                    rle_sub_blocks.append(rle)

            residual_blocks.append(((i, j), rle_sub_blocks))

    return residual_blocks


def decode_channel_residuals(reference_ch, motion_vectors, residual_blocks, Q, block_size=16):
    """
    Decode residuals for one channel and reconstruct it.
    motion_vectors: list of (position, (dy, dx)) — used only for Y channel.
                    For Cb/Cr, motion vectors are scaled by 0.5 (4:2:0 subsampling).
    """
    H, W = reference_ch.shape
    result = reference_ch.astype(np.float32).copy()

    for (i, j), rle_sub_blocks in residual_blocks:
        if i + block_size > H or j + block_size > W:
            continue

        residual = np.zeros((block_size, block_size), dtype=np.float32)

        for k, (x, y) in enumerate([(0, 0), (0, 8), (8, 0), (8, 8)]):
            zz    = rle_decode(rle_sub_blocks[k])
            block = inverse_zigzag(zz)
            dequant = block * Q
            idct    = cv.idct(np.float32(dequant))
            residual[x:x+8, y:y+8] = idct

        result[i:i+block_size, j:j+block_size] += residual

    return np.clip(result, 0, 255).astype(np.uint8)


def encode_p_frame(current_ycbcr, reference_ycbcr, fq=1, S=4):
    """
    Encode a P-frame using all three channels (Y, Cb, Cr).

    current_ycbcr  : tuple (Y, Cb_sub, Cr_sub) — current frame (subsampled chroma)
    reference_ycbcr: tuple (Y, Cb_sub, Cr_sub) — previous reconstructed frame (same format)

    Returns:
        motion_vectors : list of ((i, j), (dy, dx)) — estimated on Y channel
        residuals_y    : residual blocks for Y
        residuals_cb   : residual blocks for Cb (at half resolution, no motion estimation)
        residuals_cr   : residual blocks for Cr (at half resolution, no motion estimation)
    """
    Y_cur,  Cb_cur,  Cr_cur  = current_ycbcr
    Y_ref,  Cb_ref,  Cr_ref  = reference_ycbcr

    Q = np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=int)

    H, W = Y_cur.shape
    motion_vectors  = []
    residuals_y     = []

    # ── Y channel: full motion estimation + residual ──────────────────────────
    for i in range(0, H - 15, 16):
        for j in range(0, W - 15, 16):
            block = Y_cur[i:i + 16, j:j + 16].astype(np.float32)

            mv, pred_block = motion_estimation(block, Y_ref.astype(np.float32), i, j, S)

            residual = block - np.float32(pred_block)

            rle_sub_blocks = []
            for x in range(0, 16, 8):
                for y_off in range(0, 16, 8):
                    sub   = np.float32(residual[x:x+8, y_off:y_off+8])
                    dct   = cv.dct(sub)
                    quant = np.floor(dct / Q)
                    zz    = zigzag_scan(quant)
                    rle   = rle_encode(zz)
                    rle_sub_blocks.append(rle)

            motion_vectors.append(((i, j), mv))
            residuals_y.append(((i, j), rle_sub_blocks))

    # ── Cb / Cr channels: motion-compensated with scaled vectors (no new ME) ──
    # Chroma is at half resolution (4:2:0), so we scale Y motion vectors by 0.5
    residuals_cb = _encode_chroma_residuals(Cb_cur, Cb_ref, motion_vectors, Q)
    residuals_cr = _encode_chroma_residuals(Cr_cur, Cr_ref, motion_vectors, Q)

    return motion_vectors, residuals_y, residuals_cb, residuals_cr


def _encode_chroma_residuals(ch_cur, ch_ref, motion_vectors, Q):
    """
    Encode chroma residuals using scaled motion vectors from Y channel.
    Chroma blocks are 8×8 (half of Y's 16×16).
    """
    H, W = ch_cur.shape
    residual_blocks = []

    for (i_y, j_y), (dy, dx) in motion_vectors:
        # Scale position and MV to chroma resolution
        i  = i_y  // 2
        j  = j_y  // 2
        dy_c = dy // 2
        dx_c = dx // 2

        if i + 8 > H or j + 8 > W:
            continue

        # Predicted block from reference (motion-compensated)
        ref_i = i + dy_c
        ref_j = j + dx_c

        if ref_i < 0 or ref_j < 0 or ref_i + 8 > H or ref_j + 8 > W:
            # Out-of-bounds: use zero prediction
            pred = np.zeros((8, 8), dtype=np.float32)
        else:
            pred = ch_ref[ref_i:ref_i+8, ref_j:ref_j+8].astype(np.float32)

        block    = ch_cur[i:i+8, j:j+8].astype(np.float32)
        residual = block - pred

        dct   = cv.dct(np.float32(residual))
        quant = np.floor(dct / Q)
        zz    = zigzag_scan(quant)
        rle   = rle_encode(zz)

        residual_blocks.append(((i, j), [rle]))  # single 8×8 block

    return residual_blocks


def decode_p_frame(reference_ycbcr, motion_vectors, residuals_y, residuals_cb, residuals_cr, fq=1):
    """
    Reconstruct a P-frame (Y, Cb_sub, Cr_sub) from reference + encoded data.

    reference_ycbcr: tuple (Y, Cb_sub, Cr_sub)

    Returns tuple (Y_rec, Cb_rec, Cr_rec) — same subsampled format as input.
    """
    Y_ref, Cb_ref, Cr_ref = reference_ycbcr

    Q = np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=int)

    # ── Reconstruct Y ─────────────────────────────────────────────────────────
    H, W = Y_ref.shape
    Y_rec = Y_ref.astype(np.float32).copy()

    for (i, j), rle_sub_blocks in residuals_y:
        mv = next(m for pos, m in motion_vectors if pos == (i, j))
        dy, dx = mv

        pred = Y_ref[i + dy:i + dy + 16, j + dx:j + dx + 16]
        if pred.shape != (16, 16):
            continue

        residual = np.zeros((16, 16), dtype=np.float32)
        for k, (x, y_off) in enumerate([(0, 0), (0, 8), (8, 0), (8, 8)]):
            zz      = rle_decode(rle_sub_blocks[k])
            block   = inverse_zigzag(zz)
            dequant = block * Q
            idct    = cv.idct(np.float32(dequant))
            residual[x:x+8, y_off:y_off+8] = idct

        Y_rec[i:i+16, j:j+16] = pred + residual

    Y_rec = np.clip(Y_rec, 0, 255).astype(np.uint8)

    # ── Reconstruct Cb / Cr ───────────────────────────────────────────────────
    Cb_rec = _decode_chroma_residuals(Cb_ref, motion_vectors, residuals_cb, Q)
    Cr_rec = _decode_chroma_residuals(Cr_ref, motion_vectors, residuals_cr, Q)

    return Y_rec, Cb_rec, Cr_rec


def _decode_chroma_residuals(ch_ref, motion_vectors, residual_blocks, Q):
    """
    Decode chroma channel using scaled motion vectors.
    """
    H, W = ch_ref.shape
    result = ch_ref.astype(np.float32).copy()

    # Build a fast lookup: chroma position → rle block
    res_map = {pos: rle_list for pos, rle_list in residual_blocks}
    mv_map  = {(i_y // 2, j_y // 2): (dy // 2, dx // 2)
               for (i_y, j_y), (dy, dx) in motion_vectors}

    for (i, j), rle_list in res_map.items():
        if i + 8 > H or j + 8 > W:
            continue

        dy_c, dx_c = mv_map.get((i, j), (0, 0))
        ref_i = i + dy_c
        ref_j = j + dx_c

        if ref_i < 0 or ref_j < 0 or ref_i + 8 > H or ref_j + 8 > W:
            pred = np.zeros((8, 8), dtype=np.float32)
        else:
            pred = ch_ref[ref_i:ref_i+8, ref_j:ref_j+8].astype(np.float32)

        zz      = rle_decode(rle_list[0])
        block   = inverse_zigzag(zz)
        dequant = block * Q
        idct    = cv.idct(np.float32(dequant))

        result[i:i+8, j:j+8] = pred + idct

    return np.clip(result, 0, 255).astype(np.uint8)