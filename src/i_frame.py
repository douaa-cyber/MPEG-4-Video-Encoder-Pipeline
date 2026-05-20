import numpy as np
import cv2 as cv
from src.entropy import zigzag_scan, rle_encode, rle_decode, inverse_zigzag


# =========================
# Padding
# =========================
def pad_image(img):
    h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant'), (h, w)


# =========================
# Quant matrix
# =========================
def get_quant_matrix(fq=1):
    return np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=int)


# =========================
# ENCODE I-FRAME
# =========================
def encode_i_frame(channels, fq=1):
    """
    channels = [Y, Cb, Cr] 
    Prend en charge le format 4:2:0 (où Cb et Cr sont plus petits que Y)
    """
    result = []
    shapes = []

    for img in channels:
        img = img.astype(np.float32)

        padImg, orig_shape = pad_image(img)
        shapes.append(orig_shape)

        h_p, w_p = padImg.shape
        Q = get_quant_matrix(fq)

        rle_blocks = []

        for i in range(0, h_p, 8):
            for j in range(0, w_p, 8):
                block = padImg[i:i+8, j:j+8]

                dct = cv.dct(block)
                quant = np.floor(dct / Q)

                zz = zigzag_scan(quant)
                rle = rle_encode(zz)

                rle_blocks.append(rle)

        result.append(rle_blocks)

    return result, shapes


# =========================
# DECODE I-FRAME (CORRIGÉ)
# =========================
def decode_i_frame(rle_channels, shapes, fq=1):
    """
    Décode les canaux, gère le sous-échantillonnage 4:2:0
    et corrige l'inversion de couleur mauve/magenta.
    """
    Q = get_quant_matrix(fq)

    Y_blocks, Cb_blocks, Cr_blocks = rle_channels
    
    # Récupération des formes d'origine de CHAQUE canal (crucial pour le 4:2:0)
    (h_y, w_y) = shapes[0]
    (h_cb, w_cb) = shapes[1]
    (h_cr, w_cr) = shapes[2]

    def decode_single(rle_blocks, h, w):
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        h_p, w_p = h + pad_h, w + pad_w
        result = np.zeros((h_p, w_p), dtype=np.float32)

        expected_blocks = (h_p // 8) * (w_p // 8)

        if len(rle_blocks) != expected_blocks:
            raise ValueError(
                f"[I-FRAME ERROR] Block mismatch: expected {expected_blocks}, got {len(rle_blocks)}"
            )

        idx = 0

        for i in range(0, h_p, 8):
            for j in range(0, w_p, 8):
                zz = rle_decode(rle_blocks[idx])
                block = inverse_zigzag(zz)
                dequant = block * Q
                result[i:i+8, j:j+8] = cv.idct(np.float32(dequant))
                idx += 1

        return result[:h, :w]

    # Décodage individuel de chaque canal selon sa propre résolution d'origine
    Y  = decode_single(Y_blocks, h_y, w_y)
    Cb = decode_single(Cb_blocks, h_cb, w_cb)
    Cr = decode_single(Cr_blocks, h_cr, w_cr)

    # GESTION DU SOUS-ÉCHANTILLONNAGE 4:2:0
    # Si Cb et Cr sont plus petits que Y (4:2:0), on les redimensionne à la taille de Y
    if (h_cb, w_cb) != (h_y, w_y):
        Cb = cv.resize(Cb, (w_y, h_y), interpolation=cv.INTER_LINEAR)
    if (h_cr, w_cr) != (h_y, w_y):
        Cr = cv.resize(Cr, (w_y, h_y), interpolation=cv.INTER_LINEAR)

    # encode_i_frame reçoit [Y, Cb, Cr] → shapes[1]=Cb, shapes[2]=Cr
    # COLOR_YCrCb2BGR attend l'ordre [Y, Cr, Cb]
    ycrcb = cv.merge([Y, Cr, Cb])
    ycrcb = np.clip(ycrcb, 0, 255).astype(np.uint8)
    return cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)