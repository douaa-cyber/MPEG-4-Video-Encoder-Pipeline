import numpy as np
import cv2 as cv
from src.entropy import zigzag_scan, rle_encode, rle_decode, inverse_zigzag


def pad_image(img):
    h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    return np.pad(img, ((0, pad_h), (0, pad_w)), 'constant')


def get_quant_matrix(fq=10):
    return np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=int)


def encode_i_frame(img, fq=10):
    img = img.astype(np.float32)
    padImg = pad_image(img)
    h_p, w_p = padImg.shape
    Q = get_quant_matrix(fq)

    rle_blocks = []

    for i in range(0, h_p, 8):
        for j in range(0, w_p, 8):
            block = padImg[i:i+8, j:j+8]

            # DCT
            dct = cv.dct(block)

            # Quantization
            quant = np.floor(dct / Q)

            # Zigzag + RLE
            zz = zigzag_scan(quant)
            rle = rle_encode(zz)
            rle_blocks.append(rle)

    return rle_blocks


def decode_i_frame(rle_blocks, original_shape, fq=10):
    h, w = original_shape
    Q = get_quant_matrix(fq)

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    h_p = h + pad_h
    w_p = w + pad_w

    result = np.zeros((h_p, w_p), dtype=np.float32)
    idx = 0

    for i in range(0, h_p, 8):
        for j in range(0, w_p, 8):
            zz = rle_decode(rle_blocks[idx])
            block = inverse_zigzag(zz)
            dequant = block * Q
            result[i:i+8, j:j+8] = cv.idct(np.float32(dequant))
            idx += 1

    return np.uint8(np.clip(result[:h, :w], 0, 255))