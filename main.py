import cv2
import numpy as np
import pickle
import bz2
from src.preprocessing import load_frames, preprocess_frames
from src.i_frame import encode_i_frame, decode_i_frame
from src.p_frame import encode_p_frame, decode_p_frame
from src.entropy import flatten_stream
from evaluation import print_metrics
from visualisation import visualize_pipeline

FQ = 1
S  = 4
G  = 10

print("========== PRE-PROCESSING ==========")
frames       = load_frames("frames/")
ycbcr_frames = preprocess_frames(frames)
print(f"Chargement réussi : {len(frames)} frames prêtes.\n")

encoded_stream       = []
reconstructed_frames = []
reference_ycbcr      = None
last_mv              = None
last_residuals       = None

print("========== ENCODING ==========")
for i, frame_data in enumerate(ycbcr_frames):
    y, cb_sub, cr_sub = frame_data

    if i % G == 0:
        rle_blocks, shapes = encode_i_frame([y, cb_sub, cr_sub], fq=FQ)
        bgr_rec = decode_i_frame(rle_blocks, shapes, fq=FQ)

        ycrcb_rec   = cv2.cvtColor(bgr_rec, cv2.COLOR_BGR2YCrCb)
        Y_rec       = ycrcb_rec[:, :, 0]
        Cr_rec_full = ycrcb_rec[:, :, 1]
        Cb_rec_full = ycrcb_rec[:, :, 2]

        h, w = shapes[0]
        Cb_rec_sub = cv2.resize(Cb_rec_full, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        Cr_rec_sub = cv2.resize(Cr_rec_full, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        reference_ycbcr = (Y_rec, Cb_rec_sub, Cr_rec_sub)
        encoded_stream.append(("I", (rle_blocks, shapes)))
        reconstructed_frames.append(bgr_rec)
        print(f"Frame {i:03d} encodée avec succès [Type: I]")

    else:
        mv, res_y, res_cb, res_cr = encode_p_frame(frame_data, reference_ycbcr, fq=FQ, S=S)
        Y_rec, Cb_rec, Cr_rec = decode_p_frame(reference_ycbcr, mv, res_y, res_cb, res_cr, fq=FQ)

        h, w = y.shape
        Cb_up = cv2.resize(Cb_rec, (w, h))
        Cr_up = cv2.resize(Cr_rec, (w, h))
        ycrcb_merged = cv2.merge([Y_rec, Cr_up, Cb_up])
        bgr_rec = cv2.cvtColor(ycrcb_merged, cv2.COLOR_YCrCb2BGR)

        reference_ycbcr = (Y_rec, Cb_rec, Cr_rec)
        encoded_stream.append(("P", (mv, res_y, res_cb, res_cr)))
        reconstructed_frames.append(bgr_rec)
        last_mv        = mv
        last_residuals = res_y
        print(f"Frame {i:03d} encodée avec succès [Type: P]")

print("\n========== ENTROPY & FILE COMPRESSION ==========")
flat_data = flatten_stream(encoded_stream)
with bz2.open("video.bin", "wb") as f:
    pickle.dump(flat_data, f)
print("Fichier vidéo compressé avec succès via bin !")

print("\n========== EVALUATION ==========")
original_y_frames      = [Y for (Y, Cb, Cr) in ycbcr_frames]
reconstructed_y_frames = []
for bgr in reconstructed_frames:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    reconstructed_y_frames.append(ycrcb[:, :, 0])

print_metrics(
    original_frames=original_y_frames,
    reconstructed_frames=reconstructed_y_frames,
    encoded_stream=encoded_stream,
    bin_path="video.bin"
)

print("\n========== VISUALISATION ==========")
visualize_pipeline(
    original_frames=frames,
    ycbcr_frames=ycbcr_frames,
    encoded_stream=encoded_stream,
    reconstructed_frames=reconstructed_frames,
    motion_vectors=last_mv,
    residual_blocks=last_residuals,
    fq=FQ
)