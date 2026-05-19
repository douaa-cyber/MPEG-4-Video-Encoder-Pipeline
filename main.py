from src.preprocessing import load_frames, preprocess_frames
from src.i_frame import encode_i_frame, decode_i_frame
from src.p_frame import encode_p_frame, decode_p_frame
from src.entropy import (
    build_huffman_tree,
    build_codes,
    huffman_encode,
    flatten_stream,
    save_to_bin
)
from evaluation import print_metrics
from visualisation import visualize_pipeline

# ======================
# PARAMETERS
# ======================
FQ = 10   # quantization factor
S  = 4    # motion search window
G  = 10   # GOP size (every G-th frame is an I-frame)

# ======================
# LOAD & PREPROCESS
# ======================
frames       = load_frames("frames/")
ycbcr_frames = preprocess_frames(frames)

encoded_stream       = []
reconstructed_frames = []   # Y-channel reconstructions for metrics/visualisation
reference_frame      = None

last_mv       = None   # motion vectors of last P-frame (for visualisation)
last_residuals = None  # residuals of last P-frame (for visualisation)

# ======================
# ENCODE
# ======================
for i, frame in enumerate(ycbcr_frames):

    y = frame[:, :, 0]
    original_shape = y.shape

    if i % G == 0:
        # ----- I-FRAME -----
        rle_blocks     = encode_i_frame(y, fq=FQ)
        reference_frame = decode_i_frame(rle_blocks, original_shape, fq=FQ)

        encoded_stream.append(("I", (rle_blocks, original_shape)))
        reconstructed_frames.append(reference_frame)
        print(f"I-frame: {i}")

    else:
        # ----- P-FRAME -----
        mv, residuals   = encode_p_frame(y, reference_frame, fq=FQ, S=S)
        reconstructed   = decode_p_frame(reference_frame, mv, residuals, fq=FQ)
        reference_frame = reconstructed

        encoded_stream.append(("P", (mv, residuals)))
        reconstructed_frames.append(reconstructed)

        last_mv        = mv
        last_residuals = residuals
        print(f"P-frame: {i}")

# ======================
# ENTROPY CODING — single .bin file
# ======================
flat_data = flatten_stream(encoded_stream)
tree      = build_huffman_tree(flat_data)
codebook  = build_codes(tree)
bitstream = huffman_encode(flat_data, codebook)

save_to_bin(bitstream, tree, filename="video.bin")

# ======================
# EVALUATION
# ======================
original_y_frames = [f[:, :, 0] for f in ycbcr_frames]

print_metrics(
    original_frames=original_y_frames,
    reconstructed_frames=reconstructed_frames,
    encoded_stream=encoded_stream,
    bin_path="video.bin"
)

# ======================
# VISUALISATION
# ======================
visualize_pipeline(
    original_frames=frames,
    ycbcr_frames=ycbcr_frames,
    encoded_stream=encoded_stream,
    reconstructed_frames=reconstructed_frames,
    motion_vectors=last_mv,
    residual_blocks=last_residuals,
    fq=FQ
)