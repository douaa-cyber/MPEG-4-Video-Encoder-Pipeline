import numpy as np
import os


def compute_psnr(original, reconstructed):
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)


def compression_ratio(original_frames, bin_path="video.bin"):
    original_size = sum(f.nbytes for f in original_frames)
    compressed_size = os.path.getsize(bin_path)
    return original_size / compressed_size


def frame_breakdown(encoded_stream):
    i_count = sum(1 for t, _ in encoded_stream if t == "I")
    p_count = sum(1 for t, _ in encoded_stream if t == "P")
    total = i_count + p_count
    print(f"Total frames : {total}")
    print(f"  I-frames   : {i_count}")
    print(f"  P-frames   : {p_count}")
    return i_count, p_count


def print_metrics(original_frames, reconstructed_frames, encoded_stream, bin_path="video.bin"):
    print("\n========== METRICS ==========")

    # Frame breakdown
    i_count, p_count = frame_breakdown(encoded_stream)

    # Compression ratio
    ratio = compression_ratio(original_frames, bin_path)
    print(f"\nCompression ratio : {ratio:.2f}x")

    # PSNR per frame
    print("\nPSNR per frame:")
    for i, (orig, recon) in enumerate(zip(original_frames, reconstructed_frames)):
        frame_type = encoded_stream[i][0]
        psnr = compute_psnr(orig, recon)
        print(f"  Frame {i:03d} [{frame_type}] : {psnr:.2f} dB")

    avg_psnr = np.mean([
        compute_psnr(o, r)
        for o, r in zip(original_frames, reconstructed_frames)
        if compute_psnr(o, r) != float('inf')
    ])
    print(f"\nAverage PSNR : {avg_psnr:.2f} dB")
    print("=============================\n")