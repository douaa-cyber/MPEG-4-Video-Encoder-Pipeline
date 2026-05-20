import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_pipeline(
    original_frames,
    ycbcr_frames,
    encoded_stream,
    reconstructed_frames,
    motion_vectors,
    residual_blocks,
    fq=1
):
    n_cols = 10
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("MPEG-4 Encoder Pipeline — Visualisation Complète (GOP = 10)", fontsize=18, fontweight='bold')

    # ── 1. Original frames ────────────────────────────────────────────────────
    n_show = min(n_cols, len(original_frames))
    for k in range(n_show):
        ax = plt.subplot2grid((4, n_cols), (0, k))
        ax.imshow(cv.cvtColor(original_frames[k], cv.COLOR_BGR2RGB))
        frame_type = encoded_stream[k][0] if k < len(encoded_stream) else "?"
        ax.set_title(f"Orig {k} [{frame_type}]", fontsize=10, fontweight='bold')
        ax.axis('off')

    # ── 2. Y, Cb, Cr channels of frame 0 ──────────────────────────────────────
    ycbcr0 = ycbcr_frames[0]
    Y, Cb, Cr = ycbcr0
    channels = [Y, Cb, Cr]
    titles   = ["Y channel", "Cb channel", "Cr channel"]
    cmaps    = ["gray", "Blues", "Reds"]

    for k in range(3):
        ax = plt.subplot2grid((4, n_cols), (1, k))
        ax.imshow(channels[k], cmap=cmaps[k])
        ax.set_title(titles[k], fontsize=9)
        ax.axis('off')

    # ── 3. DCT & Quantisation on one 8x8 block ────────────────────────────────
    Q = np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=float)
    y_channel = ycbcr_frames[0][0].astype(np.float32)
    block_raw = y_channel[0:8, 0:8]
    block_centered = block_raw - 128.0
    block_dct = cv.dct(block_centered)
    block_q   = np.round(block_dct / Q)
    block_rec = cv.idct(np.float32(block_q * Q)) + 128.0

    stages      = [block_raw, block_dct, block_q, block_rec]
    stage_names = ["Raw 8×8 block", "DCT coeffs", "Quantised", "Reconstructed"]

    for k, (stage, name) in enumerate(zip(stages, stage_names)):
        ax = plt.subplot2grid((4, n_cols), (1, 5 + k))
        im = ax.imshow(stage, cmap='viridis', aspect='auto')
        ax.set_title(name, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    
    p_idx = next((i for i, (t, _) in enumerate(encoded_stream) if t == "P"), None)

    # ── 4. Motion vectors on first P-frame ────────────────────────────────────
    ax4 = plt.subplot2grid((4, n_cols), (2, 0), colspan=5)

    if p_idx is not None and reconstructed_frames is not None and p_idx < len(reconstructed_frames):
        img_to_show = reconstructed_frames[p_idx]
        if len(img_to_show.shape) == 3:
            ax4.imshow(cv.cvtColor(img_to_show, cv.COLOR_BGR2RGB))
        else:
            ax4.imshow(img_to_show, cmap='gray')

        ax4.set_title(f"Motion vectors — frame {p_idx} [P]", fontsize=10, fontweight='bold')

        # Stats
        mvs = np.array([mv for _, mv in motion_vectors])
        zero_ratio = np.mean((mvs == 0).all(axis=1)) * 100
        scale = 4
        step  = max(1, len(motion_vectors) // 80)  # ← moins de flèches
        max_mag = max(np.sqrt(mvs[:,0]**2 + mvs[:,1]**2).max(), 1)

        for idx, ((i, j), (dy, dx)) in enumerate(motion_vectors):
            if idx % step == 0:
                if dy == 0 and dx == 0:
                    continue

                # Couleur selon amplitude : jaune=faible, rouge=fort
                magnitude = np.sqrt(dy**2 + dx**2)
                color = plt.cm.autumn(magnitude / max_mag)

                ax4.annotate(
                    "",
                    xy=(j + 8 + dx * scale, i + 8 + dy * scale),
                    xytext=(j + 8, i + 8),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0)
                )
    else:
        ax4.text(0.5, 0.5, "No P-frame or reconstruction data", ha='center', va='center')
        ax4.set_title("Motion vectors", fontsize=9)

    ax4.axis('off')

    # ── 5. Residuals & reconstruction ─────────────────────────────────────────
    ax5 = plt.subplot2grid((4, n_cols), (2, 5), colspan=5)

    if p_idx is not None and reconstructed_frames is not None and p_idx < len(reconstructed_frames):
        img_shape = reconstructed_frames[p_idx].shape
        H, W = img_shape[0], img_shape[1]
        residual_map = np.zeros((H, W), dtype=np.float32)

        Q_mat = np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=float)

        for (i, j), rle_sub_blocks in residual_blocks:
            for k, (x, y) in enumerate([(0, 0), (0, 8), (8, 0), (8, 8)]):
                if i + x + 8 <= H and j + y + 8 <= W:
                    from src.entropy import rle_decode, inverse_zigzag

                    zz = rle_decode(rle_sub_blocks[k])
                    block_quant = inverse_zigzag(zz)
                    block_dct = block_quant * Q_mat
                    block_spatial = cv.idct(np.float32(block_dct))
                    residual_map[i+x:i+x+8, j+y:j+y+8] = block_spatial

        im5 = ax5.imshow(np.abs(residual_map), cmap='hot')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        ax5.set_title(f"Spatial Residual Energy — frame {p_idx} [P]", fontsize=10, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, "No P-frame data found", ha='center', va='center')
        ax5.set_title("Residual map", fontsize=9)

    ax5.axis('off')

    # ── 6. Reconstructed frames ───────────────────────────────────────────────
    if reconstructed_frames is not None and len(reconstructed_frames) > 0:
        n_rec_show = min(n_cols, len(reconstructed_frames))
        for k in range(n_rec_show):
            ax = plt.subplot2grid((4, n_cols), (3, k))
            rec_img = reconstructed_frames[k]
            if len(rec_img.shape) == 3:
                ax.imshow(cv.cvtColor(rec_img, cv.COLOR_BGR2RGB))
            else:
                ax.imshow(rec_img, cmap='gray')
            ax.set_title(f"Rec {k}", fontsize=10, fontweight='bold')
            ax.axis('off')
    else:
        ax_warn = plt.subplot2grid((4, n_cols), (3, 0), colspan=n_cols)
        ax_warn.text(0.5, 0.5, "No reconstructed frames available yet", ha='center', va='center', color='red')
        ax_warn.axis('off')

    plt.tight_layout()
    plt.savefig("pipeline_visualisation_gop10.png", dpi=150, bbox_inches='tight')
    print("Saved pipeline_visualisation_gop10.png")
    plt.show()