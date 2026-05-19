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
    fq=10
):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("MPEG-4 Encoder Pipeline — Visualisation", fontsize=16, fontweight='bold')

    # ── 1. Original frames (show first 4) ─────────────────────────────────────
    n_show = min(4, len(original_frames))
    for k in range(n_show):
        ax = fig.add_subplot(4, n_show, k + 1)
        ax.imshow(cv.cvtColor(original_frames[k], cv.COLOR_BGR2RGB))
        frame_type = encoded_stream[k][0]
        ax.set_title(f"Frame {k} [{frame_type}]", fontsize=9)
        ax.axis('off')

    # ── 2. Y, Cb, Cr channels of frame 0 ──────────────────────────────────────
        ycbcr0 = ycbcr_frames[0]

        Y, Cb, Cr = ycbcr0

        channels = [Y, Cb, Cr]
        titles   = ["Y channel", "Cb channel", "Cr channel"]
        cmaps    = ["gray", "Blues", "Reds"]
    for k in range(3):
        ax = fig.add_subplot(4, 4, 4 + k + 1)
        ax.imshow(channels[k], cmap=cmaps[k])
        ax.set_title(titles[k], fontsize=9)
        ax.axis('off')

    # ── 3. DCT & Quantisation on one 8x8 block ────────────────────────────────
    Q = np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=int)
    y_channel = ycbcr_frames[0][0].astype(np.float32)
    block_raw = y_channel[0:8, 0:8]
    block_dct = cv.dct(block_raw)
    block_q   = np.floor(block_dct / Q)
    block_rec = cv.idct(np.float32(block_q * Q))

    stages      = [block_raw, block_dct, block_q, block_rec]
    stage_names = ["Raw 8×8 block", "DCT coeffs", "Quantised", "Reconstructed"]

    for k, (stage, name) in enumerate(zip(stages, stage_names)):
        ax = fig.add_subplot(4, 4, 8 + k + 1)
        im = ax.imshow(stage, cmap='viridis', aspect='auto')
        ax.set_title(name, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    # ── 4. Motion vectors on first P-frame ────────────────────────────────────
    ax4 = fig.add_subplot(4, 2, 7)

    # Find first P-frame index
    p_idx = next((i for i, (t, _) in enumerate(encoded_stream) if t == "P"), None)

    if p_idx is not None and reconstructed_frames is not None:
        ax4.imshow(reconstructed_frames[p_idx], cmap='gray')
        ax4.set_title(f"Motion vectors — frame {p_idx} [P]", fontsize=9)

        step = max(1, len(motion_vectors) // 200)  # limit arrows for readability
        for idx, ((i, j), (dy, dx)) in enumerate(motion_vectors):
            if idx % step == 0:
                ax4.annotate(
                    "", xy=(j + dx + 8, i + dy + 8),
                    xytext=(j + 8, i + 8),
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.6)
                )
    else:
        ax4.text(0.5, 0.5, "No P-frame found", ha='center', va='center')
        ax4.set_title("Motion vectors", fontsize=9)
    ax4.axis('off')

   # ── 5. Residuals & reconstruction ─────────────────────────────────────────
    ax5 = fig.add_subplot(4, 2, 8)

    if p_idx is not None:
        H, W = reconstructed_frames[p_idx].shape
        residual_map = np.zeros((H, W), dtype=np.float32)
        
        # On recrée la matrice de quantification correspondante
        Q = np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=int)

        for (i, j), rle_sub_blocks in residual_blocks:
            for k, (x, y) in enumerate([(0, 0), (0, 8), (8, 0), (8, 8)]):
                if i + x + 8 <= H and j + y + 8 <= W:
                    from src.entropy import rle_decode, inverse_zigzag
                    
                    # 1. Décodage RLE et reconstruction du bloc zigzag
                    zz = rle_decode(rle_sub_blocks[k])
                    block_quant = inverse_zigzag(zz)
                    
                    # 2. CORRECTION : Déquantification
                    block_dct = block_quant * Q
                    
                    # 3. CORRECTION : Passage du domaine fréquentiel au domaine spatial (IDCT)
                    block_spatial = cv.idct(np.float32(block_dct))
                    
                    # Stockage du résidu spatial réel
                    residual_map[i+x:i+x+8, j+y:j+y+8] = block_spatial

        # 4. CORRECTION VISUELLE : On affiche la valeur absolue des résidus
        # pour mettre en évidence l'énergie du mouvement (les contours ressortiront en jaune/brillant)
        im5 = ax5.imshow(np.abs(residual_map), cmap='hot')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        ax5.set_title(f"Spatial Residual Energy — frame {p_idx} [P]", fontsize=9)
    else:
        ax5.text(0.5, 0.5, "No P-frame found", ha='center', va='center')
        ax5.set_title("Residual map", fontsize=9)
    ax5.axis('off')

    plt.tight_layout()
    plt.savefig("pipeline_visualisation.png", dpi=150, bbox_inches='tight')
    print("Saved pipeline_visualisation.png")
    plt.show()