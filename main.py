import os
import cv2
import numpy as np

# Importations des modules locaux du projet
from src.preprocessing import load_frames, preprocess_frames
from src.i_frame import encode_i_frame, decode_i_frame
from src.p_frame import encode_p_frame, decode_p_frame
from src.entropy import (
    build_huffman_tree,
    build_codes,
    huffman_encode,
    flatten_stream,
    save_to_bin,
    load_from_bin,
    huffman_decode
)
from evaluation import print_metrics
from visualisation import visualize_pipeline

# ==========================================
# 1. PARAMÈTRES DU CODEC
# ==========================================
FQ = 10   # Facteur de quantification (Quantization factor)
S  = 4    # Fenêtre de recherche du mouvement (Motion search window)
G  = 10   # Taille du GOP (Chaque G-th frame est une I-frame)

# ==========================================
# 2. CHARGEMENT & PRÉ-TRAITEMENT (Chroma Subsampling)
# ==========================================
print("========== PRE-PROCESSING ==========")
frames       = load_frames("frames/")
ycbcr_frames = preprocess_frames(frames)
print(f"Chargement réussi : {len(frames)} frames prêtes.\n")

# Initialisation des structures pour l'encodage
encoded_stream       = []
reconstructed_frames = []   # Reconstructions du canal Y pour les métriques
reference_frame      = None

last_mv       = None   # Vecteurs de mouvement de la dernière P-frame (pour la visuelle)
last_residuals = None  # Résidus de la dernière P-frame (pour la visuelle)

# ==========================================
# 3. PIPELINE D'ENCODAGE (Spatio-Temporel)
# ==========================================
print("========== ENCODING ==========")
for i, frame_data in enumerate(ycbcr_frames):
    # Déballage du tuple YCbCr contenant le sous-échantillonnage de la couleur
    y, cb_sub, cr_sub = frame_data
    original_shape = y.shape

    if i % G == 0:
        # ----- INTRA-FRAME (I-FRAME) -----
        rle_blocks      = encode_i_frame(y, fq=FQ)
        reference_frame = decode_i_frame(rle_blocks, original_shape, fq=FQ)

        encoded_stream.append(("I", (rle_blocks, original_shape)))
        reconstructed_frames.append(reference_frame)
        print(f"Frame {i:03d} encodée avec succès [Type: I]")

    else:
        # ----- INTER-FRAME (P-FRAME) -----
        mv, residuals   = encode_p_frame(y, reference_frame, fq=FQ, S=S)
        reconstructed   = decode_p_frame(reference_frame, mv, residuals, fq=FQ)
        reference_frame = reconstructed

        encoded_stream.append(("P", (mv, residuals)))
        reconstructed_frames.append(reconstructed)

        # Sauvegarde temporaire pour la fonction de visualisation
        last_mv        = mv
        last_residuals = residuals
        print(f"Frame {i:03d} encodée avec succès [Type: P]")

# ==========================================
# 4. ENCODAGE PAR ENTROPIE (Compression sans perte)
# ==========================================
print("\n========== ENTROPY CODING ==========")
flat_data = flatten_stream(encoded_stream)
tree      = build_huffman_tree(flat_data)
codebook  = build_codes(tree)
bitstream = huffman_encode(flat_data, codebook)

# Génération du fichier binaire compressé final
save_to_bin(bitstream, tree, filename="video.bin")

# ==========================================
# 5. ÉVALUATION DES PERFORMANCES (Metrics)
# ==========================================
original_y_frames = [Y for (Y, Cb, Cr) in ycbcr_frames]

print_metrics(
    original_frames=original_y_frames,
    reconstructed_frames=reconstructed_frames,
    encoded_stream=encoded_stream,
    bin_path="video.bin"
)

# ==========================================
# 6. PIPELINE DE DÉCODAGE COMPLET
# ==========================================
print("\n========== DECODING ==========")
print("Lecture et ouverture du fichier binaire video.bin...")
bitstring_loaded, tree_loaded = load_from_bin("video.bin")

# Étape d'extraction de Huffman
flat_data_decoded = huffman_decode(bitstring_loaded, tree_loaded)
print("Décompression Entropy (Huffman) validée sans perte !")

print("Reconstruction spatio-temporelle des images en cours...")
decoded_frames = []
reference_frame_dec = None

# Boucle du décodeur : Reconstruit la vidéo à partir du flux encodé
for i, (frame_type, content) in enumerate(encoded_stream):
    if frame_type == "I":
        rle_blocks, original_shape = content
        reconstructed_i = decode_i_frame(rle_blocks, original_shape, fq=FQ)
        reference_frame_dec = reconstructed_i
        
        decoded_frames.append(reconstructed_i)
        print(f"Frame {i:03d} décodée avec succès [Type: I]")
        
    elif frame_type == "P":
        mv, residuals = content
        reconstructed_p = decode_p_frame(reference_frame_dec, mv, residuals, fq=FQ)
        reference_frame_dec = reconstructed_p
        
        decoded_frames.append(reconstructed_p)
        print(f"Frame {i:03d} décodée avec succès [Type: P]")

# Sauvegarde physique des images reconstruites sur ton ordinateur
os.makedirs("reconstructed_output", exist_ok=True)
for idx, frame_dec in enumerate(decoded_frames):
    img_to_save = np.clip(frame_dec, 0, 255).astype(np.uint8)
    cv2.imwrite(f"reconstructed_output/frame_{idx:04d}.png", img_to_save)

print("\n🎉 Succès ! Toutes les images reconstruites sont dans 'reconstructed_output/'.")

# ==========================================
# 7. VISUALISATION DES GRAPHIQUES PLT
# ==========================================
print("\n========== VISUALISATION ==========")
print("Génération du graphique récapitulatif Matplotlib...")
visualize_pipeline(
    original_frames=frames,
    ycbcr_frames=ycbcr_frames,
    encoded_stream=encoded_stream,
    reconstructed_frames=reconstructed_frames,
    motion_vectors=last_mv,
    residual_blocks=last_residuals,
    fq=FQ
)