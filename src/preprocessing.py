import cv2
import numpy as np
import os

def load_frames(folder):
    frames = []
    # On prend tous les fichiers et on les trie
    files = sorted(os.listdir(folder))

    for f in files:
        path = os.path.join(folder, f)
        frame = cv2.imread(path)
        if frame is not None:
            frames.append(frame)

    return frames


def preprocess_frames(frames):
    preprocessed = []
    
    for f in frames:
        # 1. On convertit l'image en YCbCr
        ycbcr = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        
        # 2. On sépare les trois canaux
        Y  = ycbcr[:, :, 0]
        Cr = ycbcr[:, :, 1]
        Cb = ycbcr[:, :, 2]
        
        # 3. On extrait proprement la hauteur (H) et la largeur (W) de l'image f
        hauteur = f.shape[0]
        largeur = f.shape[1]
        
        # 4. On divise par 2 pour le format 4:2:0
        new_h = hauteur // 2
        new_w = largeur // 2
        
        # 5. On redimensionne la couleur (Chroma Subsampling)
        Cb_sub = cv2.resize(Cb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        Cr_sub = cv2.resize(Cr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # On stocke le résultat sous forme de tuple
        preprocessed.append((Y, Cb_sub, Cr_sub))
        
    return preprocessed