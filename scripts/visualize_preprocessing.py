#!/usr/bin/env python3
"""
Visualize 16-bit raw thermal vs CLAHE preprocessed images.
Saves a grid of 10 image pairs to results/.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

IMG_DIR = "/home/akin/firestereo/dataset/stereo_thermal_depth/hawkins_4/hawkins_4_train/img_left"
OUTPUT_PATH = "/home/akin/tvio_ws/results/preprocessing_comparison.png"
NUM_IMAGES = 10


def main():
    image_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    if len(image_files) < NUM_IMAGES:
        print(f"Only {len(image_files)} images found")
        return

    # Sample evenly across the sequence
    indices = np.linspace(0, len(image_files) - 1, NUM_IMAGES, dtype=int)
    selected = [image_files[i] for i in indices]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(18, 4 * NUM_IMAGES))

    for i, img_path in enumerate(selected):
        fname = os.path.basename(img_path)
        img_16bit = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Raw 16-bit direct cast (shows how dark/flat original is)
        img_raw_8bit = (img_16bit >> 8).astype(np.uint8)

        # Normalize to 8-bit
        img_norm = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # CLAHE on normalized
        img_clahe = clahe.apply(img_norm)

        # Column 1: Raw 16-bit (direct cast, no processing)
        axes[i, 0].imshow(img_raw_8bit, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title(f'{fname} — Raw 16-bit', fontsize=10)
        axes[i, 0].axis('off')

        # Column 2: Normalized 8-bit
        axes[i, 1].imshow(img_norm, cmap='gray')
        axes[i, 1].set_title(f'{fname} — Normalized 8-bit', fontsize=10)
        axes[i, 1].axis('off')

        # Column 3: CLAHE
        axes[i, 2].imshow(img_clahe, cmap='gray')
        axes[i, 2].set_title(f'{fname} — CLAHE (clipLimit=2.0, 8x8)', fontsize=10)
        axes[i, 2].axis('off')

    plt.suptitle('Thermal Image Preprocessing: Raw vs Normalized vs CLAHE', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_PATH, dpi=100, bbox_inches='tight')
    print(f"Saved to {OUTPUT_PATH}")
    plt.close()


if __name__ == '__main__':
    main()
