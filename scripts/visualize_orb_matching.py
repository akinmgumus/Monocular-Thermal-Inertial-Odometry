#!/usr/bin/env python3
"""
Visualize ORB feature matching between consecutive thermal frames.
Applies CLAHE preprocessing, detects ORB features, matches with BFMatcher,
and draws matches between frame pairs.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

IMG_DIR = "/home/akin/firestereo/dataset/stereo_thermal_depth/hawkins_4/hawkins_4_train/img_left"
OUTPUT_PATH = "/home/akin/tvio_ws/results/orb_matching.png"
NUM_PAIRS = 10


def preprocess(img_16bit):
    """16-bit thermal -> CLAHE 8-bit"""
    img_8bit = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_8bit)


def main():
    image_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    if len(image_files) < NUM_PAIRS + 1:
        print(f"Not enough images: {len(image_files)}")
        return

    # Sample evenly across the sequence (consecutive pairs)
    indices = np.linspace(0, len(image_files) - 2, NUM_PAIRS, dtype=int)

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    fig, axes = plt.subplots(NUM_PAIRS, 1, figsize=(16, 5 * NUM_PAIRS))

    for i, idx in enumerate(indices):
        img1_raw = cv2.imread(image_files[idx], cv2.IMREAD_UNCHANGED)
        img2_raw = cv2.imread(image_files[idx + 1], cv2.IMREAD_UNCHANGED)

        img1 = preprocess(img1_raw)
        img2 = preprocess(img2_raw)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            axes[i].text(0.5, 0.5, 'Not enough features', ha='center', va='center')
            axes[i].axis('off')
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep best 60%
        n_good = max(10, int(len(matches) * 0.6))
        good_matches = matches[:n_good]

        fname1 = os.path.basename(image_files[idx])
        fname2 = os.path.basename(image_files[idx + 1])

        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        axes[i].imshow(match_img)
        axes[i].set_title(
            f'{fname1} <-> {fname2}  |  '
            f'KP: {len(kp1)}/{len(kp2)}  |  '
            f'Matches: {len(matches)} -> Good: {n_good}',
            fontsize=11
        )
        axes[i].axis('off')

    plt.suptitle('ORB Feature Matching on CLAHE-Preprocessed Thermal Frames',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_PATH, dpi=100, bbox_inches='tight')
    print(f"Saved to {OUTPUT_PATH}")
    plt.close()


if __name__ == '__main__':
    main()