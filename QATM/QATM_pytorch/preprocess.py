import cv2
import os
import numpy as np
from pathlib import Path

# Paths
ROOT_DIR = "./sample"
PAGES_DIR = "/Users/saniya.kolangde/Desktop/BluePrint/QATM/QATM_pytorch/sample"
OUTPUT_DIR = os.path.join(ROOT_DIR, "qatm_templates")  # use this for QATM

# Parameters
CROP_SIZE = 256   # size of cropped patches
ZOOM_FACTORS = [1.0, 1.5, 2.0]  # generate zoomed versions of each crop
WHITE_THRESH = 0.95  # skip crops that are >95% white

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_not_empty(crop, white_thresh=WHITE_THRESH):
    """Return True if crop is not too white/empty."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean() / 255.0  # normalize 0–1
    return mean_val < white_thresh


def crop_and_zoom(img, page_name, output_dir=OUTPUT_DIR):
    """Crop into patches and create zoomed versions for QATM templates"""
    h, w = img.shape[:2]
    count = 0

    for y in range(0, h, CROP_SIZE):
        for x in range(0, w, CROP_SIZE):
            crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
            if crop.shape[0] < CROP_SIZE or crop.shape[1] < CROP_SIZE:
                continue  # skip incomplete edges

            # Skip if crop is too white
            if not is_not_empty(crop):
                continue

            # Save original crop
            crop_filename = f"{page_name}_crop_{count}.png"
            cv2.imwrite(os.path.join(output_dir, crop_filename), crop)

            # Save zoomed versions
            for z in ZOOM_FACTORS:
                if z == 1.0:
                    continue
                zoomed = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
                zoom_filename = f"{page_name}_crop_{count}_zoom{z}.png"
                cv2.imwrite(os.path.join(output_dir, zoom_filename), zoomed)

            count += 1


def generate_qatm_templates():
    """Run preprocessing on all page images and create template folder for QATM"""
    for page_file in os.listdir(PAGES_DIR):
        if not page_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        page_path = os.path.join(PAGES_DIR, page_file)
        page_name = Path(page_file).stem

        print(f"Processing {page_file}...")
        img = cv2.imread(page_path)

        if img is None:
            print(f"⚠️ Skipping {page_file}, could not read.")
            continue

        crop_and_zoom(img, page_name)

    print(f"✅ Templates ready in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_qatm_templates()
