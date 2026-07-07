import cv2
import os
import numpy as np

# Paths
ICONS_DIR = "./assets/icons"
PAGES_DIR = "./assets/dataset/pages_processed"
OUTPUT_DIR = "./outputs/shape_matches"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_icon_contours():
    icon_contours = {}
    for icon_file in os.listdir(ICONS_DIR):
        if not icon_file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
            continue

        icon_path = os.path.join(ICONS_DIR, icon_file)
        img = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Could not read {icon_file}, skipping")
            continue

        # Threshold → binary
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Take the largest contour
            c = max(contours, key=cv2.contourArea)
            icon_contours[icon_file] = c
            print(f"Loaded {icon_file} with contour size {cv2.contourArea(c)}")
        else:
            print(f"⚠️ No contour found in {icon_file}")
    return icon_contours

def match_crop(crop_img, crop_name, icon_contours):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return

    crop_contour = max(contours, key=cv2.contourArea)

    best_match = None
    best_score = 1e9  # lower = better

    for icon_name, icon_contour in icon_contours.items():
        score = cv2.matchShapes(icon_contour, crop_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        if score < best_score:
            best_score = score
            best_match = icon_name

    if best_match and best_score < 0.3:  # tune threshold
        print(f"[MATCH] {crop_name} -> {best_match} (score={best_score:.4f})")

        # Load icon in grayscale
        icon_img = cv2.imread(os.path.join(ICONS_DIR, best_match), cv2.IMREAD_GRAYSCALE)

        if icon_img is not None:
            # Resize icon to match crop height for concatenation
            icon_resized = cv2.resize(icon_img, (gray.shape[1], gray.shape[0]))

            # Concatenate side by side
            vis = cv2.hconcat([gray, icon_resized])
            out_path = os.path.join(OUTPUT_DIR, f"{crop_name}_to_{best_match}.png")
            cv2.imwrite(out_path, vis)

# Load icons as contours
icon_contours = load_icon_contours()
print(f"✅ Loaded {len(icon_contours)} icons for shape matching")

# Process all crops
for crop_file in os.listdir(PAGES_DIR):
    if not crop_file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
        continue

    crop_path = os.path.join(PAGES_DIR, crop_file)
    crop_img = cv2.imread(crop_path)

    match_crop(crop_img, os.path.splitext(crop_file)[0], icon_contours)

print("✅ Shape matching complete. Check 'outputs/shape_matches/' for results.")
