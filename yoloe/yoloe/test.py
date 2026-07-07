import os
from pathlib import Path
from pdf2image import convert_from_path
import cv2
import numpy as np
import json
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# =========================================
# CONFIG
# =========================================
PDF_FOLDER = "assets/pages"
PAGE_TIFF_FOLDER = "assets/pages_tiff"  # converted high-res pages
ICON_FOLDER = "assets/icons"
OUTPUT_FOLDER = "outputs"
DPI = 1000  # high resolution for better detection
CROP_SIZE = 256
ZOOM_FACTORS = [1.0, 1.5, 2.0]

# Detection thresholds
DETECTION_THRESHOLD = 0.7
SECONDARY_THRESHOLD = 0.7

# =========================================
# 1️⃣ Convert PDFs to high-res TIFFs
# =========================================
os.makedirs(PAGE_TIFF_FOLDER, exist_ok=True)

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
Image.MAX_IMAGE_PIXELS = None

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    print(f"Converting {pdf_file} to high-res TIFFs...")
    pages = convert_from_path(pdf_path, dpi=DPI)
    for i, page in enumerate(pages):
        page_tiff_path = os.path.join(PAGE_TIFF_FOLDER, f"{Path(pdf_file).stem}_page_{i+1}.tiff")
        page.save(page_tiff_path, "TIFF")

print("✅ PDF conversion complete.")

# =========================================
# 2️⃣ Preprocess pages (keep red only)
# =========================================
def preprocess_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_only = cv2.bitwise_and(img, img, mask=red_mask)
    white_bg = np.full_like(img, 255)
    processed = np.where(red_only > 0, red_only, white_bg)
    return cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

# Optional: save preprocessed pages
PROCESSED_FOLDER = "assets/pages_processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

tiff_files = [f for f in os.listdir(PAGE_TIFF_FOLDER) if f.lower().endswith((".tiff", ".tif"))]
for tiff_file in tiff_files:
    tiff_path = os.path.join(PAGE_TIFF_FOLDER, tiff_file)
    img = cv2.imread(tiff_path)
    processed = preprocess_image(img)
    save_path = os.path.join(PROCESSED_FOLDER, tiff_file.replace(".tiff", ".png"))
    cv2.imwrite(save_path, processed)

print("✅ Preprocessing complete.")

# =========================================
# 3️⃣ ICON DETECTOR
# =========================================
class IconDetector:
    def __init__(self, assets_folder, threshold=0.7, secondary_threshold=0.7, visual_output=True):
        self.assets_folder = Path(assets_folder)
        self.threshold = threshold
        self.secondary_threshold = secondary_threshold
        self.visual_output = visual_output
        self.templates = {}
        self.rotation_angles = list(range(0, 360, 15))
        self.scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
        self.load_templates()

    def load_templates(self):
        print("Loading icon templates...")
        for ext in ["*.tiff", "*.tif", "*.png"]:
            for tpl_path in self.assets_folder.glob(ext):
                img = cv2.imread(str(tpl_path))
                if img is None:
                    continue
                name = tpl_path.stem
                self.templates[name] = self.create_variants(preprocess_image(img))
                print(f"  {name}: {len(self.templates[name])} variants")

    def create_variants(self, template):
        variants = []
        h, w = template.shape
        for scale in self.scales:
            new_w, new_h = max(int(w*scale), 3), max(int(h*scale), 3)
            scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_CUBIC) if scale != 1.0 else template.copy()
            for angle in self.rotation_angles:
                rotated = self.rotate(scaled, angle)
                variants.append(rotated)
        return variants

    def rotate(self, template, angle):
        if angle == 0:
            return template
        h, w = template.shape
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_val, sin_val = abs(M[0,0]), abs(M[0,1])
        new_w, new_h = int(h*sin_val + w*cos_val), int(h*cos_val + w*sin_val)
        M[0,2] += (new_w/2)-center[0]
        M[1,2] += (new_h/2)-center[1]
        return cv2.warpAffine(template, M, (new_w, new_h), borderValue=255)

    def detect_icons(self, page_path, output_folder="outputs"):
        img_color = cv2.imread(page_path)
        img_gray = preprocess_image(img_color)
        detections = []

        for name, variants in self.templates.items():
            for tpl in variants:
                result1 = cv2.matchTemplate(img_gray, tpl, cv2.TM_CCOEFF_NORMED)
                result2 = cv2.matchTemplate(img_gray, tpl, cv2.TM_CCORR_NORMED)
                yloc, xloc = np.where(result1 >= self.threshold)
                for (x, y) in zip(xloc, yloc):
                    if result2[y, x] <= self.secondary_threshold:
                        continue
                    h, w = tpl.shape
                    region = img_gray[y:y+h, x:x+w]
                    if region.shape[0]<=5 or region.shape[1]<=5:
                        continue
                    # SSIM check
                    try:
                        region_resized = cv2.resize(region, (w, h))
                        if ssim(region_resized, tpl) < 0.8:
                            continue
                    except:
                        continue
                    detections.append({'x': x, 'y': y, 'width': w, 'height': h, 'name': name})

        # Save JSON
        os.makedirs(output_folder, exist_ok=True)
        json_path = os.path.join(output_folder, f"{Path(page_path).stem}.json")
        counts = {}
        for d in detections:
            counts[d['name']] = counts.get(d['name'], 0)+1
        with open(json_path, 'w') as f:
            json.dump(counts, f, indent=2)
        print(f"  {Path(page_path).stem}: {len(detections)} detections, saved to {json_path}")
        return detections

# =========================================
# 4️⃣ Run detection on all preprocessed pages
# =========================================
detector = IconDetector(ICON_FOLDER, threshold=DETECTION_THRESHOLD, secondary_threshold=SECONDARY_THRESHOLD)

preprocessed_pages = [os.path.join(PROCESSED_FOLDER, f) for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".png")]
for page in preprocessed_pages:
    detector.detect_icons(page, OUTPUT_FOLDER)

print("✅ All pages processed, detection complete.")
