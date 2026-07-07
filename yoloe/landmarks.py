import cv2
import os
import numpy as np

# Paths
ICONS_DIR = "./assets/icons"
OUTPUT_DIR = "./outputs/icon_landmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Fourier Descriptor Functions ----------
def contour_to_fd(contour, n_points=64):
    """Convert a contour to Fourier Descriptor and return sampled points"""
    contour = contour.squeeze()
    contour = contour - contour.mean(axis=0)
    if len(contour) < n_points:
        n_points = len(contour)
    indices = np.linspace(0, len(contour)-1, n_points).astype(int)
    sampled = contour[indices]
    complex_contour = sampled[:,0] + 1j*sampled[:,1]
    fd = np.fft.fft(complex_contour)
    fd = fd / (np.abs(fd[1]) + 1e-8)  # normalize for scale invariance
    return fd, sampled

def fd_distance(fd1, fd2):
    return np.sum(np.abs(fd1 - fd2))

# ---------- Geometric Feature Functions ----------
def extract_geom_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter**2 + 1e-8)
    x, y, w, h = cv2.boundingRect(contour)
    extent = area / (w*h + 1e-8)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis, minor_axis = axes
    else:
        center = (0,0)
        major_axis = minor_axis = angle = 0
    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "extent": extent,
        "center": center,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "angle": angle
    }

# ---------- Load and preprocess icons ----------
icon_data = {}
orb = cv2.ORB_create(nfeatures=200)

for file in os.listdir(ICONS_DIR):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    path = os.path.join(ICONS_DIR, file)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # ---- Preprocessing: smooth + threshold + morph ----
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # ---- Contours ----
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    contour = max(contours, key=cv2.contourArea)

    # Fourier Descriptor
    fd, sampled_points = contour_to_fd(contour)

    # Geometric features
    geom = extract_geom_features(contour)

    # ORB keypoints
    kp, des = orb.detectAndCompute(img, None)

    icon_data[file] = {
        "contour": contour,
        "fd": fd,
        "sampled_points": sampled_points,
        "geom": geom,
        "img": img,
        "kp": kp,
        "des": des
    }
    print(f"Loaded {file}")

# ---------- Visualize Landmarks ----------
for file, data in icon_data.items():
    img_vis = cv2.cvtColor(data["img"], cv2.COLOR_GRAY2BGR)

    # Fourier Descriptor landmarks (red)
    for pt in data["sampled_points"]:
        pt = tuple(pt.astype(int))
        cv2.circle(img_vis, pt, 2, (0, 0, 255), -1)

    # Geometric ellipse (green)
    geom = data["geom"]
    if geom["major_axis"] > 0 and geom["minor_axis"] > 0:
        cv2.ellipse(
            img_vis,
            (tuple(map(int, geom["center"])), (int(geom["major_axis"]), int(geom["minor_axis"])), int(geom["angle"])),
            (0, 255, 0), 1
        )

    # ORB keypoints (blue)
    for kp in data["kp"]:
        pt = tuple(np.round(kp.pt).astype(int))
        cv2.circle(img_vis, pt, 2, (255, 0, 0), -1)

    out_path = os.path.join(OUTPUT_DIR, f"{file}_landmarks.png")
    cv2.imwrite(out_path, img_vis)

print(f"✅ Landmark visualization saved in {OUTPUT_DIR}")
