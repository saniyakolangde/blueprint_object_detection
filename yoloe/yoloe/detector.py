# """
# YOLOE Icon Detector - Optimized template matching with visual output.

# Supports TIFF input files (.tiff and .tif extensions) for both icons and pages.
# Visual output is saved as PNG files.
# """

# import cv2
# import numpy as np
# from typing import List, Dict
# from pathlib import Path
# import json


# # =====================================================================================
# # DETECTION HYPERPARAMETERS - Experiment with these to improve accuracy
# # =====================================================================================

# # Primary Detection Settings
# DEFAULT_DETECTION_THRESHOLD = 0.6      # Main detection threshold (0.0-1.0, lower = more sensitive)
# SECONDARY_CONFIRMATION_THRESHOLD = 0.8  # Secondary method confirmation (higher = stricter)

# # Multi-Scale Detection
# ROTATION_ANGLES = [
#     0, 90, 180, 270,
#     45, 135, 225, 315,
#     60, 150, 240, 330,
#     30, 120, 210, 300,
#     10, 100, 190, 280,
#     350, 80, 170, 260
# ] # Degrees to test
# SCALE_FACTORS = [0.9, 1.0, 1.1, 0.95, 1.05, 0.8, 1.2]                              # Size variations to test
# MIN_TEMPLATE_SIZE = 5                                                     # Minimum template dimension after scaling

# # Template Preprocessing (affects template quality)
# TEMPLATE_CLAHE_CLIP_LIMIT = 3.0      # Contrast enhancement for templates (higher = more contrast)
# TEMPLATE_CLAHE_TILE_SIZE = (4, 4)    # CLAHE tile size for templates
# TEMPLATE_SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Edge sharpening

# # Page Preprocessing (affects detection sensitivity)
# PAGE_CLAHE_CLIP_LIMIT = 5.0          # Contrast enhancement for pages (higher = more contrast)  
# PAGE_CLAHE_TILE_SIZE = (8, 8)        # CLAHE tile size for pages
# BILATERAL_FILTER_D = 9               # Bilateral filter diameter (noise reduction)
# BILATERAL_FILTER_SIGMA_COLOR = 75    # Bilateral filter color sigma
# BILATERAL_FILTER_SIGMA_SPACE = 75    # Bilateral filter space sigma
# ADAPTIVE_THRESH_BLOCK_SIZE = 11      # Adaptive threshold block size (odd number)
# ADAPTIVE_THRESH_C = 2                # Adaptive threshold constant

# # Non-Maximum Suppression (affects duplicate removal)
# NMS_OVERLAP_THRESHOLD_SMALL = 0.2    # IoU threshold for small templates (< 30px) - more aggressive
# NMS_OVERLAP_THRESHOLD_LARGE = 0.25   # IoU threshold for large templates (>= 30px) - more aggressive  
# NMS_SIZE_THRESHOLD = 30              # Pixel size threshold for NMS categorization
# NMS_DISTANCE_THRESHOLD = 10          # Minimum distance between detections (pixels)

# # False Positive Filtering (reduces wrong detections)
# WHITE_TEMPLATE_MAX_DETECTIONS = 100   # Max detections for mostly-white templates
# WHITE_TEMPLATE_FILTER_RATIO = 0.5     # Keep only top X ratio of detections for white templates
# FAN_MAX_DETECTIONS = 50               # Max fan detections per page

# # Color Matching Settings (reduces false positives)
# ENABLE_COLOR_MATCHING = True         # Enable color consistency checking (disabled for testing)
# COLOR_TOLERANCE = 50                  # RGB tolerance for color matching (0-255) - increased tolerance
# MIN_COLOR_MATCH_RATIO = 0.6          # Minimum ratio of pixels that must match template color - lowered

# # Visual Output Settings
# VISUALIZATION_COLORS = [             # Colors for different icon types (BGR format)
#     # (255, 0, 0),    # Blue
#     # (0, 255, 0),    # Green  
#     (0, 0, 255),    # Red
#     # (255, 255, 0),  # Cyan
#     # (255, 0, 255),  # Magenta
#     # (0, 255, 255),  # Yellow
#     # (128, 0, 128),  # Purple
#     # (255, 165, 0),  # Orange
# ]
# BBOX_THICKNESS = 2                   # Bounding box line thickness
# LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
# LABEL_FONT_SCALE = 0.5
# LABEL_FONT_THICKNESS = 1
# LABEL_TEXT_COLOR = (255, 255, 255)   # White text
# SUMMARY_FONT_SCALE = 1
# SUMMARY_FONT_THICKNESS = 2
# SUMMARY_POSITION = (10, 30)

# # Performance Settings
# ENABLE_DUAL_METHOD_CONFIRMATION = True  # Use both TM_CCOEFF_NORMED and TM_CCORR_NORMED
# ENABLE_TEMPLATE_SHARPENING = True       # Apply sharpening to templates
# ENABLE_PAGE_DENOISING = True            # Apply bilateral filtering to pages

# # =====================================================================================


# class IconDetector:
#     """Optimized icon detector with visual output capabilities."""
    
#     def __init__(self, assets_folder: str, threshold: float = None, visual_output: bool = False):
#         """
#         Initialize the icon detector.
        
#         Args:
#             assets_folder: Path to folder containing icon templates
#             threshold: Detection threshold (0.0-1.0, lower = more sensitive). Uses DEFAULT_DETECTION_THRESHOLD if None.
#             visual_output: Whether to generate visual detection images
#         """
#         self.assets_folder = Path(assets_folder)
#         self.threshold = threshold if threshold is not None else DEFAULT_DETECTION_THRESHOLD
#         self.visual_output = visual_output
#         self.templates = {}
        
#         # Load detection parameters from constants
#         self.rotation_angles = ROTATION_ANGLES
#         self.scales = SCALE_FACTORS
#         self.colors = VISUALIZATION_COLORS
        
#         self.load_templates()
    
#     def load_templates(self):
#         """Load and preprocess all template images."""
#         print("Loading templates...")
        
#         # Load templates (TIFF, TIF, and PNG extensions)
#         template_patterns = ["*.tiff", "*.tif", "*.png"]
#         template_paths = []
#         for pattern in template_patterns:
#             template_paths.extend(self.assets_folder.glob(pattern))
        
#         for i, template_path in enumerate(template_paths):
#             if template_path.name.startswith('.'):
#                 continue
                
#             template_name = template_path.stem
#             template_img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            
#             if template_img is None:
#                 continue
            
#             # Preprocess template
#             processed_template = self._preprocess_template(template_img)
            
#             # Create multi-scale and rotated variants
#             all_variants = self._create_template_variants(processed_template)
            
#             # Extract dominant color from original template for color matching
#             dominant_color = self._extract_dominant_color(template_img) if ENABLE_COLOR_MATCHING else None
            
#             self.templates[template_name] = {
#                 'variants': all_variants,
#                 'color': self.colors[i % len(self.colors)],
#                 'original_size': template_img.shape,
#                 'dominant_color': dominant_color,
#                 'original_template': template_img  # Keep for color matching
#             }
            
#             print(f"  {template_name}: {len(all_variants)} variants")
    
#     def _preprocess_template(self, template: np.ndarray) -> np.ndarray:
#         """Enhance template for better matching."""
#         # Apply contrast enhancement
#         clahe = cv2.createCLAHE(clipLimit=TEMPLATE_CLAHE_CLIP_LIMIT, tileGridSize=TEMPLATE_CLAHE_TILE_SIZE)
#         enhanced = clahe.apply(template)
        
#         # Sharpen edges if enabled
#         if ENABLE_TEMPLATE_SHARPENING:
#             sharpened = cv2.filter2D(enhanced, -1, TEMPLATE_SHARPEN_KERNEL)
#             return sharpened
        
#         return enhanced
    
#     def _preprocess_page(self, page_img: np.ndarray) -> np.ndarray:
#         """Enhance page image for better detection."""
#         # Apply contrast enhancement
#         clahe = cv2.createCLAHE(clipLimit=PAGE_CLAHE_CLIP_LIMIT, tileGridSize=PAGE_CLAHE_TILE_SIZE)
#         enhanced = clahe.apply(page_img)
        
#         # Reduce noise if enabled
#         if ENABLE_PAGE_DENOISING:
#             denoised = cv2.bilateralFilter(enhanced, BILATERAL_FILTER_D, 
#                                          BILATERAL_FILTER_SIGMA_COLOR, BILATERAL_FILTER_SIGMA_SPACE)
#         else:
#             denoised = enhanced
        
#         # Apply adaptive threshold
#         processed = cv2.adaptiveThreshold(
#             denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
#             ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
#         )
        
#         return processed
    
#     def _create_template_variants(self, template: np.ndarray) -> List[Dict]:
#         """Create rotated and scaled variants of template."""
#         variants = []
        
#         for scale in self.scales:
#             # Scale template
#             if scale != 1.0:
#                 h, w = template.shape
#                 new_w, new_h = int(w * scale), int(h * scale)
#                 if new_w < MIN_TEMPLATE_SIZE or new_h < MIN_TEMPLATE_SIZE:
#                     continue
#                 scaled_template = cv2.resize(template, (new_w, new_h))
#             else:
#                 scaled_template = template.copy()
            
#             # Create rotated versions
#             for angle in self.rotation_angles:
#                 rotated_template = self._rotate_template(scaled_template, angle)
#                 if rotated_template.size > 0:
#                     variants.append({
#                         'image': rotated_template,
#                         'scale': scale,
#                         'angle': angle
#                     })
        
#         return variants
    
#     def _rotate_template(self, template: np.ndarray, angle: float) -> np.ndarray:
#         """Rotate template by given angle."""
#         if angle == 0:
#             return template
        
#         h, w = template.shape
#         center = (w // 2, h // 2)
        
#         # Create rotation matrix
#         rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
#         # Calculate new dimensions
#         cos_val = abs(rotation_matrix[0, 0])
#         sin_val = abs(rotation_matrix[0, 1])
#         new_w = int((h * sin_val) + (w * cos_val))
#         new_h = int((h * cos_val) + (w * sin_val))
        
#         # Adjust rotation matrix
#         rotation_matrix[0, 2] += (new_w / 2) - center[0]
#         rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
#         # Apply rotation
#         rotated = cv2.warpAffine(
#             template, rotation_matrix, (new_w, new_h),
#             borderMode=cv2.BORDER_CONSTANT, borderValue=255
#         )
        
#         return rotated
    
#     def _extract_dominant_color(self, template: np.ndarray) -> np.ndarray:
#         """Extract dominant color from template for color matching."""
#         if len(template.shape) == 3:
#             # Color template - use mean color
#             return np.mean(template.reshape(-1, template.shape[2]), axis=0)
#         else:
#             # Grayscale template - convert to RGB
#             mean_gray = np.mean(template)
#             return np.array([mean_gray, mean_gray, mean_gray])
    
#     def _check_color_match(self, page_region: np.ndarray, template_color: np.ndarray, 
#                           original_page: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
#         """Check if the detected region matches the expected template color."""
#         if not ENABLE_COLOR_MATCHING or template_color is None:
#             return True
        
#         # Extract color region from original page
#         if len(original_page.shape) == 3:
#             color_region = original_page[y:y+h, x:x+w]
#         else:
#             # Convert grayscale to RGB
#             gray_region = original_page[y:y+h, x:x+w]
#             color_region = cv2.cvtColor(gray_region, cv2.COLOR_GRAY2RGB)
        
#         if color_region.size == 0:
#             return False
        
#         # Calculate mean color of the region
#         region_color = np.mean(color_region.reshape(-1, color_region.shape[2]), axis=0)
        
#         # Check if colors are within tolerance
#         color_diff = np.abs(region_color - template_color)
#         color_matches = np.all(color_diff <= COLOR_TOLERANCE)
        
#         return color_matches
    
#     def detect_icons(self, page_image_path: str, output_folder: str = "outputs") -> Dict[str, int]:
#         """
#         Detect icons in a page image.
        
#         Args:
#             page_image_path: Path to the page image
#             output_folder: Folder to save results
            
#         Returns:
#             Dictionary with icon counts
#         """
#         print(f"Processing {Path(page_image_path).name}...")
        
#         # Load original page for color matching
#         original_page = cv2.imread(page_image_path)
#         if original_page is None:
#             raise ValueError(f"Could not load image: {page_image_path}")
        
#         # Load and preprocess page for detection
#         page_img = cv2.imread(page_image_path, cv2.IMREAD_GRAYSCALE)
#         processed_page = self._preprocess_page(page_img)
        
#         # Detect each icon type
#         results = {}
#         all_detections = []
        
#         for template_name, template_data in self.templates.items():
#             detections = self._detect_template(
#                 processed_page, template_data['variants'], template_name,
#                 original_page, template_data.get('dominant_color')
#             )
            
#             # Apply confidence filtering
#             filtered_detections = self._filter_detections(detections, template_name)
            
#             results[template_name] = len(filtered_detections)
            
#             # Store for visualization
#             for detection in filtered_detections:
#                 detection['name'] = template_name
#                 detection['color'] = template_data['color']
#                 all_detections.append(detection)
        
#         # Save results
#         output_path = Path(output_folder)
#         output_path.mkdir(exist_ok=True)
        
#         page_name = Path(page_image_path).stem
#         json_file = output_path / f"{page_name}.json"
        
#         with open(json_file, 'w') as f:
#             json.dump(results, f, indent=2)
        
#         # Create visual output if requested
#         if self.visual_output:
#             self._create_visual_output(
#                 page_image_path, all_detections, output_path / f"{page_name}_visual.png"
#             )
        
#         return results
    
#     def _detect_template(self, page_img: np.ndarray, variants: List[Dict], 
#                         template_name: str, original_page: np.ndarray = None, 
#                         template_color: np.ndarray = None) -> List[Dict]:
#         """Detect all instances of a template in the page."""
#         all_matches = []
        
#         for variant in variants:
#             template = variant['image']
            
#             # Primary matching method
#             result1 = cv2.matchTemplate(page_img, template, cv2.TM_CCOEFF_NORMED)
            
#             # Secondary confirmation method (if enabled)
#             if ENABLE_DUAL_METHOD_CONFIRMATION:
#                 result2 = cv2.matchTemplate(page_img, template, cv2.TM_CCORR_NORMED)
            
#             # Find matches above threshold
#             locations = np.where(result1 >= self.threshold)
            
#             for y, x in zip(locations[0], locations[1]):
#                 score1 = result1[y, x]
                
#                 # Apply secondary confirmation if enabled
#                 if ENABLE_DUAL_METHOD_CONFIRMATION:
#                     score2 = result2[y, x]
#                     if score2 <= SECONDARY_CONFIRMATION_THRESHOLD:
#                         continue
                
#                 # Check color matching if enabled
#                 h, w = template.shape
#                 if original_page is not None and not self._check_color_match(
#                     None, template_color, original_page, int(x), int(y), int(w), int(h)
#                 ):
#                     continue
                
#                 # Add valid match
#                 all_matches.append({
#                     'x': int(x),
#                     'y': int(y),
#                     'width': int(w),
#                     'height': int(h),
#                     'score': float(score1),
#                     'scale': variant['scale'],
#                     'angle': variant['angle']
#                 })
        
#         # Apply non-maximum suppression
#         return self._non_maximum_suppression(all_matches)
    
#     def _non_maximum_suppression(self, detections: List[Dict]) -> List[Dict]:
#         """Remove overlapping detections."""
#         if not detections:
#             return []
        
#         # Sort by score (descending)
#         detections.sort(key=lambda x: x['score'], reverse=True)
        
#         filtered = []
#         while detections:
#             current = detections.pop(0)
#             filtered.append(current)
            
#             # Remove overlapping detections
#             remaining = []
#             for detection in detections:
#                 overlap = self._calculate_overlap(current, detection)
#                 distance = self._calculate_distance(current, detection)
                
#                 # Adaptive threshold based on size
#                 min_dim = min(current['width'], current['height'])
#                 overlap_threshold = NMS_OVERLAP_THRESHOLD_SMALL if min_dim < NMS_SIZE_THRESHOLD else NMS_OVERLAP_THRESHOLD_LARGE
                
#                 # Keep detection if both overlap and distance criteria are met
#                 if overlap < overlap_threshold and distance > NMS_DISTANCE_THRESHOLD:
#                     remaining.append(detection)
            
#             detections = remaining
        
#         return filtered
    
#     def _calculate_overlap(self, det1: Dict, det2: Dict) -> float:
#         """Calculate overlap ratio between two detections."""
#         x1_1, y1_1 = det1['x'], det1['y']
#         x2_1, y2_1 = x1_1 + det1['width'], y1_1 + det1['height']
        
#         x1_2, y1_2 = det2['x'], det2['y']
#         x2_2, y2_2 = x1_2 + det2['width'], y1_2 + det2['height']
        
#         # Calculate intersection
#         x1_i = max(x1_1, x1_2)
#         y1_i = max(y1_1, y1_2)
#         x2_i = min(x2_1, x2_2)
#         y2_i = min(y2_1, y2_2)
        
#         if x2_i <= x1_i or y2_i <= y1_i:
#             return 0.0
        
#         intersection = (x2_i - x1_i) * (y2_i - y1_i)
#         area1 = det1['width'] * det1['height']
#         area2 = det2['width'] * det2['height']
#         union = area1 + area2 - intersection
        
#         return intersection / union if union > 0 else 0.0
    
#     def _calculate_distance(self, det1: Dict, det2: Dict) -> float:
#         """Calculate center-to-center distance between two detections."""
#         center1_x = det1['x'] + det1['width'] // 2
#         center1_y = det1['y'] + det1['height'] // 2
#         center2_x = det2['x'] + det2['width'] // 2
#         center2_y = det2['y'] + det2['height'] // 2
        
#         return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
    
#     def _filter_detections(self, detections: List[Dict], template_name: str) -> List[Dict]:
#         """Apply heuristic filtering to reduce false positives."""
#         if not detections:
#             return []
        
#         # Apply dynamic filtering based on template characteristics
#         # Check if template name suggests a socket (usually white/simple)
#         is_socket = any(keyword in template_name.lower() for keyword in ['socket', 'amp', 'usb'])
#         is_fan = 'fan' in template_name.lower()
        
#         if is_socket:
#             # Socket templates are often simple/white - apply stricter filtering
#             if len(detections) > WHITE_TEMPLATE_MAX_DETECTIONS:
#                 # Keep only highest confidence detections
#                 detections.sort(key=lambda x: x['score'], reverse=True)
#                 detections = detections[:int(len(detections) * WHITE_TEMPLATE_FILTER_RATIO)]
        
#         elif is_fan:
#             # Fans should have reasonable counts
#             if len(detections) > FAN_MAX_DETECTIONS:
#                 detections.sort(key=lambda x: x['score'], reverse=True)
#                 detections = detections[:FAN_MAX_DETECTIONS]
        
#         # General filtering - remove detections with very low confidence
#         min_confidence = 0.7 if is_socket else 0.6
#         detections = [d for d in detections if d['score'] >= min_confidence]
        
#         return detections
    
#     def _create_visual_output(self, page_image_path: str, detections: List[Dict], 
#                              output_path: Path):
#         """Create visual output with bounding boxes."""
#         # Load original color image
#         img = cv2.imread(page_image_path)
#         if img is None:
#             # Convert grayscale to color if needed
#             gray = cv2.imread(page_image_path, cv2.IMREAD_GRAYSCALE)
#             img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
#         # Draw detections
#         for detection in detections:
#             x, y = detection['x'], detection['y']
#             w, h = detection['width'], detection['height']
#             color = detection['color']
#             name = detection['name']
#             score = detection['score']
            
#             # Draw bounding box
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, BBOX_THICKNESS)
            
#             # Draw label with confidence
#             label = f"{name}: {score:.2f}"
#             label_size = cv2.getTextSize(label, LABEL_FONT, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)[0]
            
#             # Draw label background
#             cv2.rectangle(img, (x, y - label_size[1] - 10), 
#                          (x + label_size[0], y), color, -1)
            
#             # Draw label text
#             cv2.putText(img, label, (x, y - 5), LABEL_FONT, 
#                        LABEL_FONT_SCALE, LABEL_TEXT_COLOR, LABEL_FONT_THICKNESS)
        
#         # Add summary text
#         summary_text = f"Total detections: {len(detections)}"
#         cv2.putText(img, summary_text, SUMMARY_POSITION, LABEL_FONT, 
#                    SUMMARY_FONT_SCALE, LABEL_TEXT_COLOR, SUMMARY_FONT_THICKNESS)
        
#         # Save visual output
#         cv2.imwrite(str(output_path), img)
#         print(f"  Visual output saved: {output_path}")
    
#     def get_detection_summary(self, results: Dict[str, int]) -> str:
#         """Get a formatted summary of detection results."""
#         total = sum(results.values())
#         summary = [f"Total detections: {total}"]
        
#         for icon_name, count in results.items():
#             if count > 0:
#                 summary.append(f"  {icon_name}: {count}")
        
#         return "\n".join(summary)


# def detect_all_pages(pages_folder: str = "pages", assets_folder: str = "assets", 
#                     output_folder: str = "outputs", visual_output: bool = True,
#                     threshold: float = 0.6) -> Dict[str, Dict[str, int]]:
#     """
#     Detect icons in all pages.
    
#     Args:
#         pages_folder: Folder containing page images
#         assets_folder: Folder containing icon templates  
#         output_folder: Folder to save results
#         visual_output: Whether to create visual outputs
#         threshold: Detection threshold
        
#     Returns:
#         Dictionary of results for all pages
#     """
#     detector = IconDetector(assets_folder, threshold=threshold, visual_output=visual_output)
    
#     pages_path = Path(pages_folder)
    
#     # Get page files (TIFF, TIF, and PNG extensions)
#     page_patterns = ["*.tiff", "*.tif", "*.png"]
#     page_files = []
#     for pattern in page_patterns:
#         page_files.extend(pages_path.glob(pattern))
    
#     if not page_files:
#         raise ValueError(f"No image files found in {pages_folder}")
    
#     print(f"Processing {len(page_files)} pages...")
    
#     all_results = {}
#     total_detections = 0
    
#     for page_file in page_files:
#         results = detector.detect_icons(str(page_file), output_folder)
#         all_results[page_file.name] = results
        
#         page_total = sum(results.values())
#         total_detections += page_total
        
#         print(f"  {page_file.name}: {page_total} detections")
    
#     print(f"\nSummary: {total_detections} total detections across {len(page_files)} pages")
    
#     # Save combined results
#     output_path = Path(output_folder)
#     combined_file = output_path / "all_results.json"
    
#     with open(combined_file, 'w') as f:
#         json.dump(all_results, f, indent=2)
    
#     print(f"Combined results saved: {combined_file}")
    
#     return all_results


import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path
import json
from skimage.metrics import structural_similarity as ssim
# =====================================================================================
# DETECTION HYPERPARAMETERS
# =====================================================================================
DEFAULT_DETECTION_THRESHOLD = 0.65
SECONDARY_CONFIRMATION_THRESHOLD = 0.7

ROTATION_ANGLES = list(range(0, 360, 15))
SCALE_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
MIN_TEMPLATE_SIZE = 5

NMS_OVERLAP_THRESHOLD = 0.8
NMS_DISTANCE_THRESHOLD = 10

BBOX_THICKNESS = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 1
LABEL_TEXT_COLOR = (255, 255, 255)
SUMMARY_FONT_SCALE = 1
SUMMARY_FONT_THICKNESS = 2
SUMMARY_POSITION = (10, 30)

# =====================================================================================
# PREPROCESSING HELPERS
# =====================================================================================

def preprocess_template(img: np.ndarray) -> np.ndarray:
    """Keep only red parts of the template on white background."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red mask
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_only = cv2.bitwise_and(img, img, mask=red_mask)
    white_background = np.full_like(img, 255)
    processed = np.where(red_only > 0, red_only, white_background)
    # Convert to grayscale for template matching
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    return processed_gray

def preprocess_page(img: np.ndarray) -> np.ndarray:
    """Keep red only on white background for detection."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_only = cv2.bitwise_and(img, img, mask=red_mask)
    white_background = np.full_like(img, 255)
    processed = np.where(red_only > 0, red_only, white_background)
    return cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

def is_circle_like(region, circularity_threshold=0.85):
    """Reject plain circles by checking contour circularity"""
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > circularity_threshold:  # very round
            return True
    return False

def edge_density(region):
    """Compute edge density (more detail = higher value)"""
    edges = cv2.Canny(region, 50, 150)
    return np.sum(edges > 0) / float(region.shape[0] * region.shape[1])

# =====================================================================================
# ICON DETECTOR CLASS
# =====================================================================================

class IconDetector:
    def __init__(self, assets_folder: str, threshold: float = None, visual_output: bool = False):
        self.assets_folder = Path(assets_folder)
        self.threshold = threshold if threshold is not None else DEFAULT_DETECTION_THRESHOLD
        self.visual_output = visual_output
        self.templates = {}
        self.rotation_angles = ROTATION_ANGLES
        self.scales = SCALE_FACTORS
        self.load_templates()

    def load_templates(self):
        print("Loading templates...")
        template_paths = []
        for ext in ["*.tiff", "*.tif", "*.png"]:
            template_paths.extend(self.assets_folder.glob(ext))

        for template_path in template_paths:
            if template_path.name.startswith('.'):
                continue
            name = template_path.stem
            img = cv2.imread(str(template_path))
            if img is None:
                continue
            processed = preprocess_template(img)
            variants = self._create_variants(processed)
            self.templates[name] = variants
            print(f"  {name}: {len(variants)} variants")

    def _create_variants(self, template: np.ndarray) -> List[np.ndarray]:
        variants = []
        for scale in self.scales:
            h, w = template.shape
            new_w, new_h = max(int(w * scale), 3), max(int(h * scale), 3)  # avoid zero size
            scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_CUBIC) if scale != 1.0 else template.copy()
            for angle in self.rotation_angles:
                rotated = self._rotate(scaled, angle)
                if rotated.size > 0:
                    variants.append(rotated)
        return variants

    def _rotate(self, template: np.ndarray, angle: float) -> np.ndarray:
        if angle == 0:
            return template
        h, w = template.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_val, sin_val = abs(M[0, 0]), abs(M[0, 1])
        new_w, new_h = int(h * sin_val + w * cos_val), int(h * cos_val + w * sin_val)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(template, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    def detect_icons(self, page_image_path: str, output_folder: str = "outputs") -> Dict[str, int]:
        print(f"Processing {Path(page_image_path).name}...")
        original_page = cv2.imread(page_image_path)
        if original_page is None:
            raise ValueError(f"Could not load image: {page_image_path}")
        processed_page = preprocess_page(original_page)

        # Collect all matches across all classes first
        all_detections = []
        for name, variants in self.templates.items():
            matches = self._detect_template(processed_page, variants, name)
            all_detections.extend(matches)

        # Apply cross-class NMS
        all_detections = self._cross_class_nms(all_detections)

        # Count results per class
        results = {}
        for det in all_detections:
            results[det['name']] = results.get(det['name'], 0) + 1

        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        page_name = Path(page_image_path).stem
        with open(output_path / f"{page_name}.json", 'w') as f:
            json.dump(results, f, indent=2)

        if self.visual_output:
            self._visualize(original_page, all_detections, output_path / f"{page_name}_visual.png")

        return results

    # def _detect_template(self, page: np.ndarray, variants: List[np.ndarray], class_name: str) -> List[Dict]:
    #     matches = []
    #     for template in variants:
    #         result1 = cv2.matchTemplate(page, template, cv2.TM_CCOEFF_NORMED)
    #         result2 = cv2.matchTemplate(page, template, cv2.TM_CCORR_NORMED)
    #         yloc, xloc = np.where(result1 >= self.threshold)
    #         for (x, y) in zip(xloc, yloc):
    #             if result2[y, x] <= SECONDARY_CONFIRMATION_THRESHOLD:
    #                 continue
    #             h, w = template.shape
    #             matches.append({
    #                 'x': int(x), 'y': int(y),
    #                 'width': int(w), 'height': int(h),
    #                 'score': float(result1[y, x]),
    #                 'name': class_name
    #             })
    #     return matches

    def _detect_template(self, page: np.ndarray, variants: List[np.ndarray], class_name: str) -> List[Dict]:
        matches = []
        for template in variants:
            result1 = cv2.matchTemplate(page, template, cv2.TM_CCOEFF_NORMED)
            result2 = cv2.matchTemplate(page, template, cv2.TM_CCORR_NORMED)
            yloc, xloc = np.where(result1 >= self.threshold)

            for (x, y) in zip(xloc, yloc):
                if result2[y, x] <= SECONDARY_CONFIRMATION_THRESHOLD:
                    continue

                h, w = template.shape
                region = page[y:y+h, x:x+w]

                if region.shape[0] <= 5 or region.shape[1] <= 5:
                    continue

                # ✅ SSIM check
                try:
                    region_resized = cv2.resize(region, (w, h))
                    score = ssim(region_resized, template)
                    if score < 0.80:   # stricter threshold
                        continue
                except:
                    continue

                # ✅ Edge density ratio check
                Et = edge_density(template)
                Er = edge_density(region)
                if Er < 0.5 * Et:   # candidate has too few edges compared to template
                    continue

                # ✅ Circle rejection (skip for fan class)
                if class_name != "fan" and is_circle_like(region):
                    continue

                matches.append({
                    'x': int(x), 'y': int(y),
                    'width': int(w), 'height': int(h),
                    'score': float(result1[y, x]),
                    'name': class_name
                })

        return matches


    def _cross_class_nms(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []
        detections.sort(key=lambda d: d['score'], reverse=True)
        filtered = []
        while detections:
            best = detections.pop(0)
            filtered.append(best)
            # remove any detection overlapping significantly with best
            detections = [d for d in detections if self._iou(best, d) < 0.1]
        return filtered


    def _iou(self, d1: Dict, d2: Dict) -> float:
        x1, y1, x2, y2 = d1['x'], d1['y'], d1['x']+d1['width'], d1['y']+d1['height']
        a1, b1, a2, b2 = d2['x'], d2['y'], d2['x']+d2['width'], d2['y']+d2['height']
        xi1, yi1, xi2, yi2 = max(x1, a1), max(y1, b1), min(x2, a2), min(y2, b2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = d1['width']*d1['height'] + d2['width']*d2['height'] - inter
        return inter / union if union > 0 else 0.0

    def _distance(self, d1: Dict, d2: Dict) -> float:
        c1x, c1y = d1['x']+d1['width']//2, d1['y']+d1['height']//2
        c2x, c2y = d2['x']+d2['width']//2, d2['y']+d2['height']//2
        return np.sqrt((c1x-c2x)**2 + (c1y-c2y)**2)

    def _visualize(self, img, detections, save_path):
        """Visualize detections with bounding boxes and labels."""
        vis_img = img.copy()
        for det in detections:
            x, y, w, h = det["x"], det["y"], det["width"], det["height"]
            name = det["name"]
            score = det["score"]
            color = (0, 255, 0)  # bright green

            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 3)

            label = f"{name} {score:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, LABEL_FONT, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
            )
            label_y = max(y - 5, 0)
            label_bg_top = max(label_y - th - baseline, 0)
            label_bg_bottom = min(label_y + baseline, vis_img.shape[0] - 1)
            cv2.rectangle(vis_img, (x, label_bg_top), (x + tw, label_bg_bottom), color, -1)
            cv2.putText(vis_img, label, (x, label_y), LABEL_FONT,
                        LABEL_FONT_SCALE, (0, 0, 0), LABEL_FONT_THICKNESS, cv2.LINE_AA)

        total = len(detections)
        summary_text = f"Total detections: {total}"
        cv2.putText(vis_img, summary_text, (10, 30), LABEL_FONT, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(save_path), vis_img)
        print(f"  Visual saved: {save_path}")

# =====================================================================================
# BATCH DETECTION FUNCTION
# =====================================================================================

def detect_all_pages(pages_folder: str = "assets/dataset/pages_processed", 
                     assets_folder: str = "assets/icons",
                     output_folder: str = "outputs", visual_output: bool = True,
                     threshold: float = 0.7):

    detector = IconDetector(assets_folder, threshold=threshold, visual_output=visual_output)
    pages_path = Path(pages_folder)
    page_files = []
    for ext in ["*.tiff", "*.tif", "*.png"]:
        page_files.extend(pages_path.glob(ext))
    if not page_files:
        raise ValueError(f"No images in {pages_folder}")

    all_results, total = {}, 0
    for f in page_files:
        results = detector.detect_icons(str(f), output_folder)
        all_results[f.name] = results
        page_total = sum(results.values())
        total += page_total
        print(f"  {f.name}: {page_total} detections")

    with open(Path(output_folder) / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary: {total} detections across {len(page_files)} pages")
    return all_results

# =====================================================================================
# RUN DETECTION
# =====================================================================================

if __name__ == "__main__":
    detect_all_pages()
