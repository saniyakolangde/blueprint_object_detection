# YOLOE - Simplified Icon Detection Tool

Clean, focused icon detection with visual output and evaluation framework.

## 🚀 Quick Start

```bash
# Run detection on default folders
uv run main.py

# Run with custom settings  
uv run main.py --threshold 0.5 --no-visual

# Run complete test suite
uv run test.py
```

## 📁 Project Structure

```
yoloe/
├── assets/
│   ├── icons/           # Icon template images (.tiff/.tif)
│   └── pages/           # Page images to analyze (.tiff/.tif)  
├── outputs/             # Generated results
│   ├── page1.json       # Detection counts
│   ├── page1_visual.png # Visual output with bounding boxes
│   └── all_results.json # Combined results
├── evals/
│   ├── evaluator.py     # Evaluation framework
│   └── config.json      # Ground truth configuration
├── yoloe/
│   └── detector.py      # Single optimized detector
├── main.py              # Main detection script
└── test.py              # Test runner with evaluation
```

## 🎯 Features

- **Single detector file** - All detection logic in `yoloe/detector.py`
- **TIFF support** - Handles both .tiff and .tif input files for icons and pages
- **Visual output** - Bounding boxes with confidence scores and colors per icon type
- **Optimized matching** - 80 variants per template (16 rotations × 5 scales)
- **Smart filtering** - Reduces false positives with adaptive thresholds
- **Easy evaluation** - Benchmark against ground truth with `test.py`
- **Configurable hyperparameters** - Easy experimentation with detection constants

## 📊 Current Performance

**Detection Results:**
- Page 1: 24 icons (8 sockets, 14 isolators, 2 USB)
- Page 2: 18 icons (1 fan, 7 sockets, 9 isolators, 1 USB)

**Evaluation Metrics (vs Ground Truth):**
- F1 Score: 0.220
- Precision: 0.257  
- Recall: 0.220

## 🔧 Usage

### Basic Detection
```bash
uv run main.py
```

### Custom Parameters
```bash
# Lower threshold for more detections
uv run main.py --threshold 0.5

# Skip visual output for speed
uv run main.py --no-visual

# Custom folders
uv run main.py --pages custom/pages --assets custom/icons
```

### With Evaluation
```bash
# Run detection + evaluation
uv run test.py

# Edit ground truth first
# Edit evals/config.json with actual icon counts
uv run test.py
```

## 🖼️ Visual Output

The system generates visual outputs showing:
- **Colored bounding boxes** around detected icons
- **Labels with confidence scores** 
- **Different colors per icon type**
- **Detection summary** on image

Example: `outputs/page1_visual.png`

## ⚙️ Configuration

### Detection Settings
- **Threshold**: 0.6 (lower = more sensitive)
- **Rotations**: 16 angles (every 22.5°) 
- **Scales**: 0.8x to 1.2x
- **Filtering**: Smart false positive reduction

### Ground Truth (evals/config.json)
```json
{
  "pages": {
    "page1.png": {
      "expected_counts": {
        "fan": 0,
        "10amp-socket": 4,
        "15amp-socket": 4, 
        "dbl-10amp-usb-socket": 24,
        "perm-connect-isolator": 5
      }
    }
  }
}
```

## 🛠️ Improving Detection

### 1. Adjust Threshold
```bash
# More sensitive (more detections, possibly more false positives)
uv run main.py --threshold 0.5

# Less sensitive (fewer detections, higher precision)
uv run main.py --threshold 0.7
```

### 2. Template Quality
- Ensure templates have clear edges and high contrast
- Crop templates tightly (minimal white space)
- Templates should be representative of page icons

### 3. Page Preprocessing  
The detector automatically applies:
- Contrast enhancement (CLAHE)
- Noise reduction (bilateral filtering)
- Adaptive thresholding

## 📈 Evaluation Framework

Run `test.py` to:
1. Execute detection on all pages
2. Compare against ground truth configuration
3. Generate detailed metrics (precision, recall, F1)
4. Save evaluation results to `outputs/`

**Key Metrics:**
- **Precision**: How many detections were correct
- **Recall**: How many actual icons were found
- **F1 Score**: Harmonic mean of precision and recall

## 🚨 Troubleshooting

### No Detections
- Lower threshold: `--threshold 0.5`
- Check template quality and contrast
- Verify page images contain target icons

### Too Many False Positives  
- Raise threshold: `--threshold 0.7`
- Check templates aren't mostly white/background
- Review visual output to identify problem areas

### Missing Specific Icons
- Check if template matches page icon style
- Try different preprocessing on templates
- Consider creating multiple templates per icon type

## 🎯 Next Steps for 100% Accuracy

1. **Custom YOLO Training** - Train on your specific page/icon combinations
2. **Template Refinement** - Extract actual icon instances from pages as templates  
3. **Multi-threshold Ensemble** - Combine results from different thresholds
4. **Feature-based Matching** - Use SIFT/ORB for rotation/scale invariant detection

---

**Perfect for**: Electrical diagrams, architectural plans, technical drawings with repeated symbols.