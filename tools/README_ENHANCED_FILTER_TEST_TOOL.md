# Enhanced Filter Test Tool - Usage Guide

**Created:** 2026-01-10
**Purpose:** Comparative testing framework for PROMPT-5 noise filtering variants
**Location:** `/home/ws/tools/enhanced_filter_test_tool.py`

---

## Overview

The Enhanced Filter Test Tool provides comprehensive comparative analysis of 5 noise filtering variants:

1. **original** - Baseline (variance-based, single component)
2. **density_enhanced** - MAD + increased z_variance
3. **ring_aware** - Ring thresholds + multi-component
4. **full_enhanced** - All enhancements (recommended)
5. **temporal** - Full + 3-frame history (future)

---

## Prerequisites

### Required Python Packages

```bash
# Install rosbags for MCAP reading
pip3 install rosbags

# Install matplotlib for visualization
pip3 install matplotlib

# Install numpy (usually already available)
pip3 install numpy
```

### Required ROS2 Package

The tool imports filter implementations from the drivable_area_detector package. Ensure it's built:

```bash
cd /home/ws
colcon build --packages-select drivable_area_detector
source install/setup.bash
```

---

## Usage

### Basic Usage

```bash
python3 /home/ws/tools/enhanced_filter_test_tool.py \
    /home/ws/rosbag/spanlidar/spanlidar_0.mcap \
    --max-frames 300 \
    --output-dir /home/ws/tools/output
```

### Advanced Usage

```bash
# Test specific variants only
python3 /home/ws/tools/enhanced_filter_test_tool.py \
    /home/ws/rosbag/sy27_road_ground_truth_live/sy27_..._0.mcap \
    --max-frames 500 \
    --variants original full_enhanced \
    --intensity-range 0.0 10.0 \
    --output-dir /home/ws/tools/output_sy27
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `bag_file` | str | Required | Path to MCAP bag file |
| `--max-frames` | int | 300 | Maximum frames to process |
| `--output-dir` | str | `/home/ws/tools/output` | Output directory |
| `--variants` | list | `original density_enhanced ring_aware full_enhanced` | Variants to test |
| `--intensity-range` | float float | `0.0 8.0` | Ground truth intensity range |

---

## Output Files

The tool generates 6 output files:

### 1. variant_comparison.png

Bar chart comparing all variants across 4 metrics:
- Precision (target: >90%)
- Recall/Detection Rate (target: >50%)
- F1 Score (target: >60%)
- Latency (budget: <100ms)

**Use case:** Quick visual comparison to identify best overall variant

### 2. precision_recall_curves.png

Precision-Recall space plot with:
- Per-frame scatter points (small dots)
- Average performance markers (large stars)
- F1 contour lines (0.4, 0.5, 0.6, 0.7, 0.8)

**Use case:** Understand precision/recall trade-offs

### 3. temporal_analysis.png

F1 score over time for each variant:
- Line plots showing temporal stability
- Average F1 score per variant
- Target line at F1=0.60

**Use case:** Identify variants with temporal jitter or instability

### 4. latency_distribution.png

Box plot of processing times:
- Median, quartiles, outliers
- Mean line
- 100ms budget line

**Use case:** Verify real-time compliance and identify latency outliers

### 5. results_summary.json

Machine-readable results:
```json
{
  "test_date": "2026-01-10 14:30:00",
  "bag_file": "/home/ws/rosbag/spanlidar/spanlidar_0.mcap",
  "frames_processed": 300,
  "variants": [
    {
      "variant": "full_enhanced",
      "avg_precision": 0.9123,
      "avg_recall": 0.6845,
      "avg_f1": 0.7821,
      "avg_latency_ms": 18.5
    }
  ],
  "rankings": {
    "f1": [["full_enhanced", 0.7821], ["ring_aware", 0.7512], ...],
    "precision": [...],
    "recall": [...],
    "latency": [...]
  }
}
```

**Use case:** Automated analysis, CI/CD integration, data export

### 6. recommendations.md

Production deployment recommendation:
- Best variant by F1 score
- Target compliance summary
- Rankings by all metrics
- Variant comparison table
- Usage instructions for applying recommendation

**Use case:** Final decision-making and configuration updates

---

## Workflow

### Step 1: Run Comparative Test

```bash
python3 /home/ws/tools/enhanced_filter_test_tool.py \
    /home/ws/rosbag/spanlidar/spanlidar_0.mcap \
    --max-frames 300 \
    --output-dir /home/ws/tools/output_spanlidar
```

Expected output:
```
======================================================================
Processing bag: /home/ws/rosbag/spanlidar/spanlidar_0.mcap
Max frames: 300
Testing variants: original, density_enhanced, ring_aware, full_enhanced
======================================================================

Step 1: Reading bag file...
  ✓ Extracted 300 frames

Step 2: Testing variant 'original'...
  ✓ Avg F1: 0.643 | Precision: 0.889 | Recall: 0.512 | Latency: 12.3ms

Step 2: Testing variant 'density_enhanced'...
  ✓ Avg F1: 0.712 | Precision: 0.902 | Recall: 0.589 | Latency: 14.1ms

Step 2: Testing variant 'ring_aware'...
  ✓ Avg F1: 0.751 | Precision: 0.908 | Recall: 0.643 | Latency: 16.8ms

Step 2: Testing variant 'full_enhanced'...
  ✓ Avg F1: 0.782 | Precision: 0.912 | Recall: 0.685 | Latency: 18.5ms

Step 3: Generating visualizations...
  ✓ Saved: /home/ws/tools/output_spanlidar/variant_comparison.png
  ✓ Saved: /home/ws/tools/output_spanlidar/precision_recall_curves.png
  ✓ Saved: /home/ws/tools/output_spanlidar/temporal_analysis.png
  ✓ Saved: /home/ws/tools/output_spanlidar/latency_distribution.png

Step 4: Exporting results...
  ✓ Saved: /home/ws/tools/output_spanlidar/results_summary.json

Step 5: Generating recommendations...
  ✓ Saved: /home/ws/tools/output_spanlidar/recommendations.md

======================================================================
TESTING COMPLETE
======================================================================

Results saved to: /home/ws/tools/output_spanlidar
Best variant (by F1): full_enhanced (F1=0.782)

Next steps:
  1. Review recommendations.md
  2. Check visualization plots
  3. Update params.yaml with recommended variant
======================================================================
```

### Step 2: Review Results

```bash
# View recommendations
cat /home/ws/tools/output_spanlidar/recommendations.md

# View JSON results
jq . /home/ws/tools/output_spanlidar/results_summary.json

# Open visualizations
xdg-open /home/ws/tools/output_spanlidar/variant_comparison.png
```

### Step 3: Apply Recommendation

Edit `/home/ws/src/drivable_area_detector/config/params.yaml`:

```yaml
road_points_extraction_node:
  # ... existing params ...

  # Apply recommended variant from test results
  filter_variant: "full_enhanced"  # Was: "original"
```

### Step 4: Rebuild and Test

```bash
cd /home/ws
colcon build --packages-select drivable_area_detector
source install/setup.bash

# Test on live bag
ros2 bag play /home/ws/rosbag/spanlidar/spanlidar_0.mcap --clock --rate 1.0
```

---

## Architecture

### Class Structure

```
enhanced_filter_test_tool.py
│
├── Data Structures
│   ├── FilterTestConfig        # Test configuration
│   ├── FrameResult             # Per-frame metrics
│   └── VariantResult           # Aggregate per-variant
│
├── Processing Components
│   ├── GroundTruthGenerator    # Intensity-based pseudo GT
│   ├── MetricsCalculator       # Precision/Recall/F1/Ranking
│   ├── VisualizationGenerator  # 4 plot types
│   └── BagProcessor            # MCAP reading + variant iteration
│
└── Main CLI
    └── main()                  # Orchestration + JSON export
```

### Processing Flow

```
1. Read bag file (MCAP) → Extract point clouds
2. For each variant:
   a. Apply filter configuration
   b. Process all frames
   c. Calculate per-frame metrics
   d. Aggregate results
3. Rank variants by F1, precision, recall, latency
4. Generate 4 visualization plots
5. Export JSON results
6. Generate recommendations markdown
```

---

## Metrics Explained

### Precision

**Definition:** True Positives / (True Positives + False Positives)

**Interpretation:** What percentage of detected road points are actually road?

**Target:** >90% (minimize false positives)

**Example:** Precision = 0.91 means 91% of detected points are correct, 9% are noise

### Recall (Detection Rate)

**Definition:** True Positives / (True Positives + False Negatives)

**Interpretation:** What percentage of actual road points were detected?

**Target:** >50% (maximize coverage)

**Example:** Recall = 0.68 means 68% of road surface detected, 32% missed

### F1 Score

**Definition:** 2 × Precision × Recall / (Precision + Recall)

**Interpretation:** Harmonic mean of precision and recall

**Target:** >60% (balanced performance)

**Example:** F1 = 0.78 indicates good balance between precision and recall

### Retention Rate

**Definition:** Output Points / Input Points

**Interpretation:** What percentage of input points passed filtering?

**Use case:** Sanity check (typical range: 30-70%)

### Latency

**Definition:** Processing time per frame in milliseconds

**Interpretation:** Real-time compliance

**Target:** <100ms per frame (10Hz operation)

---

## Ground Truth Generation

### Intensity-Based Proxy

The tool uses **intensity-based pseudo ground truth**:

```python
ground_truth_mask = (intensity >= min_intensity) & (intensity <= max_intensity)
```

**Default range:** `[0.0, 8.0]` (asphalt LOW intensity)

**Rationale:**
- Asphalt typically has LOW intensity (0-8)
- Road markings have HIGH intensity (20-60), excluded
- Grass/non-road has variable intensity (usually >10)

### Limitations

**This is NOT true ground truth:**
- Intensity-based classification is imperfect
- No manual annotation
- May misclassify some road types

**Use case:** Comparative analysis (same GT for all variants)

**Recommendation:** For production validation, create manual annotations

---

## Troubleshooting

### Error: "rosbags library not found"

**Solution:**
```bash
pip3 install rosbags
```

### Error: "Could not import filter module"

**Cause:** drivable_area_detector package not built or not in path

**Solution:**
```bash
cd /home/ws
colcon build --packages-select drivable_area_detector
source install/setup.bash
```

### Error: "No suitable topics found"

**Cause:** Bag file doesn't contain expected topics

**Expected topics:**
- `/patchworkpp/ground` (preferred)
- `/road_surface` (alternative)
- `/points` (fallback)

**Check bag topics:**
```bash
ros2 bag info /path/to/bag.mcap
```

### Visualization plots not generated

**Cause:** matplotlib not installed or no display available

**Solution:**
```bash
pip3 install matplotlib

# If no display (headless server)
export MPLBACKEND=Agg
```

### Processing very slow

**Cause:** Too many frames or complex filtering

**Solution:** Reduce `--max-frames`:
```bash
python3 enhanced_filter_test_tool.py bag.mcap --max-frames 100
```

---

## Performance Considerations

### Expected Processing Time

| Frames | Variants | Est. Time | Notes |
|--------|----------|-----------|-------|
| 100 | 4 | 30-60s | Quick test |
| 300 | 4 | 2-3 min | Standard test |
| 500 | 4 | 4-6 min | Comprehensive |
| 300 | 5 (with temporal) | 3-5 min | Temporal adds ~20% overhead |

### Memory Usage

- **Typical:** ~500MB RAM for 300 frames
- **Peak:** ~1GB RAM for 500 frames with all variants
- **Output files:** ~5MB total (4 PNG + 1 JSON + 1 MD)

---

## Example Results Interpretation

### Scenario 1: High Precision, Low Recall

```
full_enhanced:
  Precision: 0.95
  Recall: 0.35
  F1: 0.51
```

**Interpretation:**
- Very accurate when detecting (95% correct)
- But misses 65% of road surface
- Too conservative/aggressive filtering

**Action:** Increase detection rate (relax thresholds)

### Scenario 2: Low Precision, High Recall

```
ring_aware:
  Precision: 0.72
  Recall: 0.88
  F1: 0.79
```

**Interpretation:**
- Detects most road surface (88%)
- But includes 28% false positives (noise)
- Too permissive filtering

**Action:** Improve precision (stricter validation)

### Scenario 3: Balanced Performance

```
full_enhanced:
  Precision: 0.91
  Recall: 0.68
  F1: 0.78
```

**Interpretation:**
- Good balance (91% accurate, 68% coverage)
- F1=0.78 exceeds target (>0.60)
- **Recommended for production**

**Action:** Deploy this variant

---

## Integration with CI/CD

### Automated Testing Example

```bash
#!/bin/bash
# CI/CD integration script

# Run test
python3 /home/ws/tools/enhanced_filter_test_tool.py \
    /home/ws/rosbag/spanlidar/spanlidar_0.mcap \
    --max-frames 200 \
    --output-dir /tmp/ci_output

# Parse results
BEST_F1=$(jq -r '.rankings.f1[0][1]' /tmp/ci_output/results_summary.json)
BEST_VARIANT=$(jq -r '.rankings.f1[0][0]' /tmp/ci_output/results_summary.json)

# Validate against targets
if (( $(echo "$BEST_F1 > 0.60" | bc -l) )); then
  echo "✅ CI PASS: Best F1 ($BEST_F1) exceeds target (0.60)"
  echo "Recommended variant: $BEST_VARIANT"
  exit 0
else
  echo "❌ CI FAIL: Best F1 ($BEST_F1) below target (0.60)"
  exit 1
fi
```

---

## Related Documentation

- **PROMPT5_OPTIMIZATION_REPORT.md** - Technical details of enhancements
- **FILTER_TUNING_GUIDE.md** - Manual parameter tuning guide
- **CLAUDE.md Section 14** - PROMPT-5 implementation results
- **test_noise_filtering.py** - Unit test suite for individual enhancements

---

## Contact

**Created by:** PROMPT-5 Implementation
**Date:** 2026-01-10
**Status:** Complete ✅

For issues or questions, refer to project documentation in `/home/ws/doc/`

---

**END OF USAGE GUIDE**
