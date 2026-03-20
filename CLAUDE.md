# CLAUDE.md – vigseg

This file provides **mandatory rules** and **project context** for Claude Code when editing or generating code in this repository.

---

## **1. Project Overview**

**vigseg** is a Vision Transformer analysis pipeline for kidney tissue samples. It extracts multi-scale ViT embeddings from microscopy images, clusters them, and compares the results against spatial transcriptomics data. The biological focus is Ischemia/Reperfusion Kidney Injury (I/R) at three time points (10 hours, 2 days, 14 days).

**This is a standalone repository.** It does not depend on or interact with any sibling repositories.

**Current project goal:** Clean-up, organization, and systematic mistake catching. Claude must prioritize code quality, consistency, and correctness over adding new features.

**Core pipeline stages:**

```
1. Filter segmentation masks     (vigseg.preprocessing.filter_masks)
2. Convert to binary TIFF        (vigseg.preprocessing.binary_conversion)
3. Extract ViT embeddings        (vigseg.extraction.dynamic_patches_vit / uniform_tiling_vit)
4. Filter features by scale      (vigseg.clustering.filter_features)
5. Cluster embeddings            (vigseg.clustering.cluster_dynamic_patches / cluster_uniform_tiles)
```

---

## **2. Directory Structure**

The project uses a **src-layout** Python package (`vigseg`) installable via `pip install -e .`.

```
vigseg/
├── src/vigseg/                             # Main package (src-layout)
│   ├── __init__.py                        # Public API, __version__
│   ├── cli.py                             # CLI entry point (vigseg command)
│   ├── preprocessing/                     # Pipeline stages 1-2
│   │   ├── filter_masks.py               # Stage 1: QA mask filtering
│   │   └── binary_conversion.py          # Stage 2: mask-to-binary TIFF
│   ├── extraction/                        # Pipeline stage 3
│   │   ├── dynamic_patches_vit.py        # Stage 3a: per-nuclei ViT extraction
│   │   └── uniform_tiling_vit.py         # Stage 3b: grid-based ViT extraction
│   ├── clustering/                        # Pipeline stages 4-5
│   │   ├── filter_features.py            # Stage 4: filter features by box size
│   │   ├── cluster_dynamic_patches.py    # Stage 5a: cluster dynamic patches
│   │   ├── cluster_uniform_tiles.py      # Stage 5b: cluster uniform tiles
│   │   └── cluster_spots_by_nuclei.py    # Spot-nuclei clustering analysis
│   ├── visualization/                     # All visualization modules
│   │   ├── overlay_masks.py              # Mask overlay visualization
│   │   ├── cluster_circles.py            # Circle-based cluster visualization
│   │   └── crop.py                       # Image cropping utility
│   ├── comparison/                        # Cluster comparison framework
│   │   ├── cluster_metrics.py            # ARI, NMI, silhouette
│   │   ├── spatial_analysis.py           # Moran's I, LISA, Getis-Ord G
│   │   ├── visualization_suite.py        # Comparison visualizations
│   │   └── main_analysis.py              # Orchestration
│   └── utilities/                         # Shared utilities
│       ├── color_generation.py           # High-contrast color generation
│       ├── color_config.py               # Color palette configuration
│       ├── spatial_alignment.py          # Spatial alignment verification
│       ├── prepare_vit_data.py           # ViT data preparation
│       ├── cluster_comparison_simple.py  # Simple cluster comparison
│       └── assess_vit_quality.py         # ViT quality assessment
├── tests/                                 # All tests (flat, prefixed by module)
├── archive/                               # Orphaned files (preserved, not in package)
├── comparison_analysis/                   # Output directories (gitignored)
│   ├── results/                           # Analysis outputs
│   ├── visualizations/                    # Generated figures
│   └── reports/                           # Scientific reports
├── data/                                  # Input data (gitignored)
├── masks/                                 # Segmentation masks (gitignored)
├── results/                               # All analysis outputs (gitignored)
├── pyproject.toml                         # Build config, metadata, CLI entry points
├── pipeline.sh                            # Main analysis pipeline (bash wrapper)
├── environment.yml                        # Conda environment (Python 3.10)
├── requirements.txt                       # Python dependencies (pip-only alternative)
├── README.md
├── CLAUDE.md                              # This file
└── .gitignore
```

---

## **3. Running the Pipeline**

### **Full Pipeline**
```bash
./pipeline.sh
```

The pipeline is controlled by step flags at the top of `pipeline.sh`. Set each to `True` or `False`:
- `RUN_FILTER_MASKS` — Step 3.1: Filter segmentation masks.
- `RUN_BINARY_CONVERSION` — Step 3.2: Convert mask set to binary TIFF.
- `RUN_VIT_EXTRACTION` — Step 3.3: Extract ViT patch embeddings.
- `RUN_FEATURE_FILTERING` — Step 3.4: Filter features by box sizes.
- `RUN_CLUSTERING` — Step 3.5: Cluster the embeddings.

### **Python CLI**
```bash
pip install -e .
vigseg --help
vigseg --steps filter_masks binary vit_extraction filtering clustering
```

### **Run Tests**
```bash
pytest tests/ -v
```

---

## **4. Environment Expectations**

Claude must always assume:

- **You are running inside WSL** (Linux environment) or Windows.
- **Python 3.10** via a conda environment named `vitseg_310`.
- The conda environment is already activated.
- Dependencies from `environment.yml` are installed.

**Activating the environment (if needed):**
```bash
conda activate vitseg_310
```

**Recreating the environment (if needed):**
```bash
conda env create -f environment.yml
conda activate vitseg_310
pip install -e .
```

**Large file rule:**
Many source files and data files are large. Claude must **never** load large data files (`.tif`, `.npy`, `.csv` with 100k+ rows) in full. Use targeted extraction with line ranges or grep.

---

## **5. Instructions for Claude**

### **5.1 General Behavior — Clean-Up and Organization Focus**

The primary goal of this project phase is **clean-up, organization, and systematic mistake catching**. Claude must:

- **Audit before editing.** Before modifying any file, read it and identify inconsistencies, dead code, unused imports, duplicated logic, naming issues, and missing error handling.
- Prefer **minimal, local, safe edits** that preserve existing structure.
- **Do not** attempt large-scale rewrites or architectural changes unless explicitly asked.
- **Do not** add new features unless explicitly asked. The focus is on fixing what exists.
- When finding a bug, fix the **root cause**, not the symptom. Never add try/except wrappers, guard clauses, or default returns to suppress errors without understanding why they occur (see Section 5.7).
- Flag suspicious code with a `# TODO(cleanup):` comment if the fix is non-trivial and requires user input.

### **5.2 Coding Style**

#### **Comments & Documentation**
- All explanatory comments must be full sentences ending with a **full stop**.
- Function-level docstrings must be **Google-style**, including:

```python
"""Short summary.

Args:
    param_name (type): Description.
Returns:
    type: Description.
Raises:
    ErrorType: Description.
"""
```

- Include parameter types, return types, and assumptions.

#### **Titles / Subtitles**
- Titles must use: `"""Title"""`
- Subtitles must use: `'''Subtitle'''`
- No alternative formats.

#### **Code Quality**
- Prefer **small, testable functions** rather than long monolithic blocks.
- Maintain vectorized NumPy operations; avoid Python loops in tight paths.
- Strive for optimized efficient code that matches the style of the rest of the file.
- Remove dead code, unused imports, and commented-out blocks during clean-up.
- Ensure consistent naming conventions within each file.

### **5.3 Systematic Mistake Catching**

Claude must actively look for and report these categories of issues:

1. **Import issues:** Unused imports, missing imports, circular imports.
2. **Variable shadowing:** Local variables shadowing outer scope or built-in names.
3. **Inconsistent defaults:** The same parameter having different default values in different files.
4. **Path issues:** Hardcoded paths that break when the script is run from a different directory. All paths should be resolved relative to `__file__`, not `os.getcwd()`.
5. **Type mismatches:** Functions receiving unexpected types due to upstream changes.
6. **Silent failures:** Bare `except:` clauses, empty `except` blocks, or `pass` in exception handlers.
7. **Resource leaks:** File handles, GPU memory, or large arrays not being released.
8. **Copy-paste errors:** Duplicated code blocks with subtle differences that should be unified.
9. **Off-by-one errors:** Especially in coordinate handling, array indexing, and patch boundary calculations.
10. **Coordinate system inconsistencies:** Pixel coordinates vs. tissue coordinates, Y-axis orientation differences.

When finding an issue, Claude must:
1. Fix it if the fix is safe and obvious.
2. Add a `# TODO(cleanup):` comment if the fix requires user input.
3. Report it in the session summary.

### **5.4 Performance Expectations**

- Always preserve or improve computational complexity.
- Maintain memory-efficient patterns (spatial batching, streaming) already present in the codebase.
- When modifying GPU-accelerated code, preserve CPU fallback paths.
- Never load entire result directories or large data arrays into memory unnecessarily.

### **5.5 Testing**

For **any** nontrivial code change:

- Add or update tests under `tests/`.
- Tests must cover:
  - Input validation and edge cases.
  - Coordinate handling and transformations.
  - Feature extraction correctness.
  - Clustering reproducibility with fixed seeds.
  - Color generation and consistency.
- Use deterministic seeds in tests unless randomness is the purpose.
- Run tests using:

```bash
pytest tests/ -v
```

### **5.6 File Organization**

- All source code lives in `src/vigseg/` organized by functional subpackage.
- Subpackages: `preprocessing`, `extraction`, `clustering`, `visualization`, `comparison`, `utilities`.
- All imports use the `vigseg.subpackage.module` pattern (e.g. `from vigseg.utilities.color_generation import generate_color_palette`).
- Tests are flat under `tests/`: `tests/test_<module_name>.py`.
- Keep public function signatures **stable** unless explicitly changing them everywhere:
  - Update all call sites.
  - Update docstrings.
  - Update tests.
  - Update `pipeline.sh` if the function is called from the pipeline.

### **5.7 Bug Fixing Policy**

When debugging an error, Claude **must** find and fix the **root cause**. Claude must **never** make the bug silent by adding graceful handling, try/except wrappers, default returns, or guard clauses that suppress the symptom without understanding why it occurs.

**Prohibited patterns:**
- Adding `if len(x) == 0: return` to hide a mismatch that should not happen.
- Wrapping a crash site in `try/except` to swallow the error.
- Adding fallback values or empty-array guards at the symptom location instead of tracing why the data is wrong upstream.
- Any modification to the function that **receives** bad data, when the real fix is in the code that **produces** bad data.

**Required approach:**
1. When an error occurs, trace the data flow **upstream** to find where the incorrect state originates.
2. Add temporary debug prints or assertions to confirm the root cause before applying a fix.
3. Fix the code that **produces** the wrong data, not the code that **consumes** it.
4. Remove all debug prints after the fix is confirmed.
5. If the root cause is genuinely unclear after investigation, **ask the user** rather than silencing the error.

### **5.8 Script Path Resolution Policy**

All executable scripts **must** work correctly when invoked from **any** working directory.

**For Python files:**
```python
# GOOD: Resolve relative to the script's own location.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(repo_root, "data")

# BAD: Breaks when called from a different directory.
data_dir = os.path.join("..", "data")
```

**For shell scripts:**
```bash
# GOOD: Resolve from script location.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$script_dir" && pwd)"

# BAD: Assumes CWD.
REPO_DIR="../"
```

---

## **6. Author Header Block**

For every **Python file** that Claude creates or substantially modifies, include:

```python
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: <filename>.py.
Description:
    <Brief description of what this file does.>

Dependencies:
    • Python >= 3.10.
    • <list relevant dependencies>

Usage:
    <how to run this script>
"""
```

---

## **7. Git Policy**

- Never git add, commit, stash, tag, or pull.
- Only update and deal with the local version of this repository.
- Never leave comments related to version changes like: `# Version: 1.1 — removed X`.

---

## **8. Scientific Context**

### **Kidney Injury Model**
- **Condition:** Ischemia/Reperfusion (I/R) Injury.
- **Time points:** 10 hours, 2 days (48 hours), 14 days.
- **Controls:** Sham-operated animals.
- **Analysis samples:** IRI1, IRI2, IRI3, sham1, sham2, sham3.

### **ViT Model**
- **Model:** DINO ViTs (facebook/dino-vits16) — self-supervised Vision Transformer.
- **Multi-scale patches:** 16px, 32px, 64px capturing cellular to tissue-level patterns.
- **Two extraction modes:**
  - **Dynamic patches:** Centered on individual nuclei from segmentation masks.
  - **Uniform tiling:** Regular grid across the entire image.

### **Quality Control Thresholds (Morphological)**
- Pixel area: 20–900.
- Circularity: 0.56–1.00.
- Solidity: 0.765–1.00.
- Eccentricity: 0.00–0.975.
- Aspect ratio: 0.50–3.20.
- Hole fraction: 0.00–0.001.

### **Statistical Metrics for Cluster Comparison**
- **ARI** (Adjusted Rand Index): Cluster agreement corrected for chance.
- **NMI** (Normalized Mutual Information): Information overlap.
- **Silhouette analysis:** Cluster quality assessment.
- **Moran's I:** Global spatial autocorrelation.
- **LISA:** Local Indicators of Spatial Association.
- **Getis-Ord G:** Hotspot/coldspot detection.

---

## **9. Tips & Tricks**

### **Working with This Codebase**

- **Two ViT extraction modes exist:** Dynamic patches (per-nuclei, `vigseg.extraction.dynamic_patches_vit`) and uniform tiling (`vigseg.extraction.uniform_tiling_vit`). The pipeline flag `USE_DYNAMIC_PATCHES` controls which is used. NEVER confuse them.
- **Two clustering modules exist:** `vigseg.clustering.cluster_dynamic_patches` (for dynamic patches, requires label maps) and `vigseg.clustering.cluster_uniform_tiles` (for uniform tiles, works with tile coordinates only). They are NOT interchangeable.
- **Coordinate systems differ.** ViT coordinates are small pixel-based (X: 2–502, Y: 1–551). Spatial transcriptomics coordinates are large tissue-based (X: 4,662–20,782, Y: 5,388–25,028). Always verify which coordinate system is in use before combining data.
- **Results directory is large.** Many subdirectories with output files. NEVER attempt to list or read it in full.
- **Source code is in `src/vigseg/`** organized by functional subpackage. All imports use the `vigseg.subpackage.module` pattern.
- **Feature files come in two formats:** CSV and NPY. The clustering scripts prefer NPY for speed but fall back to CSV. After feature filtering, only CSV files exist (no NPY).

### **Working with Claude Code Effectively**

- **NEVER read large files in full.** Many source files exceed 30KB. Use targeted line ranges or grep for specific functions.
- **Use subagents for exploration.** Delegate broad file searches to Explore agents to keep main context clean.
- **Verify after changes.** Run `pytest tests/ -v` to check correctness.
- **Background long runs.** ViT extraction and full-image clustering can take hours. Use `run_in_background` and monitor.
- **Check pipeline.sh after code changes.** If a script's CLI interface changes, update `pipeline.sh` to match.

### **Common Mistakes to Watch For**

- **Stem mismatch in pipeline.sh:** The pipeline derives file stems from the binary image name. If a script outputs files with a different naming convention, the next pipeline stage will fail with "file not found".
- **Y-axis flipping:** Some visualizations flip the Y axis, others do not. The `--flip-y` flag controls this. Inconsistent use produces mirrored/inverted results.
- **Memory exhaustion:** Large TIFF images (10,000+ x 10,000+ pixels) can exhaust RAM if loaded fully. The clustering modules already use memory-efficient streaming patterns.
- **Cluster count mismatch:** The `K_INIT` parameter in `pipeline.sh` must match expectations in downstream analysis scripts that assume a specific number of clusters.
