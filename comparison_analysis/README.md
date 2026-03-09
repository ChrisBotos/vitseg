# ViT-Spatial Cluster Comparison Analysis

## Overview

This directory contains a comprehensive scientific analysis framework for quantitatively evaluating the alignment between ViT-derived clusters and spatial spot clusters. The analysis provides rigorous statistical methods to measure cluster agreement, spatial correlation, and biological relevance.

## Directory Structure

```
comparison_analysis/
├── scripts/                    # Core analysis modules
│   ├── cluster_metrics.py      # Statistical alignment metrics
│   ├── spatial_analysis.py     # Spatial correlation analysis
│   ├── visualization_suite.py  # Publication-quality visualizations
│   ├── statistical_tests.py    # Significance testing framework
│   └── report_generator.py     # Automated report generation
├── results/                    # Analysis outputs
│   ├── metrics/               # Statistical results
│   ├── spatial/               # Spatial analysis results
│   └── comparisons/           # Cluster comparison data
├── visualizations/            # Generated plots and figures
│   ├── cluster_maps/          # Side-by-side cluster visualizations
│   ├── sankey_diagrams/       # Cluster transition flows
│   ├── scatter_plots/         # Centroid analysis
│   └── heatmaps/             # Overlap matrices
├── reports/                   # Generated scientific reports
└── tests/                     # Test suite and validation
```

## Key Features

### Statistical Metrics
- **Adjusted Rand Index (ARI)** for cluster agreement measurement
- **Normalized Mutual Information (NMI)** for information overlap analysis
- **Silhouette Analysis** for cluster quality comparison
- **Confusion Matrix Analysis** for cluster membership transitions

### Spatial Analysis
- **Moran's I** for spatial autocorrelation
- **Spatial Clustering Validation** metrics
- **Distance-based Correlation** analysis
- **Neighborhood Analysis** for local cluster patterns

### Visualization Suite
- **Side-by-side Cluster Maps** with consistent color schemes
- **Sankey Diagrams** showing cluster membership transitions
- **Scatter Plots** of cluster centroids in feature space
- **Heatmaps** of cluster overlap matrices
- **Publication-quality** output with scientific color palettes

### Statistical Testing
- **Permutation Tests** for cluster alignment significance
- **Bootstrap Confidence Intervals** for stability metrics
- **Multiple Comparison Corrections** (Bonferroni, FDR)
- **Effect Size Calculations** for practical significance

## Usage

### Basic Analysis
```bash
# Run complete comparison analysis
python comparison_analysis/scripts/main_analysis.py \
    --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_clusters data/metadata_complete.csv \
    --output comparison_analysis/results/complete_analysis \
    --samples IRI1 IRI2 IRI3 \
    --cluster_column figure_idents
```

### Individual Components
```bash
# Calculate alignment metrics only
python comparison_analysis/scripts/cluster_metrics.py \
    --vit_data results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_data data/metadata_complete.csv \
    --output comparison_analysis/results/metrics/

# Generate visualizations only
python comparison_analysis/scripts/visualization_suite.py \
    --input comparison_analysis/results/metrics/ \
    --output comparison_analysis/visualizations/
```

## Scientific Methodology

### Cluster Alignment Assessment
1. **Data Preprocessing**: Spatial coordinate alignment and cluster label standardization
2. **Metric Calculation**: Comprehensive statistical measures of cluster agreement
3. **Significance Testing**: Permutation-based statistical validation
4. **Effect Size Analysis**: Practical significance assessment

### Spatial Correlation Analysis
1. **Spatial Weight Matrix**: Construction based on tissue sample geometry
2. **Autocorrelation Testing**: Moran's I and related spatial statistics
3. **Local Clustering**: LISA (Local Indicators of Spatial Association)
4. **Spatial Validation**: Cross-validation with spatial constraints

### Biological Interpretation
1. **Tissue Context**: Integration with known kidney tissue architecture
2. **Cell Type Mapping**: Alignment with established cell type markers
3. **Functional Relevance**: Assessment of biological meaningfulness
4. **Clinical Implications**: Relevance to disease states and conditions

## Output Interpretation

### Alignment Metrics
- **ARI > 0.7**: Strong cluster alignment
- **ARI 0.3-0.7**: Moderate alignment
- **ARI < 0.3**: Weak alignment
- **NMI > 0.8**: High information overlap
- **Silhouette Score > 0.5**: Well-separated clusters

### Spatial Metrics
- **Moran's I > 0.3**: Significant spatial clustering
- **p-value < 0.05**: Statistically significant spatial pattern
- **Local clustering**: Hotspots and coldspots identification

## Dependencies

- Python >= 3.10
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn, plotly
- rich (enhanced console output)
- statsmodels (spatial statistics)
- networkx (graph analysis)

## Citation

When using this analysis framework, please cite:
```
Botos, C. et al. (2025). Quantitative Assessment of ViT-Derived Spatial Clustering 
in Kidney Tissue Analysis. Journal of Computational Biology.
```

## Contact

For questions or issues, contact:
- **Author**: Christos Botos
- **Email**: botoschristos@gmail.com
- **Affiliation**: Leiden University Medical Center
