import pandas as pd
import os

# Check current directory
print("Current directory:", os.getcwd())

# Check if files exist
vit_path = "results/spot_nuclei_clustering/spot_clusters.csv"
spatial_path = "data/metadata_complete.csv"

print(f"ViT file exists: {os.path.exists(vit_path)}")
print(f"Spatial file exists: {os.path.exists(spatial_path)}")

if os.path.exists(vit_path):
    vit_data = pd.read_csv(vit_path)
    print(f"ViT samples: {sorted(vit_data['sample'].unique())}")
    print(f"ViT shape: {vit_data.shape}")
    print(f"ViT columns: {list(vit_data.columns)[:10]}...")  # First 10 columns
    
if os.path.exists(spatial_path):
    spatial_data = pd.read_csv(spatial_path)
    print(f"Spatial samples: {sorted(spatial_data['sample'].unique())}")
    print(f"Spatial shape: {spatial_data.shape}")
    print(f"Spatial columns: {list(spatial_data.columns)}")
