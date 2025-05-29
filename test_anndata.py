import squidpy as sq                    # Recommended for spatial Visium data
import matplotlib.pyplot as plt         # Standard plotting library

# 1. Load a public Visium H&E dataset via Squidpy
print("\n▶ Loading Visium H&E dataset via Squidpy…\n")
adata = sq.datasets.visium_hne_adata()

# 2. Show a summary of the AnnData object
print(f"▶ AnnData summary:\n{adata}\n")

# 3. Ensure gene names are unique
print("▶ Ensuring unique gene names…\n")
adata.var_names_make_unique()
print(f"First 5 gene names: {list(adata.var_names[:5])}\n")

# 4. Preview spot metadata and gene metadata
print("▶ Spot metadata (.obs) preview:\n", adata.obs.head(), "\n")
print("▶ Gene metadata (.var) preview:\n", adata.var.head(), "\n")

# 5. Inspect spatial coordinates
coords = adata.obsm["spatial"]
print(f"▶ Spatial coordinates shape: {coords.shape}\n")

# 6. Display the high-resolution histology image
print("▶ Displaying histology image…\n")
img = adata.uns["spatial"][next(iter(adata.uns["spatial"]))]["images"]["hires"]
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis('off')
plt.show()
