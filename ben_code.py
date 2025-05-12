import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import anndata as ad
import scanpy as sc
from collections import Counter
import numpy as np
from scipy.spatial.distance import cosine
from itertools import product

nieromics_dir = "/exports/humgen/bmanzato/nieromics_dir"
datadir = '/exports/humgen/bmanzato/nieromics_dir/benedetta/deconv'
lts = "/exports/archive/hg-funcgenom-research/IRI_multimodal_project/Stereo-seq_IRI"

tissue = 'IRI1'

### read spatial data
iri1_spt = ad.read_h5ad(f"/exports/humgen/bmanzato/IRI_multimodal_project/Stereo-seq_IRI/kidney_b{tissue}.h5ad")
sc.pp.normalize_total(iri1_spt)
sc.pp.log1p(iri1_spt)
iri1_spt

stseq_ann = pd.read_csv(f"{nieromics_dir}/spatial_multiomics_data_MS/transcriptomics/metadata_complete.csv",index_col=0)
stseq_ann = stseq_ann[stseq_ann['condition']=='IRI']
stseq_ann['banksy'] = stseq_ann['banksy'].astype('Int64')
stseq_ann.columns = ['x','y','condition','sample','SS_SD','SS_CT']
stseq_ann = stseq_ann[stseq_ann['sample']==tissue]
stseq_ann = stseq_ann[stseq_ann['SS_SD']!='NA']
stseq_ann.index = stseq_ann.index.str[:-2]
iri1_spt.obs = iri1_spt.obs.merge(stseq_ann[['SS_SD', 'SS_CT']], left_index=True, right_index=True, how='left')

# Function to map SS_SD values to categories
def map_ss_sd_group(ss_sd_value):
    if ss_sd_value == 7:
        return 'INJ1'
    elif ss_sd_value == 9:
        return 'INJ2'
    elif ss_sd_value in [2, 3, 4]:
        return 'Hthy'
    else:
        return pd.NA  # Or NaN if you prefer

# Apply the function to create a new column
iri1_spt.obs['SS_SD_Group'] = iri1_spt.obs['SS_SD'].apply(map_ss_sd_group)

# filter out NA
iri1_spt = iri1_spt[iri1_spt.obs['SS_SD_Group'].notna()]

# read the new CT annotations (from 3rd of July)
barcode_ann = pd.read_csv(f"{lts}/barcode_annotation.csv",index_col=0)
barcode_ann['_group'] = barcode_ann.index.str[-1]

# Create two separate DataFrames based on the grouping
sham_ann = barcode_ann[barcode_ann['_group'].isin(['1', '2', '3'])].copy()
biri_ann = barcode_ann[barcode_ann['_group'].isin(['4', '5', '6'])].copy()
biri_ann.drop('_group', axis=1, inplace=True)
biri_ann.index = biri_ann.index.str[:-2]
#biri_ann = biri_ann.iloc[0:48347,:] # biri1 only
iri1_spt.obs['celltype'] = biri_ann

### INJ 1 and INJ 2 are the injured microenvironments