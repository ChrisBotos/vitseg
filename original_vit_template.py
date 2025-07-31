## Extract Image features from Images from Marleen
#### Benedetta Manzato
#### 01-05-2025
#### 224x224

from transformers import ViTImageProcessor, ViTModel, ViTFeatureExtractor
from PIL import Image, ImageOps
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
import requests
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
import anndata as ad

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from tqdm import tqdm
import os

import torch
print(torch.cuda.is_available())  # This should return True
print(torch.cuda.get_device_name(0))  # This should return the name of your GPU

# Set directory paths — now **relative** to wherever you launch this script
base_dir   = os.path.abspath(os.path.dirname(__file__))  # script’s folder
valid_data = base_dir   # process directly from project root

# ensure the output ViT folder exists
os.makedirs(os.path.join(valid_data, "ViT"), exist_ok=True)

# Load the ViT model
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')


# Function to divide the image into patches
def get_image_patches(img, patch_size, stride):
    patches = []
    img_width, img_height = img.size

    for i in range(0, img_height - patch_size + 1, stride):
        for j in range(0, img_width - patch_size + 1, stride):
            patch = img.crop((j, i, j + patch_size, i + patch_size))
            patches.append(patch)

    return patches

# Function to divide the image into patches and get their coordinates
def get_image_patches_coord(img, patch_size, stride):
    patches = []
    coordinates_dict = {}
    patch_id = 0

    img_width, img_height = img.size

    for i in range(0, img_height - patch_size + 1, stride):
        for j in range(0, img_width - patch_size + 1, stride):
            patch = img.crop((j, i, j + patch_size, i + patch_size))
            patches.append(patch)

            coords = []
            for y in range(i, i + patch_size):
                for x in range(j, j + patch_size):
                    coords.append((x, y))
            coordinates_dict[patch_id] = coords
            patch_id += 1

    return patches, coordinates_dict

# Load the image, add border, and extract features with progress reporting
def ext_features_coord(model, tissue, imgformat):
    print(f"############################################ Starting with {tissue}")

    # Define the path to the image
    img_path = os.path.join(valid_data, "data", f"{tissue}.{imgformat}")
    img = Image.open(img_path).convert('RGB')

    # Get dimensions
    img_width, img_height = img.size

    # Divide the image into smaller patches
    patches = get_image_patches(img, patch_size=16, stride=16)

    # Feature extraction
    print("Extracting features...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), # resize
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    features = []
    total_patches = len(patches)

    print(f"Total number of patches: {total_patches}")
    print("Processing patches...")

    # Progress bar with estimated time remaining
    for i, patch in enumerate(tqdm(patches, desc="Extracting Features", unit="patch")):
        img_t = transform(patch).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            patch_features = model(img_t)
            features.append(patch_features.cpu().numpy())

    features_com = np.concatenate(features, axis=0)

    # Save the features as a CSV file
    df_features = pd.DataFrame(features_com)
    df_features.to_csv(os.path.join(valid_data, "ViT", f"MF_ViT_{tissue}.csv"), index=False)

    print(f"Number of patches for {tissue}: {df_features.shape[0]}")
    print(f"Feature extraction complete. Features saved to: ViT/MF_ViT_{tissue}.csv")

    # Divide the image into smaller patches and get their coordinates
    print("Extracting coordinates...")
    patches, coordinates_dict = get_image_patches_coord(img, patch_size=16, stride=16)

    # Extract specific coordinate for each patch
    for key in range(len(coordinates_dict)):
        coordinates_dict[key] = [coordinates_dict[key][119]]  # Extracting the values into lists of x and y

    x = [coordinates_dict[i][0][0] for i in coordinates_dict]
    y = [coordinates_dict[i][0][1] for i in coordinates_dict]

    # Create the DataFrame for coordinates and save it as CSV
    coord = pd.DataFrame({'x': x, 'y': y})
    coord.to_csv(os.path.join(valid_data, "ViT", f"ViT_coord_{tissue}.csv"), index=False)

    print(f"Coordinates extracted and saved to: ViT/ViT_coord_{tissue}.csv")

# Run on your image only
# Ensure your file is named IRI_regist_cropped.tif under data/
ext_features_coord(vits16, 'IRI_regist_cropped', 'tif')
