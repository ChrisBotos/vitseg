import argparse
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans

def main():
    # Parse command-line arguments provided by the user.
    parser = argparse.ArgumentParser(
        description="Cluster DINO patch features and associate cluster labels with patch coordinates."
    )
    parser.add_argument(
        "-c", "--coords_csv", required=True, type=Path,
        help="Path to the CSV file containing x_center and y_center coordinates."
    )
    parser.add_argument(
        "-f", "--features_csv", required=True, type=Path,
        help="Path to the CSV file containing DINO feature vectors."
    )
    parser.add_argument(
        "-n", "--n_clusters", type=int, default=10,
        help="Number of clusters to form with KMeans."
    )
    parser.add_argument(
        "-o", "--output_prefix", type=str, default=None,
        help="Optional prefix for the output cluster file name."
    )
    args = parser.parse_args()

    # Read the coordinates CSV into a dataframe.
    coords_df = pd.read_csv(args.coords_csv)
    # Read the features CSV into a dataframe.
    features_df = pd.read_csv(args.features_csv)

    # Validate that the number of coordinates matches the number of feature vectors.
    if len(coords_df) != len(features_df):
        raise ValueError(
            "The number of rows in coords CSV must match the number of rows in features CSV."
        )

    # Initialize KMeans clustering model with the user-specified number of clusters.
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
    # Fit the clustering model to DINO features and predict cluster labels for each patch.
    labels = kmeans.fit_predict(features_df.values)

    # Combine original coordinates with their corresponding cluster labels.
    result_df = coords_df.copy()
    result_df['cluster'] = labels

    # Determine output file name and ensure results are saved in the same folder as inputs.
    prefix = args.output_prefix or args.features_csv.stem
    output_file = args.coords_csv.parent / f"{prefix}_clusters.csv"
    result_df.to_csv(output_file, index=False)

    # Inform the user of successful saving of cluster results.
    print(f"Cluster assignments saved to: {output_file}")

if __name__ == "__main__":
    main()
