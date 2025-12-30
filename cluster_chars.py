import os
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
import hdbscan


def load_paths(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f]

# Load and preprocess images
def load_and_preprocess(paths, size=(32, 32)):
    data = []
    for path in paths:
        img = cv2.imread(path, 0)
        if img is None:
            continue
        img = cv2.resize(img, size) / 255.0
        data.append(img.flatten())
    return np.array(data)

# Redistribute outliers to nearest clusters or assign unique clusters
def redistribute_outliers(X_reduced, labels):
    outlier_mask = labels == -1
    if not np.any(outlier_mask):
        return labels

    non_outliers = X_reduced[~outlier_mask]
    non_outlier_labels = labels[~outlier_mask]
    outliers = X_reduced[outlier_mask]
    outlier_indices = np.where(outlier_mask)[0]

    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(non_outliers)
    distances, indices = knn.kneighbors(outliers)

    DISTANCE_THRESHOLD = 4.0
    close_enough = distances[:, 0] < DISTANCE_THRESHOLD
    valid_indices = outlier_indices[close_enough]

    neighbor_labels = non_outlier_labels[indices[close_enough]]
    majority_labels, _ = mode(neighbor_labels, axis=1, keepdims=True)

    final_labels = labels.copy()
    final_labels[valid_indices] = majority_labels.ravel()

    bad_indices = outlier_indices[~close_enough]
    current_max_cluster = final_labels.max() + 1

    for idx in bad_indices:
        final_labels[idx] = current_max_cluster
        current_max_cluster += 1

    return final_labels

# Main clustering pipeline
def run_clustering(txt_file, output_txt="clusters.txt"):
    image_paths = load_paths(txt_file)
    X = load_and_preprocess(image_paths)

    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=3, min_samples=5)

    labels = clusterer.fit_predict(X_reduced)

    labels = redistribute_outliers(X_reduced, labels)

    # Generate output text file 
    df = pd.DataFrame({"Path": image_paths, "Cluster": labels})
    df["Filename"] = df["Path"].apply(lambda x: os.path.basename(x))
    clusters = df.groupby("Cluster")["Filename"].apply(list).to_dict()

    with open(output_txt, "w", encoding="utf-8", newline="") as f:
        for cluster_id in sorted(clusters.keys()):
            filenames = sorted(clusters[cluster_id])
            f.write(" ".join(filenames) + "\n")

    print(f"Cluster file saved: {output_txt}")
    generate_html(df, "clusters.html")

# Generate HTML with images grouped by cluster
def generate_html(df, html_output):
    clusters = df.groupby("Cluster")["Path"].apply(list).to_dict()

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Character Clusters</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px; }
            .cluster { margin-bottom: 40px; background: white; padding: 20px; border-radius: 8px; }
            h2 { color: #333; }
            img { width: 40px; height: 40px; margin: 2px; border: 1px solid #ddd; }
            hr { border: 1px solid #ccc; }
        </style>
    </head>
    <body>
    """

    for cluster_id in sorted(clusters.keys()):
        html += f"<div class='cluster'>\n"
        html += f"<h2>Cluster {cluster_id}</h2>\n"
        for img_path in clusters[cluster_id]:
            rel_path = os.path.relpath(img_path, start=os.path.dirname(html_output))
            html += f"<img src='{rel_path}' title='{img_path}'>\n"
        html += "</div>\n<hr>"

    html += "</body></html>"
    with open(html_output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML file generated: {html_output}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python clustering_script.py <input.txt>")
        sys.exit(1)
    txt_file = sys.argv[1]
    run_clustering(txt_file)