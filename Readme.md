# Character Clustering Tool

## What It Does
This script groups similar character images—letters, numbers, fragments, or combined symbols—into clusters. It uses PCA for dimensionality reduction and HDBSCAN to find groups, then handles outliers by reassigning or giving them their own clusters.

## How It Works
1. **Preprocessing**: Convert images to grayscale, resize to 32×32 pixels, and normalize pixel values.  
2. **Feature Extraction**: Flatten each image into a 1,024-length vector.  
3. **Dimensionality Reduction**: Apply PCA to keep 95% of the variance and reduce noise.  
4. **Clustering**: Run HDBSCAN with `min_cluster_size=5`, `min_samples=5`, and `cluster_selection_epsilon=3`.  
5. **Outlier Handling**: For points labeled as outliers, check the closest cluster using KNN; if it’s within a distance of 4.0, add them there. Otherwise, assign each one to a separate new cluster.

## Method Details
This script uses **PCA (Principal Component Analysis)** to reduce dimensionality while preserving 95% of the data variance.  
Clustering is done using **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**.
- HDBSCAN automatically determines the number of clusters.
- It uses the **Euclidean distance metric** to evaluate similarity.
- It's particularly useful for datasets with clusters of varying density.
- Outliers detected by HDBSCAN (label `-1`) are handled using **K-Nearest Neighbors (KNN)** and a threshold of **4.0**.

## Expected Duration
The script takes approximately 20–30 seconds.

## How to Run
1. Clone or unpack the project folder.
2. Run the setup script to create and activate a virtual environment and install dependencies:
   ```bash
   ./setup.sh
   ```
3. Activate the virtual environment and run the clustering script:
   ```bash
   source venv/bin/activate
   python cluster_chars.py input.txt
   ```

## Files Included
- `cluster_chars.py`: Main script.  
- `input.txt`: List of image paths, one per line.  
- `clusters.txt`: Output—each line lists filenames in the same cluster, separated by spaces.  
- `clusters.html`: Simple HTML view showing each cluster, separated by `<HR>` tags.  
- `setup.sh`: Sets up a Python virtual environment and installs required packages.  
- `README.md`: This file.

## Custom Options
- **Image size**: Change `size=(32,32)` in `load_and_preprocess`.
- **PCA variance**: Adjust `n_components` in the PCA call (default is 0.95).
- **HDBSCAN settings**: Modify `min_cluster_size`, `min_samples`, or `cluster_selection_epsilon` in `run_clustering`.
- **Distance threshold**: Tweak `DISTANCE_THRESHOLD` in `redistribute_outliers` (default is 4.0).
