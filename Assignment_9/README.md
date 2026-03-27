# Experiment 9: Clustering Human Activity Recognition Data

**Name:** Monesh M  
**Roll No:** 3122235001084

## Objective
To implement and analyze the performance of clustering algorithms (**K-Means**, **DBSCAN**, and **Hierarchical Agglomerative Clustering**) on the Human Activity Recognition (HAR) dataset.

## Dataset Overview
- **Dataset:** Human Activity Recognition Using Smartphones (UCI HAR)
- **Samples:** 10,299 (Train + Test combined)
- **Features:** 561 (Time and frequency domain variables)
- **Activities:** 6 (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying)

## Implementation Details
1. **Preprocessing:** Data was standardized using `StandardScaler`.
2. **K-Means:** Analyzed using Elbow Method and Silhouette Score ($k=2$ to $8$). Final model used $k=6$.
3. **DBSCAN:** Applied with `eps=15` and `min_samples=10`.
4. **Hierarchical:** Ward's linkage was used to generate the dendrogram and form 6 clusters.
5. **Visualization:** PCA was used to project the 561D features into 2D for cluster visualization.

## Performance Metrics

| Algorithm | Silhouette Score | ARI | NMI |
|-----------|------------------|-----|-----|
| K-Means | 0.110 | 0.420 | 0.559 |
| DBSCAN | 0.119 | 0.080 | 0.154 |
| Hierarchical | 0.117 | 0.460 | 0.601 |

## Key Findings
- **Best Performer:** Hierarchical Clustering (Ward's Linkage) yielded the highest Adjusted Rand Index (0.46) and Normalized Mutual Info (0.60).
- **Inertia vs Clusters:** The Elbow method showed a significant drop in WCSS up to $k=6$, aligning with the actual number of activities.
- **Challenge:** Distinguishing between stationary activities (Sitting vs. Standing) remains a challenge for unsupervised algorithms.

## Visualizations
All plots are located in `images/PNG/` and `images/EPS/`.
- `activity_distribution.png`: Frequency of each activity in the dataset.
- `pca_ground_truth.png`: PCA projection with true labels.
- `kmeans_analysis.png`: Elbow and Silhouette plots for K-Means.
- `hierarchical_dendrogram.png`: Dendrogram for HAC.
- `metrics_comparison.png`: Comparison of ARI and Silhouette scores.
- `kmeans_clusters_pca.png`: Final K-Means cluster assignments in 2D space.
