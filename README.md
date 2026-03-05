# 3D Outlier Detection with Local Outlier Factor (LOF)

This project detects **outliers in a three-dimensional dataset** using the **Local Outlier Factor (LOF)** algorithm from scikit-learn.

LOF identifies anomalies as points that **deviate strongly from their local neighborhood** and lie in **low-density regions** compared to nearby points.  
Before applying LOF, the dataset is **standardized** to ensure all features contribute equally to distance-based calculations.

---

## Method

### 1) Standardization
The features are scaled using `StandardScaler`:

- centers each feature to mean ≈ 0  
- scales each feature to standard deviation ≈ 1


### 2) Local Outlier Factor (LOF)
LOF is applied with:

- `n_neighbors=10`

The model assigns a label to each point:

- `1`  → inlier (normal point)
- `-1` → outlier (anomalous point)

---

## Output

<img width="1200" height="600" alt="three_dimensional" src="https://github.com/user-attachments/assets/9aad6169-2d82-44d7-b695-65e3cad399fa" />

The script generates two 3D scatter plots:

1. **Original Data** (all points in black)
2. **Outliers Detection** (inliers in black, detected outliers in red)


The black points form several clusters that reflect the main structure of the dataset. These clusters represent regions where data points have similar characteristics and lie close to their neighboring observations in the feature space. In contrast, points that appear far from these dense regions are detected as outliers because they significantly differ from the local data distribution.

---
