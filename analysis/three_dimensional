import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
"""
Outlier detection using LOF method

In this implementation outliers are defined
as points that are strongly deviating from 
the "normal" trends and appear in considerably
low-density regions compared to their local neighbors

The dataset is standardized before applying LOF
"""

df = pd.read_csv("data/outlier2.csv")

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

lof = LocalOutlierFactor(n_neighbors=10)
labels = lof.fit_predict(df_scaled)

df["status"] = labels
non_o = df[df["status"] == 1]
out = df[df["status"] == -1]

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], color = "black")
ax1.set_title("Original Data")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(non_o.iloc[:, 0], non_o.iloc[:, 1], non_o.iloc[:, 2], color ="black")
ax2.scatter(out.iloc[:, 0], out.iloc[:, 1], out.iloc[:, 2], color = "red")
ax2.set_title("Outliers Detection")

for a in [ax1, ax2]:
    a.set_xlabel("Feature 1")
    a.set_ylabel("Feature 2")
    a.set_zlabel("Feature 3")

plt.show()
