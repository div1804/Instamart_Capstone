import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("INSTAMART PART B - CUSTOMER SEGMENT PROFILING")

# Step 1: Load RFM data
rfm = pd.read_csv("outputs/rfm_table.csv")

# Step 2: Select RFM features
rfm_features = rfm[["recency", "frequency", "monetary"]]

# Step 3: Standardise the features
scaler = StandardScaler()
scaled_rfm = scaler.fit_transform(rfm_features)

# Step 4: Apply K-Means with K=4
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(scaled_rfm)

# Step 5: Calculate segment profiles
segment_profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print("\nCUSTOMER SEGMENT PROFILES")
print(segment_profile)

# Step 6: Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_rfm)

print("\nPCA EXPLAINED VARIANCE")
print("Component 1:", round(pca.explained_variance_ratio_[0], 3))
print("Component 2:", round(pca.explained_variance_ratio_[1], 3))
print(
    "Total explained variance:",
    round(sum(pca.explained_variance_ratio_), 3)
)

# Step 7: Save profile
os.makedirs("outputs", exist_ok=True)

segment_profile.to_csv(
    "outputs/segment_profile.csv"
)

print("\nSEGMENT PROFILING COMPLETED!")
print("File: outputs/segment_profile.csv")