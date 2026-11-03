import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("INSTAMART PART B - K-MEANS ANALYSIS")

# Load RFM data
rfm = pd.read_csv("outputs/rfm_table.csv")

# Select RFM features
rfm_features = rfm[["recency", "frequency", "monetary"]]

# Standardise the features
scaler = StandardScaler()
scaled_rfm = scaler.fit_transform(rfm_features)

print("\nRFM FEATURES STANDARDISED SUCCESSFULLY!")

# Test K values from 2 to 8
results = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(scaled_rfm)

    inertia = model.inertia_
    silhouette = silhouette_score(scaled_rfm, labels)

    results.append({
        "K": k,
        "Inertia": round(inertia, 2),
        "Silhouette Score": round(silhouette, 3)
    })

# Create results table
results_df = pd.DataFrame(results)

print("\nK-MEANS COMPARISON")
print(results_df.to_string(index=False))

# Save results
os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/kmeans_metrics.csv",
    index=False
)

print("\nK-MEANS ANALYSIS COMPLETED!")
print("File: outputs/kmeans_metrics.csv")