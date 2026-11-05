import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("INSTAMART PART B - SAVING CUSTOMER SEGMENTS")

# Step 1: Load RFM data
rfm = pd.read_csv("outputs/rfm_table.csv")

# Step 2: Standardise RFM features
features = rfm[["recency", "frequency", "monetary"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Apply K-Means
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(scaled_features)

# Step 4: Assign business names
segment_names = {
    0: "Regular Low-Spend Customers",
    1: "Premium High-Value Customers",
    2: "Loyal Frequent Customers",
    3: "At-Risk Inactive Customers"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Step 5: Identify Premium customers
rfm["is_premium"] = (
    rfm["segment_id"] == 1
).astype(int)

# Step 6: Select required columns
customer_segments = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

# Step 7: Save output
os.makedirs("outputs", exist_ok=True)

customer_segments.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

print("\nFIRST 10 CUSTOMER SEGMENTS")
print(customer_segments.head(10).to_string(index=False))

print("\nSEGMENT COUNTS")
print(customer_segments["segment_name"].value_counts())

print("\nPREMIUM CUSTOMER COUNT")
print(customer_segments["is_premium"].sum())

print("\nCUSTOMER SEGMENTS SAVED SUCCESSFULLY!")
print("File: outputs/customer_segments.csv")