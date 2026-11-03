import pandas as pd
import os

print("INSTAMART PART B - RFM ANALYSIS")

# Load cleaned orders
orders = pd.read_csv("data/clean/orders.csv")

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Keep completed orders only
completed_orders = orders[
    orders["status"] == "completed"
].copy()

# Snapshot moment
snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

# Find the latest completed order for each customer
last_order = (
    completed_orders
    .groupby("customer_id")["order_ts"]
    .max()
)

# Calculate Recency
recency = (snapshot_date - last_order).dt.days

# Calculate Frequency and Monetary
rfm = (
    completed_orders
    .groupby("customer_id")
    .agg(
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    )
)

# Combine RFM values
rfm["recency"] = recency

# Arrange columns
rfm = rfm.reset_index()

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Round monetary values
rfm["monetary"] = rfm["monetary"].round(2)

# Display results
print("\nFIRST 10 CUSTOMERS - RFM TABLE")
print(rfm.head(10))

print("\nRFM SUMMARY")
print(rfm[["recency", "frequency", "monetary"]].describe())

print("\nRFM VALIDATION")
print("Total customers:", len(rfm))
print("Recency median:", rfm["recency"].median())
print("Recency minimum:", rfm["recency"].min())
print("Recency maximum:", rfm["recency"].max())
print("Frequency mean:", round(rfm["frequency"].mean(), 2))
print("Frequency maximum:", rfm["frequency"].max())
print("Monetary mean:", round(rfm["monetary"].mean(), 2))
print("Monetary maximum:", rfm["monetary"].max())

# Save RFM table
os.makedirs("outputs", exist_ok=True)

rfm.to_csv("outputs/rfm_table.csv", index=False)

print("\nRFM TABLE SAVED SUCCESSFULLY!")
print("File: outputs/rfm_table.csv")