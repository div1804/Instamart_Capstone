
import pandas as pd
import os

print("INSTAMART PART B - LAPSE PREDICTION")
print("STEP 1: BUILD LAPSE COHORT")

# Step 1: Load cleaned orders
orders = pd.read_csv("data/clean/orders.csv")

# Convert order timestamp into datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Step 2: Define cutoff and prediction window
cutoff_date = pd.Timestamp("2025-09-01")
window_end = pd.Timestamp("2025-10-01")

# Step 3: Select completed orders before cutoff
historical_orders = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] < cutoff_date)
]

# Step 4: Select completed orders during September
september_orders = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= cutoff_date) &
    (orders["order_ts"] < window_end)
]

# Step 5: Identify eligible customers
cohort = pd.DataFrame({
    "customer_id": historical_orders["customer_id"].unique()
})

# Step 6: Identify customers who purchased in September
september_customer_ids = september_orders["customer_id"].unique()

# Step 7: Create lapse target label
cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_customer_ids)
).astype(int)

# Step 8: Display results
print("\nLAPSE COHORT SUMMARY")
print("Total eligible customers:", len(cohort))

print("\nLAPSED / NOT LAPSED COUNTS")
print(cohort["lapsed"].value_counts().sort_index())

print("\nLAPSED CUSTOMER PERCENTAGE")
print(round(cohort["lapsed"].mean() * 100, 2), "%")

# Step 9: Save cohort
os.makedirs("outputs", exist_ok=True)

cohort.to_csv(
    "outputs/lapse_cohort.csv",
    index=False
)

print("\nFIRST 10 CUSTOMERS")
print(cohort.head(10).to_string(index=False))

print("\nLAPSE COHORT CREATED SUCCESSFULLY!")
print("File: outputs/lapse_cohort.csv")
