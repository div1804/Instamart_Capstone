
import pandas as pd
import os

print("INSTAMART PART B - CUSTOMER FEATURES")

# Step 1: Load datasets
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Step 2: Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Step 3: Define cutoff date
cutoff_date = pd.Timestamp("2025-09-01")

# Step 4: Use only orders before cutoff
historical_orders = orders[
    orders["order_ts"] < cutoff_date
].copy()

# Step 5: Separate completed and cancelled orders
completed = historical_orders[
    historical_orders["status"] == "completed"
].copy()

cancelled = historical_orders[
    historical_orders["status"] == "cancelled"
].copy()

# Step 6: Calculate recency
last_order = completed.groupby("customer_id")["order_ts"].max()

features = pd.DataFrame({
    "customer_id": last_order.index
})

features["recency"] = (
    cutoff_date - pd.to_datetime(last_order).reset_index(drop=True)
).dt.days

# Step 7: Calculate frequency and monetary
rfm_features = completed.groupby("customer_id").agg(
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean")
).reset_index()

features = features.merge(
    rfm_features,
    on="customer_id",
    how="left"
)

# Step 8: Calculate cancellation rate
total_orders = historical_orders.groupby("customer_id").size()

cancelled_orders = cancelled.groupby("customer_id").size()

features["total_orders"] = features["customer_id"].map(total_orders)
features["cancelled_orders"] = (
    features["customer_id"].map(cancelled_orders).fillna(0)
)

features["cancellation_rate"] = (
    features["cancelled_orders"] / features["total_orders"]
)

features.drop(
    columns=["total_orders", "cancelled_orders"],
    inplace=True
)

# Step 9: Calculate favourite product category
completed_with_category = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_counts = completed_with_category.groupby(
    ["customer_id", "category"]
).size().reset_index(name="purchase_count")

favourite_category = category_counts.loc[
    category_counts.groupby("customer_id")["purchase_count"].idxmax(),
    ["customer_id", "category"]
]

favourite_category = favourite_category.rename(
    columns={"category": "favourite_category"}
)

features = features.merge(
    favourite_category,
    on="customer_id",
    how="left"
)

# Step 10: Add customer details
features = features.merge(
    customers[["customer_id", "city", "age", "gender", "signup_date"]],
    on="customer_id",
    how="left"
)

# Step 11: Calculate customer tenure
features["tenure_days"] = (
    cutoff_date - features["signup_date"]
).dt.days

features.drop(columns=["signup_date"], inplace=True)

# Step 12: Check feature data
print("\nFEATURE DATASET SHAPE")
print(features.shape)

print("\nFIRST 10 CUSTOMERS")
print(features.head(10).to_string(index=False))

print("\nMISSING VALUES")
print(features.isnull().sum())

# Step 13: Save features
os.makedirs("outputs", exist_ok=True)

features.to_csv(
    "outputs/lapse_features.csv",
    index=False
)

print("\nCUSTOMER FEATURES CREATED SUCCESSFULLY!")
print("File: outputs/lapse_features.csv")
