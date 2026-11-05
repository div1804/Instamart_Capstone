
import os
import sqlite3
import pandas as pd

print("INSTAMART DATA CLEANING STARTED")

# Create required folders
os.makedirs("data/clean", exist_ok=True)

# Load raw datasets
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

print("\nRaw datasets loaded successfully")
print("Customers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)


# --------------------------------
# 1. CLEAN CUSTOMER DATA
# --------------------------------

customers_clean = customers.copy()

customers_clean["city"] = (
    customers_clean["city"].str.strip().str.title()
)

customers_clean["gender"] = (
    customers_clean["gender"].astype("string").str.strip().str.lower()
)

customers_clean["gender"] = customers_clean["gender"].replace({
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
    "": "unknown"
}).fillna("unknown")

customers_clean["age"] = pd.to_numeric(
    customers_clean["age"], errors="coerce"
)

customers_clean.loc[
    ~customers_clean["age"].between(10, 100), "age"
] = float("nan")

customers_clean["age"] = customers_clean.groupby(
    "city"
)["age"].transform(
    lambda values: values.fillna(values.median())
)

customers_clean["signup_date"] = pd.to_datetime(
    customers_clean["signup_date"], errors="coerce"
)

customers_clean.to_csv(
    "data/clean/customers.csv", index=False
)

print("\nCustomer cleaning completed")


# --------------------------------
# 2. CLEAN PRODUCT DATA
# --------------------------------

products_clean = products.copy()

products_clean.to_csv(
    "data/clean/products.csv", index=False
)

print("Product data saved")


# --------------------------------
# 3. CLEAN ORDER DATA
# --------------------------------

orders_clean = orders.copy()

orders_clean["order_ts"] = pd.to_datetime(
    orders_clean["order_ts"], errors="coerce"
)

# Remove exact duplicate rows
orders_clean = orders_clean.drop_duplicates()

# Keep the latest record for repeated order IDs
orders_clean = orders_clean.sort_values(by="order_ts")

orders_clean = orders_clean.drop_duplicates(
    subset="order_id", keep="last"
).reset_index(drop=True)

# Convert order amount to numeric
orders_clean["order_amount"] = pd.to_numeric(
    orders_clean["order_amount"], errors="coerce"
)

# Fill missing completed-order amounts
price_map = products_clean.set_index(
    "product_id"
)["price_inr"]

missing_mask = orders_clean["order_amount"].isna()

orders_clean.loc[missing_mask, "order_amount"] = (
    orders_clean.loc[missing_mask, "product_id"].map(price_map)
    * orders_clean.loc[missing_mask, "quantity"]
    * (
        1 - orders_clean.loc[missing_mask, "discount_pct"] / 100
    )
).round(2)

# Replace negative delivery times with missing values
orders_clean.loc[
    orders_clean["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# Add product category temporarily
category_map = products_clean.set_index(
    "product_id"
)["category"]

orders_clean["category"] = orders_clean[
    "product_id"
].map(category_map)

# Fill missing delivery times for completed orders
completed_mask = orders_clean["status"] == "completed"

category_medians = (
    orders_clean.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders_clean.loc[completed_mask, "delivery_minutes"] = (
    orders_clean.loc[completed_mask, "delivery_minutes"]
    .fillna(category_medians)
)

# Remove temporary category column
orders_clean = orders_clean.drop(columns=["category"])

# Save cleaned orders
orders_clean.to_csv(
    "data/clean/orders.csv", index=False
)

print("Order cleaning completed")


# --------------------------------
# 4. CREATE SQLITE DATABASE
# --------------------------------

conn = sqlite3.connect("data/instamart.db")

orders_db = orders_clean.copy()

orders_db["order_ts"] = (
    orders_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
)

customers_clean.to_sql(
    "customers", conn, if_exists="replace", index=False
)

products_clean.to_sql(
    "products", conn, if_exists="replace", index=False
)

orders_db.to_sql(
    "orders", conn, if_exists="replace", index=False
)

conn.close()

print("\nSQLite database created successfully")


# --------------------------------
# 5. FINAL SUMMARY
# --------------------------------

print("\nFINAL CLEANING SUMMARY")
print("----------------------")
print("Customers:", len(customers_clean))
print("Products:", len(products_clean))
print("Orders:", len(orders_clean))
print(
    "Completed orders:",
    (orders_clean["status"] == "completed").sum()
)
print(
    "Cancelled orders:",
    (orders_clean["status"] == "cancelled").sum()
)
print(
    "Missing completed-order amounts:",
    orders_clean.loc[
        orders_clean["status"] == "completed",
        "order_amount"
    ].isna().sum()
)

print("\nCLEANING SCRIPT COMPLETED SUCCESSFULLY!")
