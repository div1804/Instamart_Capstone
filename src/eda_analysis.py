
import sqlite3
import pandas as pd

print("INSTAMART EXPLORATORY DATA ANALYSIS")

# Connect to database
conn = sqlite3.connect("data/instamart.db")

# Load completed orders
completed_orders = pd.read_sql_query("""
SELECT *
FROM orders
WHERE status = 'completed'
""", conn)

customers = pd.read_sql_query(
    "SELECT * FROM customers", conn
)

products = pd.read_sql_query(
    "SELECT * FROM products", conn
)

conn.close()

# --------------------------------
# 1. CITY AND CATEGORY ANALYSIS
# --------------------------------

merged_data = (
    completed_orders
    .merge(customers[["customer_id", "city"]], on="customer_id")
    .merge(products[["product_id", "category"]], on="product_id")
)

pivot_result = pd.pivot_table(
    merged_data,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True
)

print("\nPANDAS PIVOT TABLE")
print(pivot_result)

print("\nGrand Total:", round(
    pivot_result.loc["All", "All"], 2
))

# --------------------------------
# 2. CITY-WISE ANALYSIS
# --------------------------------

city_analysis = (
    merged_data
    .groupby("city")
    .agg(
        total_revenue=("order_amount", "sum"),
        average_order_amount=("order_amount", "mean"),
        order_count=("order_amount", "count")
    )
    .round(2)
)

print("\nCITY-WISE ANALYSIS")
print(city_analysis)

# --------------------------------
# 3. STATISTICAL ANALYSIS
# --------------------------------

amounts = completed_orders["order_amount"]

mean_value = amounts.mean()
median_value = amounts.median()
skew_value = amounts.skew()

q1 = amounts.quantile(0.25)
q2 = amounts.quantile(0.50)
q3 = amounts.quantile(0.75)

iqr = q3 - q1

lower_fence = q1 - (1.5 * iqr)
upper_fence = q3 + (1.5 * iqr)

lower_outliers = (amounts < lower_fence).sum()
upper_outliers = (amounts > upper_fence).sum()

print("\nSTATISTICAL ANALYSIS")
print("Mean:", round(mean_value, 2))
print("Median:", round(median_value, 2))
print("Skewness:", round(skew_value, 2))
print("Q1:", round(q1, 2))
print("Q2:", round(q2, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Lower outliers:", lower_outliers)
print("Upper outliers:", upper_outliers)

# --------------------------------
# 4. CORRELATION ANALYSIS
# --------------------------------

correlation_matrix = completed_orders[
    [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes"
    ]
].corr()

print("\nCORRELATION MATRIX")
print(correlation_matrix.round(3))

corr_pairs = (
    correlation_matrix
    .where(
        ~__import__("numpy").eye(
            len(correlation_matrix),
            dtype=bool
        )
    )
    .stack()
)

strongest_pair = corr_pairs.abs().idxmax()

print("\nStrongest correlation pair:", strongest_pair)
print(
    "Correlation value:",
    round(
        correlation_matrix.loc[
            strongest_pair[0], strongest_pair[1]
        ],
        3
    )
)

print("\nEDA ANALYSIS COMPLETED SUCCESSFULLY!")
