import pandas as pd
import numpy as np
import os

print("INSTAMART PART B - CUSTOMER SEGMENTATION")

# Load cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

print("\nCUSTOMER DATA")
print("Shape:", customers.shape)
print(customers.head())

print("\nORDER DATA")
print("Shape:", orders.shape)
print(orders.head())

# Check order date range
print("\nORDER DATE RANGE")
print("Minimum date:", orders["order_ts"].min())
print("Maximum date:", orders["order_ts"].max())

# Check order status
print("\nORDER STATUS COUNTS")
print(orders["status"].value_counts())

# Check customer IDs
print("\nUNIQUE CUSTOMERS IN ORDERS:")
print(orders["customer_id"].nunique())

print("\nDATA INSPECTION COMPLETED!")