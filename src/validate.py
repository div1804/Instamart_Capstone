
import pandas as pd
import sqlite3

conn = sqlite3.connect("data/instamart.db")

customers = pd.read_sql_query("SELECT * FROM customers", conn)
products = pd.read_sql_query("SELECT * FROM products", conn)
orders = pd.read_sql_query("SELECT * FROM orders", conn)

conn.close()

checks = []

checks.append(len(customers) == 1200)
checks.append(len(products) == 40)
checks.append(len(orders) == 12309)
checks.append(customers["customer_id"].is_unique)
checks.append(products["product_id"].is_unique)
checks.append(orders["order_id"].is_unique)
checks.append((orders["status"] == "pending").sum() == 0)
checks.append(orders["order_amount"].isna().sum() == 0)

completed = orders[orders["status"] == "completed"]

checks.append(completed["delivery_minutes"].isna().sum() == 0)
checks.append(round(completed["order_amount"].sum(), 2) == 2998668.80)

if all(checks):
    print("ALL CHECKPOINTS PASSED")
else:
    print("VALIDATION FAILED")
    for i, result in enumerate(checks, start=1):
        print(f"Checkpoint {i}: {result}")

import sqlite3

conn = sqlite3.connect("data/instamart.db")
cursor = conn.cursor()

checks = []

# 1. Customer count
cursor.execute("SELECT COUNT(*) FROM customers")
checks.append(cursor.fetchone()[0] == 1200)

# 2. Product count
cursor.execute("SELECT COUNT(*) FROM products")
checks.append(cursor.fetchone()[0] == 40)

# 3. Order count
cursor.execute("SELECT COUNT(*) FROM orders")
checks.append(cursor.fetchone()[0] == 12309)

# 4. Unique customer IDs
cursor.execute("""
SELECT COUNT(*) = COUNT(DISTINCT customer_id)
FROM customers
""")
checks.append(cursor.fetchone()[0] == 1)

# 5. Unique product IDs
cursor.execute("""
SELECT COUNT(*) = COUNT(DISTINCT product_id)
FROM products
""")
checks.append(cursor.fetchone()[0] == 1)

# 6. Unique order IDs
cursor.execute("""
SELECT COUNT(*) = COUNT(DISTINCT order_id)
FROM orders
""")
checks.append(cursor.fetchone()[0] == 1)

# 7. No pending orders
cursor.execute("""
SELECT COUNT(*) FROM orders
WHERE status = 'pending'
""")
checks.append(cursor.fetchone()[0] == 0)

# 8. No missing order amounts
cursor.execute("""
SELECT COUNT(*) FROM orders
WHERE order_amount IS NULL
""")
checks.append(cursor.fetchone()[0] == 0)

# 9. No missing delivery times for completed orders
cursor.execute("""
SELECT COUNT(*) FROM orders
WHERE status = 'completed'
AND delivery_minutes IS NULL
""")
checks.append(cursor.fetchone()[0] == 0)

# 10. Completed revenue
cursor.execute("""
SELECT ROUND(SUM(order_amount), 2)
FROM orders
WHERE status = 'completed'
""")
checks.append(cursor.fetchone()[0] == 2998668.80)

conn.close()

if all(checks):
    print("ALL CHECKPOINTS PASSED")
else:
    print("VALIDATION FAILED")
    print("Check results:", checks)
