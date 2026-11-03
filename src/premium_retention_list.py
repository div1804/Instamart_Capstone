import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# 1. Load datasets
dataset = pd.read_csv("outputs/lapse_dataset.csv")
segments = pd.read_csv("outputs/customer_segments.csv")

# 2. Recreate the same train-test split
X = dataset.drop(columns=["customer_id", "lapsed"])
y = dataset["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. Preserve test customer IDs
test_indices = X_test.index

test_customer_ids = dataset.loc[
    test_indices, "customer_id"
].reset_index(drop=True)

# 4. Load the saved model
model = joblib.load("outputs/lapse_model.joblib")

# 5. Predict lapse probabilities
lapse_probabilities = model.predict_proba(X_test)[:, 1]

# 6. Create prediction results
retention_data = pd.DataFrame({
    "customer_id": test_customer_ids,
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.reset_index(drop=True)
})

# 7. Identify Premium customers
premium_customers = segments[
    segments["is_premium"] == True
]["customer_id"]

retention_data = retention_data[
    retention_data["customer_id"].isin(premium_customers)
]

# 8. Rank customers by lapse probability
retention_data = retention_data.sort_values(
    by="lapse_probability",
    ascending=False
)

# 9. Select Top 10 customers
top_10 = retention_data.head(10).copy()

# 10. Count genuinely lapsed customers
actually_lapsed_count = top_10["actually_lapsed"].sum()

# 11. Save retention list
top_10.to_csv(
    "outputs/premium_retention_list.csv",
    index=False
)

# 12. Display results
print("PREMIUM CUSTOMER RETENTION LIST")
print("=" * 60)

print(top_10.to_string(index=False))

print("\nTotal Premium Customers in Test Set:",
      len(retention_data))

print("Top 10 Customers:", len(top_10))

print("Actually Lapsed Customers in Top 10:",
      actually_lapsed_count)

print("\nRetention list saved successfully!")
print("File: outputs/premium_retention_list.csv")