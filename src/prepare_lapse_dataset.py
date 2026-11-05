
import pandas as pd
import os

print("INSTAMART PART B - PREPARE LAPSE DATASET")

# Step 1: Load both datasets
features = pd.read_csv("outputs/lapse_features.csv")
cohort = pd.read_csv("outputs/lapse_cohort.csv")

# Step 2: Merge customer features with target labels
dataset = features.merge(
    cohort,
    on="customer_id",
    how="inner"
)

# Step 3: Check dataset
print("\nFINAL DATASET SHAPE")
print(dataset.shape)

print("\nFIRST 10 ROWS")
print(dataset.head(10).to_string(index=False))

# Step 4: Check target distribution
print("\nLAPSE TARGET DISTRIBUTION")
print(dataset["lapsed"].value_counts().sort_index())

print("\nLAPSE PERCENTAGE")
print(round(dataset["lapsed"].mean() * 100, 2), "%")

# Step 5: Check missing values
print("\nMISSING VALUES")
print(dataset.isnull().sum())

# Step 6: Save final dataset
os.makedirs("outputs", exist_ok=True)

dataset.to_csv(
    "outputs/lapse_dataset.csv",
    index=False
)

print("\nFINAL LAPSE DATASET SAVED SUCCESSFULLY!")
print("File: outputs/lapse_dataset.csv")
