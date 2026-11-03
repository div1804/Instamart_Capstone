
import pandas as pd

from sklearn.model_selection import train_test_split

print("INSTAMART PART B - TRAIN TEST SPLIT")

# Step 1: Load dataset
dataset = pd.read_csv("outputs/lapse_dataset.csv")

# Step 2: Separate features and target
X = dataset.drop(
    columns=["customer_id", "lapsed"]
)

y = dataset["lapsed"]

# Step 3: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Step 4: Display results
print("\nTRAINING DATA SHAPE")
print(X_train.shape)

print("\nTESTING DATA SHAPE")
print(X_test.shape)

print("\nTRAINING TARGET DISTRIBUTION")
print(y_train.value_counts().sort_index())

print("\nTESTING TARGET DISTRIBUTION")
print(y_test.value_counts().sort_index())

print("\nTOTAL TRAINING ROWS:", len(X_train))
print("TOTAL TESTING ROWS:", len(X_test))

print("\nTRAIN TEST SPLIT COMPLETED!")
