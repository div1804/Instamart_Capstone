
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("INSTAMART PART B - BASELINE MODEL")

# Step 1: Load dataset
dataset = pd.read_csv("outputs/lapse_dataset.csv")

# Step 2: Separate target
X = dataset.drop(columns=["customer_id", "lapsed"])
y = dataset["lapsed"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Step 4: Create baseline model
baseline = DummyClassifier(strategy="most_frequent")

# Step 5: Train model
baseline.fit(X_train, y_train)

# Step 6: Predict
y_pred = baseline.predict(X_test)

# Step 7: Evaluate
print("\nBASELINE ACCURACY")
print(round(accuracy_score(y_test, y_pred), 4))

print("\nLAPSED CUSTOMER F1 SCORE")
print(round(f1_score(y_test, y_pred, zero_division=0), 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nBASELINE MODEL COMPLETED!")
