import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Load dataset
dataset = pd.read_csv("outputs/lapse_dataset.csv")

X = dataset.drop(columns=["customer_id", "lapsed"])
y = dataset["lapsed"]

# 2. Define features
numeric_features = [
    "recency", "frequency", "monetary",
    "avg_discount", "cancellation_rate",
    "avg_delivery", "age", "tenure_days"
]

categorical_features = [
    "city", "gender", "favourite_category"
]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("ORIGINAL TRAINING CLASS COUNTS")
print(y_train.value_counts())

# 4. Separate training classes
train_data = X_train.copy()
train_data["lapsed"] = y_train

majority = train_data[train_data["lapsed"] == 0]
minority = train_data[train_data["lapsed"] == 1]

# 5. Oversample the lapsed class
minority_oversampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

# 6. Combine original and additional records
balanced_train = pd.concat(
    [majority, minority_oversampled]
)

balanced_train = balanced_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_train_balanced = balanced_train.drop(columns=["lapsed"])
y_train_balanced = balanced_train["lapsed"]

print("\nBALANCED TRAINING CLASS COUNTS")
print(y_train_balanced.value_counts())

# 7. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 8. Selected Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42
    ))
])

# 9. Train using balanced training data
model.fit(X_train_balanced, y_train_balanced)

# 10. Evaluate on original test data
y_pred = model.predict(X_test)

print("\nOVERSAMPLED RANDOM FOREST RESULTS")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\nOVERSAMPLING COMPLETED!")