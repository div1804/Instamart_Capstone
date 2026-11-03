import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# 1. Load dataset
dataset = pd.read_csv("outputs/lapse_dataset.csv")

X = dataset.drop(columns=["customer_id", "lapsed"])
y = dataset["lapsed"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. Prepare preprocessing
numeric_features = [
    "recency", "frequency", "monetary",
    "avg_discount", "cancellation_rate",
    "avg_delivery", "age", "tenure_days"
]

categorical_features = [
    "city", "gender", "favourite_category"
]

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

# 4. Create selected Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42
    ))
])

# 5. Train final selected model on original training data
model.fit(X_train, y_train)

# 7. Generate predictions before saving
predictions_before = model.predict(X_test)

# 8. Save model
joblib.dump(model, "outputs/lapse_model.joblib")

print("\nModel saved successfully!")

# 9. Reload model
loaded_model = joblib.load("outputs/lapse_model.joblib")

print("Model reloaded successfully!")

# 10. Verify predictions
predictions_after = loaded_model.predict(X_test)

if np.array_equal(predictions_before, predictions_after):
    print("\nSUCCESS: Predictions match exactly!")
else:
    print("\nERROR: Predictions do not match!")

print("\nFINAL MODEL SAVING COMPLETED!")