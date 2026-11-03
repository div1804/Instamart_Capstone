
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("INSTAMART PART B - LOGISTIC REGRESSION")

# Step 1: Load dataset
dataset = pd.read_csv("outputs/lapse_dataset.csv")

# Step 2: Separate features and target
X = dataset.drop(columns=["customer_id", "lapsed"])
y = dataset["lapsed"]

# Step 3: Identify feature types
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Step 5: Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Step 6: Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

# Step 7: Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Step 8: Create complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Step 9: Define hyperparameter grid
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

# Step 10: GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Step 11: Train model
grid_search.fit(X_train, y_train)

# Step 12: Best model
best_model = grid_search.best_estimator_

print("\nBEST LOGISTIC REGRESSION PARAMETERS")
print(grid_search.best_params_)

print("\nBEST CROSS-VALIDATION F1 SCORE")
print(round(grid_search.best_score_, 4))

# Step 13: Test predictions
y_pred = best_model.predict(X_test)

# Step 14: Evaluation
print("\nLOGISTIC REGRESSION TEST RESULTS")

print("Accuracy:",
      round(accuracy_score(y_test, y_pred), 4))

print("Precision:",
      round(precision_score(y_test, y_pred, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, y_pred, zero_division=0), 4))

print("F1 Score:",
      round(f1_score(y_test, y_pred, zero_division=0), 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))
# Step 15: Display Logistic Regression coefficients

feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

coefficients = best_model.named_steps["model"].coef_[0]

coefficient_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coefficient_table["Absolute_Value"] = coefficient_table["Coefficient"].abs()

coefficient_table = coefficient_table.sort_values(
    by="Absolute_Value",
    ascending=False
)

print("\nTOP 3 LOGISTIC REGRESSION COEFFICIENTS")

print(
    coefficient_table[
        ["Feature", "Coefficient"]
    ].head(3).to_string(index=False)
)
print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print("\nLOGISTIC REGRESSION COMPLETED!")

