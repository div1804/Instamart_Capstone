
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

print("INSTAMART PART B - DATA PREPROCESSING")

# Step 1: Load dataset
dataset = pd.read_csv("outputs/lapse_dataset.csv")

# Step 2: Separate features and target
X = dataset.drop(
    columns=["customer_id", "lapsed"]
)

y = dataset["lapsed"]

# Step 3: Identify numerical and categorical columns
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

# Step 5: Create numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Step 6: Create categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

# Step 7: Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Step 8: Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Step 9: Transform testing data
X_test_processed = preprocessor.transform(X_test)

# Step 10: Display results
print("\nTRAINING PROCESSED SHAPE")
print(X_train_processed.shape)

print("\nTESTING PROCESSED SHAPE")
print(X_test_processed.shape)

# Step 11: Check missing values
print("\nMISSING VALUES IN ORIGINAL FEATURES")
print(X.isnull().sum())

print("\nCHECK PROCESSED DATA")
print("Training missing values:",
      pd.isna(X_train_processed).sum())

print("Testing missing values:",
      pd.isna(X_test_processed).sum())

# Step 12: Display frequency scaling information
frequency_index = numeric_features.index("frequency")

frequency_scaler = (
    preprocessor.named_transformers_["numeric"]
    .named_steps["scaler"]
)

print("\nFREQUENCY SCALER MEAN")
print(frequency_scaler.mean_[frequency_index])

print("\nFULL COHORT FREQUENCY MEAN")
print(X["frequency"].mean())

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")