import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer

file_name = input("Enter the dataset filename (e.g., diabetes.csv or heart.csv): ")
target_col = input("Enter the target column name (e.g., Outcome or HeartDisease): ")

data = pd.read_csv(file_name)
print(f"\n✅ Loaded {file_name} successfully!\n")
print("First 5 rows:\n", data.head(), "\n")

X = data.drop(target_col, axis=1)
y = data[target_col]

X = pd.get_dummies(X, drop_first=True)

imputer = SimpleImputer(strategy="mean") 
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("⚙️ Model Evaluation Results:")
print("------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("------------------------------------\n")

choice = input("Do you want to predict a new case? (yes/no): ").strip().lower()
if choice == "yes":
    print(f"\nEnter {len(X.columns)} values in order:")
    values = []
    for col in X.columns:
        val = float(input(f"{col}: "))
        values.append(val)

    sample_df = pd.DataFrame([values], columns=X.columns)
    sample_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_scaled)[0]
    print("\n🔍 Prediction Result:")
    print("Input Data:\n", sample_df)
    print("\nPredicted Outcome:", "Positive (1)" if prediction == 1 else "Negative (0)")
