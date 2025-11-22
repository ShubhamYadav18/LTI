import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import models, layers

# 1. Load Datasets
df_purchase = pd.read_csv("User_product_purchase_details_p2.csv")
df_user = pd.read_csv("user_demographics.csv")

df = pd.merge(df_purchase, df_user, on="User_ID", how="left")



# 2. Create Target Column
df["High_Value_Purchase"] = (df["Purchase"] >= 10000).astype(int)

df = df.fillna(0)


if "Product_ID" in df.columns:
    df = df.drop("Product_ID", axis=1)


# 3. Encode Categorical Columns (Label Encoding)
le = LabelEncoder()

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col].astype(str))


# 4. Train-Test Split
X = df.drop(["High_Value_Purchase", "Purchase"], axis=1)
y = df["High_Value_Purchase"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 5. Scale Numeric Columns
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6. Logistic Regression
print("\n---------------- Logistic Regression ----------------")
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, pred_lr))
print(confusion_matrix(y_test, pred_lr))
print(classification_report(y_test, pred_lr))


# 7. Random Forest
print("\n---------------- Random Forest ----------------")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
pred_rf = rf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, pred_rf))
print(confusion_matrix(y_test, pred_rf))
print(classification_report(y_test, pred_rf))


# 8. XGBoost

print("\n---------------- XGBoost ----------------")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    eval_metric="logloss"
)
xgb_model.fit(X_train_scaled, y_train)
pred_xgb = xgb_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, pred_xgb))
print(confusion_matrix(y_test, pred_xgb))
print(classification_report(y_test, pred_xgb))



# 9. Keras MLP
print("\n---------------- Keras MLP ----------------")

input_dim = X_train_scaled.shape[1]

model = models.Sequential([
    layers.Dense(128, activation="relu", input_shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["precision"])

model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=1)

pred_mlp_prob = model.predict(X_test_scaled).flatten()
pred_mlp = (pred_mlp_prob >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, pred_mlp))
print(confusion_matrix(y_test, pred_mlp))
print(classification_report(y_test, pred_mlp))


