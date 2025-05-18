import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score
import pickle
import os

# Load data dataset
df = pd.read_excel("data/HARGA_RUMAH_JAKSEL.xlsx", header=1)
df.dropna(how='all', inplace=True)  # hapus row kosong

# Preprocessing
# Encode GRS (ada/tidak)
df['GRS'] = df['GRS'].map({'ada':1, 'tidak':0})

# Features input
target = 'HARGA'
features = ['LT','LB','JKT','JKM','GRS']

# Scaling features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Classify "HARGA" using quartile
df['kelas'] = pd.qcut(df[target], q=3, labels=['MURAH','SEDANG','MAHAL'])
le_kelas = LabelEncoder()
df['kelas_enc'] = le_kelas.fit_transform(df['kelas'])

# Split data
X = df[features]
y_reg = df[target]
y_clf = df['kelas_enc']
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# Train models
# Regression model
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_reg_train)

# Classification model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_clf_train)

# Evalutation metrics
print("--- Regression Evaluation ---")
y_reg_pred = reg.predict(X_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2_score(y_reg_test, y_reg_pred):.2f}")

print("\n--- Classification Report ---")
y_clf_pred = clf.predict(X_test)
y_clf_proba = clf.predict_proba(X_test)
print(classification_report(y_clf_test, y_clf_pred, target_names=le_kelas.classes_))
auc = roc_auc_score(y_clf_test, y_clf_proba, multi_class='ovr')
print(f"AUC-ROC: {auc:.2f}")

# Save models & label_encoder
os.makedirs('models', exist_ok=True)
with open('models/regression_model.pkl','wb') as f:
    pickle.dump(reg, f)
with open('models/classification_model.pkl','wb') as f:
    pickle.dump(clf, f)
with open('models/label_encoder.pkl','wb') as f:
    pickle.dump((scaler, le_kelas), f)