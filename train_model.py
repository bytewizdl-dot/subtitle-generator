import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Muat Data
df = pd.read_csv('predictive_maintenance.csv')

# 2. Buang (Drop) kolom yang tidak relevan untuk prediksi
df = df.drop(columns=['UDI', 'Product ID', 'Failure Type'])

# 3. Pisahkan Fitur (X) dan Target (y)
X = df.drop(columns=['Target'])
y = df['Target']

# 4. Kategorikan jenis kolom untuk pre-processing
numeric_features = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
categorical_features = ['Type']

# 5. Buat Transformer (Standarisasi angka & Encode teks L/M/H menjadi angka)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 6. Buat Pipeline Model dengan FINE-TUNING Random Forest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,          # Fine-tuning: Jumlah pohon
        max_depth=7,               # Fine-tuning: Mencegah overfitting
        class_weight='balanced',   # Fine-tuning: Menangani data tidak seimbang
        random_state=42
    ))
])

# 7. Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Latih Pipeline
pipeline.fit(X_train, y_train)

# 9. Evaluasi (Memastikan AUC >= 0.70 sesuai syarat tugas)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("--- Hasil Evaluasi ---")
print(classification_report(y_test, y_pred))
print(f"Nilai AUC: {roc_auc_score(y_test, y_proba):.3f}")

# 10. Simpan Pipeline dan Nama Kolom Input untuk Streamlit
joblib.dump(pipeline, "Manufact_model.pkl")

# Simpan nama kolom fitur mentah (sebelum di-transform) agar Streamlit tahu input apa saja yang dibutuhkan
input_features = X.columns.tolist()
joblib.dump(input_features, "feature_names.pkl")

print("\nBerhasil menyimpan 'Manufact_model.pkl' dan 'feature_names.pkl'")