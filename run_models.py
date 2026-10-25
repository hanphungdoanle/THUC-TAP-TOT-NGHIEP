import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thư viện Sklearn (Tiền xử lý & ML)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

# Thư viện XGBoost
from xgboost import XGBClassifier

# Thư viện TensorFlow / Keras (Deep Learning)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# =========================================================
# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# =========================================================
print("1. Đang tải và tiền xử lý dữ liệu...")
csv_path = 'data.csv'

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Không tìm thấy file {csv_path}. Vui lòng đặt file dữ liệu cùng thư mục với code.")

# Sử dụng sep=';' và encoding='utf-8-sig' 
df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig', on_bad_lines='skip')

# Xóa các dòng rỗng hoàn toàn
df.dropna(how='all', inplace=True)

# Chỉ giữ lại các dòng có kết quả 'Passed' hoặc 'Failed'
valid_results = ['Passed', 'Failed']
df_clean = df[df['Test Result'].isin(valid_results)].copy()

# Chuyển đổi nhãn Test Result: Passed -> 1, Failed -> 0
df_clean['Target'] = df_clean['Test Result'].map({'Passed': 1, 'Failed': 0})

# Điền các giá trị NaN ở cột đặc trưng để không bị lỗi lúc fit mô hình
df_clean['Features'] = df_clean['Features'].fillna('Unknown')
df_clean['Title'] = df_clean['Title'].fillna('Unknown')
df_clean['Build'] = df_clean['Build'].fillna('Unknown')

# Tách biến độc lập (X) và biến phụ thuộc (y)
X = df_clean[['Features', 'Title', 'Build']]
y = df_clean['Target']

print(f"-> Số lượng mẫu sau khi làm sạch: {len(X)} (Passed: {sum(y==1)}, Failed: {sum(y==0)})")

# Chia tập Train / Test (Tỉ lệ 80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline Tiền xử lý:
# - Title (Văn bản): Dùng TF-IDF
# - Features, Build (Danh mục): Dùng OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=150, min_df=2), 'Title'),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Features', 'Build'])
    ]
)

# Fit và Transform dữ liệu
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# =========================================================
# 2. HÀM ĐÁNH GIÁ MÔ HÌNH VÀ LƯU KẾT QUẢ
# =========================================================
results_dict = {}
probas_dict = {}
preds_dict = {}

def evaluate_and_store(model_name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    
    results_dict[model_name] = {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4)
    }
    probas_dict[model_name] = y_proba
    preds_dict[model_name] = y_pred

# =========================================================
# 3. HUẤN LUYỆN 3 MÔ HÌNH HỌC MÁY (MACHINE LEARNING)
# =========================================================
print("2. Đang huấn luyện các mô hình Machine Learning...")

# --- Mô hình 1: Logistic Regression (Baseline) ---
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_processed, y_train)
y_pred_lr = lr_model.predict(X_test_processed)
y_proba_lr = lr_model.predict_proba(X_test_processed)[:, 1]
evaluate_and_store('Logistic Regression', y_test, y_pred_lr, y_proba_lr)

# --- Mô hình 2: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_processed, y_train)
y_pred_rf = rf_model.predict(X_test_processed)
y_proba_rf = rf_model.predict_proba(X_test_processed)[:, 1]
evaluate_and_store('Random Forest', y_test, y_pred_rf, y_proba_rf)

# --- Mô hình 3: XGBoost ---
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_processed, y_train)
y_pred_xgb = xgb_model.predict(X_test_processed)
y_proba_xgb = xgb_model.predict_proba(X_test_processed)[:, 1]
evaluate_and_store('XGBoost', y_test, y_pred_xgb, y_proba_xgb)

# =========================================================
# 4. HUẤN LUYỆN 1 MÔ HÌNH HỌC SÂU (DEEP LEARNING - MLP)
# =========================================================
print("3. Đang huấn luyện mô hình Deep Learning (MLP)...")
tf.random.set_seed(42)
input_dim = X_train_processed.shape[1]

dl_model = Sequential([
    Dense(64, activation='relu', input_dim=input_dim),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # Dùng sigmoid cho phân loại nhị phân
])

dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = dl_model.fit(
    X_train_processed, y_train,
    epochs=50, batch_size=8,
    validation_split=0.2, verbose=0
)

# Predict
y_proba_dl = dl_model.predict(X_test_processed).ravel()
y_pred_dl = (y_proba_dl >= 0.5).astype(int)
evaluate_and_store('Deep Learning (MLP)', y_test, y_pred_dl, y_proba_dl)

# =========================================================
# 5. XUẤT BẢNG KẾT QUẢ VÀ LƯU HÌNH ẢNH BÁO CÁO
# =========================================================
print("4. Đang lưu kết quả và biểu đồ...")

# Xuất bảng kết quả ra CSV
df_results = pd.DataFrame(results_dict).T
print("\n=== BẢNG TỔNG HỢP KẾT QUẢ MÔ HÌNH ===")
print(df_results)
df_results.to_csv('bang_ket_qua.csv', encoding='utf-8-sig')

# --- Hình 1: Ma trận nhầm lẫn (Confusion Matrices) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
models = list(results_dict.keys())

for idx, model_name in enumerate(models):
    ax = axes[idx // 2, idx % 2]
    cm = confusion_matrix(y_test, preds_dict[model_name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Predict Fail', 'Predict Pass'],
                yticklabels=['Actual Fail', 'Actual Pass'])
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrix của 4 Mô hình', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('bieu_do_confusion_matrix.png', dpi=300)
plt.close()

# --- Hình 2: Đường cong ROC Curve ---
plt.figure(figsize=(8, 6))
for model_name in models:
    fpr, tpr, _ = roc_curve(y_test, probas_dict[model_name])
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results_dict[model_name]['ROC-AUC']})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('Đường cong ROC giữa các mô hình', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('bieu_do_roc_curve.png', dpi=300)
plt.close()

# --- Hình 3: Tiến trình huấn luyện Học sâu ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Deep Learning Loss', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_title('Deep Learning Accuracy', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('bieu_do_deep_learning_history.png', dpi=300)
plt.close()

print("\n[HOÀN TẤT] File csv kết quả và 3 file ảnh biểu đồ (.png) đã được lưu thành công")
# =========================================================
# 6. DEMO THỰC TẾ: DỰ ĐOÁN 1 TEST CASE MỚI CHO THẦY XEM
# =========================================================
print("\n" + "="*60)
print("DEMO: DỰ ĐOÁN KẾT QUẢ CHO 1 TEST CASE ĐỘC LẬP BẰNG 4 MÔ HÌNH")
print("="*60)

# 1. Nhập thông tin 1 test case hoàn toàn mới
test_case_moi = pd.DataFrame({
    'Features': ['Upload & Validate COMTRADE Files'],
    'Title': ['Verify that system rejects invalid .txt file formats'],
    'Build': ['v.0.1.1']
})

print("Thông tin Test Case đầu vào:")
print(test_case_moi.to_string(index=False))

# 2. Tiền xử lý test case sang dạng số học 
test_case_processed = preprocessor.transform(test_case_moi)

# 3. Yêu cầu cả 4 mô hình cùng đưa ra phán đoán
print("\nKẾT QUẢ DỰ ĐOÁN TỪ CÁC MÔ HÌNH:")

# Logistic Regression
proba_lr = lr_model.predict_proba(test_case_processed)[0][1]
print(f"- Logistic Regression : Xác suất Pass = {proba_lr*100:5.2f}% -> Kết luận: [{'Passed' if proba_lr >= 0.5 else 'Failed'}]")

# Random Forest
proba_rf = rf_model.predict_proba(test_case_processed)[0][1]
print(f"- Random Forest       : Xác suất Pass = {proba_rf*100:5.2f}% -> Kết luận: [{'Passed' if proba_rf >= 0.5 else 'Failed'}]")

# XGBoost
proba_xgb = xgb_model.predict_proba(test_case_processed)[0][1]
print(f"- XGBoost             : Xác suất Pass = {proba_xgb*100:5.2f}% -> Kết luận: [{'Passed' if proba_xgb >= 0.5 else 'Failed'}]")

# Deep Learning
proba_dl = dl_model.predict(test_case_processed, verbose=0).ravel()[0]
print(f"- Deep Learning (MLP) : Xác suất Pass = {proba_dl*100:5.2f}% -> Kết luận: [{'Passed' if proba_dl >= 0.5 else 'Failed'}]")
print("="*60)