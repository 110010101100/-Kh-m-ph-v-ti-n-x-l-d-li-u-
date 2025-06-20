# %%
# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# %%
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target  # Thêm cột nhãn (0: Setosa, 1: Versicolor, 2: Virginica)





print("=== Bước 1.1: Kiểm tra dữ liệu ban đầu ===")
print("Thông tin cơ bản về dữ liệu:")
print(data.head())  # Hiển thị 5 dòng đầu
print("\nSố lượng mẫu và đặc trưng:", data.shape)  # (150, 5)
print("\nTên các lớp:", iris.target_names)  # ['setosa', 'versicolor', 'virginica']
# Kiểm tra outlier bằng IQR (Interquartile Range)
Q1 = data.drop('target', axis=1).quantile(0.25)
Q3 = data.drop('target', axis=1).quantile(0.75)
IQR = Q3 - Q1
outliers = ((data.drop('target', axis=1) < (Q1 - 1.5 * IQR)) | (data.drop('target', axis=1) > (Q3 + 1.5 * IQR))).sum()
print("\nSố lượng outlier trên mỗi đặc trưng:")
print(outliers)


# %%
print("\n=== Bước 1.2: Phân tích dữ liệu ===")
print("Thông tin cấu trúc dữ liệu:")
print(data.info())  # Kiểu dữ liệu, số hàng/cột
print("\nThống kê mô tả (mean, std, min, max, ...):")
print(data.describe())  # Mean, std, min, max, percentile
print("\nKiểm tra giá trị null:")
print(data.isnull().sum())  # Kiểm tra số lượng giá trị null
print("\nPhân bố các lớp (target):")
print(data['target'].value_counts())  # Đếm số lượng mỗi lớp

# Kiểm tra outlier bằng IQR (Interquartile Range)
Q1 = data.drop('target', axis=1).quantile(0.25)
Q3 = data.drop('target', axis=1).quantile(0.75)
IQR = Q3 - Q1
outliers = ((data.drop('target', axis=1) < (Q1 - 1.5 * IQR)) | (data.drop('target', axis=1) > (Q3 + 1.5 * IQR))).sum()
print("\nSố lượng outlier trên mỗi đặc trưng:")
print(outliers)

# %%
print("\n=== Bước 1.3: Tiền xử lý dữ liệu ===")
scaler = StandardScaler()  # Sử dụng StandardScaler
X = data.drop('target', axis=1)  # Lấy các đặc trưng
X_scaled = scaler.fit_transform(X)  # Chuẩn hóa dữ liệu
data_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)  # Chuyển về DataFrame
data_scaled['target'] = data['target']  # Thêm lại cột nhãn
print("Dữ liệu sau khi chuẩn hóa (mẫu đầu tiên):")
print(data_scaled.head())
print("\nThống kê sau chuẩn hóa:")
print(data_scaled.describe())  # Kiểm tra mean ~0, std ~1

# Kiểm tra mean và std sau chuẩn hóa
print("\nKiểm tra mean và std sau chuẩn hóa (đảm bảo mean ~0, std ~1):")
stats = data_scaled.drop('target', axis=1).describe()
print(stats)
print("\nMean của các đặc trưng:", stats.loc['mean'].values)
print("Std của các đặc trưng:", stats.loc['std'].values)


# Lưu scaler và dữ liệu chuẩn hóa
joblib.dump(scaler, 'scaler.pkl')
print("Scaler đã được lưu vào 'scaler.pkl'")
data_scaled.to_csv('iris_scaled.csv', index=False)
print("Dữ liệu chuẩn hóa đã được lưu vào 'iris_scaled.csv'")

# Bước 1.4: Vẽ Correlation Heatmap
print("\n=== Bước 1.4: Vẽ Correlation Heatmap ===")
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Iris Dataset")
plt.show()


# %%
print("\n=== Bước 1.5: Vẽ Histograms ===")
data.hist(figsize=(10, 8), bins=20)
plt.suptitle("Histograms of Iris Features")
plt.tight_layout()
plt.show()

# Bước 1.6: Vẽ Boxplot 
print("\n=== Bước 1.6: Vẽ Boxplot (bổ sung) ===")
data.drop('target', axis=1).boxplot(figsize=(10, 6))
plt.title("Boxplot of Iris Features")
plt.show()

# %%
print("\n=== Bước 1.5: Vẽ Histograms ===")
data.hist(figsize=(10, 8), bins=20)
plt.suptitle("Histograms of Iris Features")
plt.tight_layout()
plt.show()

# Bước 1.6: Vẽ Boxplot 
print("\n=== Bước 1.6: Vẽ Boxplot (bổ sung) ===")
data.drop('target', axis=1).boxplot(figsize=(10, 6))
plt.title("Boxplot of Iris Features")
plt.show()

# %%
print("\n=== Bước 1.7: Quan sát và phân tích biểu đồ ===")
print("""
Hướng dẫn quan sát và phân tích biểu đồ:
1. **Correlation Heatmap**:
   - petal length và petal width có tương quan cao (~0.96), là đặc trưng quan trọng.
   - sepal length và sepal width có tương quan thấp (~0.12).
2. **Histograms**:
   - petal length và petal width phân biệt rõ Setosa với các loài khác.
   - sepal length và sepal width có phân bố chồng lấn.
3. **Boxplot**:
   - sepal width có vài outlier, các đặc trưng khác ổn định.
""")

# %%
# Import các thư viện cần thiết cho bước 2 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
print("\n=== Bước 2.1: Chuẩn bị dữ liệu ===")
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop('target', axis=1))
data_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)
data_scaled['target'] = data['target']

# %%
print("\n=== Bước 2.2: Phân chia dữ liệu Train-Test ===")
X = data_scaled.drop('target', axis=1)
y = data_scaled['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# %%
print("\n=== Bước 2.3: Xây dựng Baseline model ===")
X_original = data.drop('target', axis=1)
y_original = data['target']

X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42, stratify=y_original
)

def baseline_rule(petal_length):
    if petal_length < 2.5:
        return 0
    elif petal_length < 5.0:
        return 1
    else:
        return 2

petal_length_train = X_train_orig.iloc[:, 2].values

manual_predictions = [
    baseline_rule(petal_length) for petal_length in petal_length_train
]
manual_accuracy = np.mean(manual_predictions == y_train_orig.values)

print(f"Baseline model accuracy: {manual_accuracy * 100:.2f}%")

# %%
print("\n=== Bước 2.4: Huấn luyện MLPClassifier ===")
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    learning_rate='constant',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, 'iris_mlp_model.pkl')

print(f"Loss sau huấn luyện: {model.loss_:.4f}")

# %%
print("\n=== Bước 3.1: Load dữ liệu và model ===")
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop('target', axis=1))
data_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)
data_scaled['target'] = data['target']

X = data_scaled.drop('target', axis=1)
y = data_scaled['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load('iris_mlp_model.pkl')

print("\n=== Bước 3.2: Dự đoán ===")
y_pred = model.predict(X_test)
print("y_pred:", y_pred)

# %%
print("=== Bước 3.3: Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# %%
print("=== Bước 3.4: Metrics ===")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision (macro): {precision * 100:.2f}%")
print(f"Recall (macro): {recall * 100:.2f}%")
print(f"F1-score (macro): {f1 * 100:.2f}%")

# %%
print("\n=== Bước 3.5: Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - MLP Base Model")
plt.show()


#=============================== Buoc 4 ===============================


# step4_mlp_gridsearch.py

print("=== Bước 4.1: Kiểm tra và tạo dữ liệu đầu vào ===")
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

if not os.path.exists("iris_scaled.csv"):
    print("📂 Chưa có file iris_scaled.csv – Đang tạo từ bộ Iris gốc...")
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.drop('target', axis=1))
    data_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)
    data_scaled['target'] = data['target']

    data_scaled.to_csv('iris_scaled.csv', index=False)
    print("✅ Đã lưu iris_scaled.csv (dữ liệu đã chuẩn hóa).")
else:
    print("✅ Đã tìm thấy iris_scaled.csv.")

print("📈 Xem thử 5 dòng đầu:")
print(pd.read_csv("iris_scaled.csv").head())


print("\n=== Bước 4.2: Cấu hình pipeline và tham số tìm kiếm ===")
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("iris_scaled.csv")
X = df.drop("target", axis=1)
y = df["target"]

print(f"🔢 Dữ liệu huấn luyện: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=1000, random_state=42))
])

param_grid = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'mlp__activation': ['relu', 'tanh', 'logistic'],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate_init': [0.001, 0.01],
}

total_combinations = (
    len(param_grid['mlp__hidden_layer_sizes']) *
    len(param_grid['mlp__activation']) *
    len(param_grid['mlp__solver']) *
    len(param_grid['mlp__alpha']) *
    len(param_grid['mlp__learning_rate_init'])
)
print(f"🔧 Tổng số tổ hợp siêu tham số: {total_combinations}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


print("\n=== Bước 4.3: Tiến hành huấn luyện bằng GridSearchCV ===")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

print("⏳ Bắt đầu huấn luyện... (mỗi dấu 'Fitting' là một tổ hợp tham số)")
grid_search.fit(X, y)
print("✅ Huấn luyện hoàn tất.")


print("\n=== Bước 4.4: Lưu mô hình và xem kết quả ===")
joblib.dump(grid_search, "mlp_gridsearch_model.pkl")
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv("mlp_gridsearch_results.csv", index=False)
print("💾 Đã lưu mô hình và kết quả vào 'mlp_gridsearch_model.pkl' và 'mlp_gridsearch_results.csv'.")

print("\n🎯 Tham số tốt nhất tìm được:")
print(grid_search.best_params_)
print(f"🏆 Độ chính xác cao nhất (cross-validated): {grid_search.best_score_ * 100:.2f}%")

print("\n🔝 Top 3 cấu hình tốt nhất:")
top3 = results_df.sort_values(by='mean_test_score', ascending=False).head(3)
print(top3[['mean_test_score', 'params']])
