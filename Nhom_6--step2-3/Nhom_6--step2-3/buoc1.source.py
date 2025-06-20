# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Bước 1.1: Tải và kiểm tra dữ liệu
# Tải Iris Dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target  # Thêm cột nhãn (0: Setosa, 1: Versicolor, 2: Virginica)

# Kiểm tra dữ liệu ban đầu
print("=== Bước 1.1: Kiểm tra dữ liệu ban đầu ===")
print("Thông tin cơ bản về dữ liệu:")
print(data.head())  # Hiển thị 5 dòng đầu
print("\nSố lượng mẫu và đặc trưng:", data.shape)  # (150, 5)
print("\nTên các lớp:", iris.target_names)  # ['setosa', 'versicolor', 'virginica']

# Bước 1.2: Phân tích dữ liệu
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

# Bước 1.3: Tiền xử lý dữ liệu
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

# Bước 1.5: Vẽ Histograms
print("\n=== Bước 1.5: Vẽ Histograms ===")
data.hist(figsize=(10, 8), bins=20)
plt.suptitle("Histograms of Iris Features")
plt.tight_layout()
plt.show()

# Bước 1.6: Vẽ Boxplot (bổ sung)
print("\n=== Bước 1.6: Vẽ Boxplot (bổ sung) ===")
data.drop('target', axis=1).boxplot(figsize=(10, 6))
plt.title("Boxplot of Iris Features")
plt.show()