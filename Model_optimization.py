import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Đọc dữ liệu
df = pd.read_csv('Iris.csv')
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Xử lý nhãn
X = df.drop('species', axis=1)
y = df['species']
le = LabelEncoder()
y = le.fit_transform(y)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Thiết lập lưới tham số
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

# Grid Search
mlp = MLPClassifier(max_iter=1000, random_state=42)
grid = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Kết quả
print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# Đánh giá mô hình tốt nhất
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Phân tích Overfitting/Underfitting theo Network Depth
depths = [(50,), (100,), (50, 50), (100, 50)]
train_scores = []
test_scores = []

for h in depths:
    clf = MLPClassifier(hidden_layer_sizes=h, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.plot(range(len(depths)), train_scores, label='Train Accuracy', marker='o')
plt.plot(range(len(depths)), test_scores, label='Test Accuracy', marker='x')
plt.xticks(range(len(depths)), ['(50,)', '(100,)', '(50,50)', '(100,50)'])
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy')
plt.title('Overfitting / Underfitting vs Network Depth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
