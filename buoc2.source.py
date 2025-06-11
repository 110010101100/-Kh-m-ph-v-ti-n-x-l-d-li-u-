# Import các thư viện cần thiết cho bước 2 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Bước 2: Xây dựng mô hình cơ bản
print("\n=== Bước 2: Huấn luyện mô hình MLP ===")
X = data_scaled.drop('target', axis=1)
y = data_scaled['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    learning_rate='constant',
    max_iter=1000,  # tăng từ 200 lên 1000
    random_state=42
)
model.fit(X_train, y_train)
# Bước 3: Đánh giá mô hình cơ bản

print("\n=== Bước 3: Đánh giá mô hình MLP cơ bản ===")

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Báo cáo chi tiết các chỉ số
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Các chỉ số tổng quát
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))

# Vẽ confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - MLP Base Model")
plt.show()