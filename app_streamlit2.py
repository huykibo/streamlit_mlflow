import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

# ---------------------------
# 1️⃣ Mô tả Ứng dụng
# ---------------------------
st.title("🚢 Dự đoán sống sót trên Titanic")
st.sidebar.title("Chọn tính năng")
page = st.sidebar.radio("", ["📖 Xem thông tin", "📊 Khám phá dữ liệu", "⚙️ Xử lý dữ liệu", "🚀 Huấn luyện mô hình"], index=0)

# ---------------------------
# 2️⃣ Tải dữ liệu
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# ---------------------------
# 3️⃣ Tiền xử lý dữ liệu
# ---------------------------
def preprocess_data(df):
    df = df.copy()
    # Loại bỏ các cột không cần thiết
    dropped_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=dropped_cols, errors='ignore', inplace=True)
    
    # Log MLflow cho bước loại bỏ cột
    mlflow.end_run()
    with mlflow.start_run():
        mlflow.log_param("dropped_columns", dropped_cols)
    
    # Mã hóa các biến phân loại bằng LabelEncoder
    for col in ["Sex", "Embarked"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        with mlflow.start_run(nested=True):
            mlflow.log_param(f"label_encoder_{col}", list(le.classes_))
    
    # Điền giá trị thiếu: Age với trung vị, Embarked với mode
    df.fillna({"Age": df["Age"].median(), "Embarked": df["Embarked"].mode()[0]}, inplace=True)
    with mlflow.start_run(nested=True):
        mlflow.log_param("missing_values_filled", {"Age": "median", "Embarked": "mode"})
    
    # Chuẩn hóa dữ liệu: chỉ với các cột số Age và Fare
    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
    with mlflow.start_run(nested=True):
        mlflow.log_param("scaled_columns", ["Age", "Fare"])
    
    return df

# ---------------------------
# 4️⃣ Chia tập dữ liệu (70/15/15)
# ---------------------------
def split_data(df):
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    # Chia Train (70%) và Tạm (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_temp, y_train, y_temp

# ---------------------------
# 5️⃣ Huấn luyện mô hình với MLflow và Cross Validation
# ---------------------------
def train_model_with_mlflow(X_train, y_train, X_val, y_val):
    param_grid = [10, 50, 100, 200]
    best_acc = 0
    best_model = None
    best_n_estimators = None
    acc_scores = []
    
    mlflow.end_run()
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")
        for n in param_grid:
            with mlflow.start_run(nested=True):
                model = RandomForestClassifier(n_estimators=n, random_state=42)
                model.fit(X_train, y_train)
                # Sử dụng Cross Validation 5-fold trên tập Validation
                acc = cross_val_score(model, X_val, y_val, cv=5).mean()
                acc_scores.append(acc)
                mlflow.log_param("n_estimators", n)
                mlflow.log_metric("accuracy", acc)
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_n_estimators = n
        mlflow.log_param("best_n_estimators", best_n_estimators)
        mlflow.log_metric("best_accuracy", best_acc)
    
    return best_model, best_acc, best_n_estimators, param_grid, acc_scores

# ---------------------------
# 6️⃣ Vẽ biểu đồ hiệu suất mô hình
# ---------------------------
def plot_model_performance(param_grid, acc_scores):
    fig, ax = plt.subplots()
    sns.lineplot(x=param_grid, y=acc_scores, marker="o", ax=ax)
    ax.set_xlabel("Số lượng cây (n_estimators)")
    ax.set_ylabel("Độ chính xác (accuracy)")
    ax.set_title("Đánh giá số lượng cây tốt nhất cho Random Forest")
    st.pyplot(fig)

# ---------------------------
# 7️⃣ Đánh giá mô hình
# ---------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    st.subheader("📊 Báo cáo đánh giá mô hình")
    st.write(""" 
    🔹 **Accuracy Score:** Độ chính xác tổng thể của mô hình  
    🔹 **Classification Report:** Hiển thị Precision, Recall, F1-score cho từng lớp  
    🔹 **Confusion Matrix:** Ma trận nhầm lẫn giúp kiểm tra lỗi phân loại  
    """)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    st.text(classification_report(y_test, y_pred))
    
    st.write("### 🔍 **Giải thích kết quả:**")
    st.write(f"🔹 **Accuracy (Độ chính xác):** {report['accuracy']:.2f} - Mô hình dự đoán đúng {report['accuracy']*100:.0f}% tổng số mẫu.")
    st.write(f"🔹 **Precision cho lớp 0 (Không sống):** {report['0']['precision']:.2f}, lớp 1 (Sống): {report['1']['precision']:.2f}.")
    st.write(f"🔹 **Recall cho lớp 0 (Không sống):** {report['0']['recall']:.2f}, lớp 1 (Sống): {report['1']['recall']:.2f}.")
    st.write(f"🔹 **F1-Score cho lớp 0 (Không sống):** {report['0']['f1-score']:.2f}, lớp 1 (Sống): {report['1']['f1-score']:.2f}.")
    st.write("Mô hình có thể có hiệu suất khác nhau giữa các lớp, cần cân nhắc thêm về cân bằng dữ liệu hoặc điều chỉnh hyperparameter.")
    
    cm = confusion_matrix(y_test, y_pred)
    st.write("📊 **Ma trận nhầm lẫn:**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Không sống", "Sống"], yticklabels=["Không sống", "Sống"])
    st.pyplot(fig)
    
    st.write("### 🔍 **Giải thích ma trận nhầm lẫn:**")
    st.write(f"🔹 **True Negatives (TN):** {cm[0, 0]} - Mô hình dự đoán đúng trường hợp không sống.")
    st.write(f"🔹 **False Positives (FP):** {cm[0, 1]} - Mô hình dự đoán sai là sống nhưng thực tế không sống.")
    st.write(f"🔹 **False Negatives (FN):** {cm[1, 0]} - Mô hình dự đoán sai là không sống nhưng thực tế sống.")
    st.write(f"🔹 **True Positives (TP):** {cm[1, 1]} - Mô hình dự đoán đúng trường hợp sống.")

# ---------------------------
# 8️⃣ Giao diện Streamlit
# ---------------------------
if page == "📖 Xem thông tin":
    st.subheader("📖 Thông tin ứng dụng")
    st.write(""" 
    Ứng dụng này hỗ trợ phân tích và dự đoán khả năng sống sót trên tàu Titanic bằng cách sử dụng mô hình học máy **Random Forest**.
    
    Các bước thực hiện:
    - Khám phá dữ liệu
    - Tiền xử lý dữ liệu (loại bỏ cột, mã hóa, điền giá trị thiếu, chuẩn hóa) – được log qua MLflow.
    - Chia tập dữ liệu theo tỷ lệ 70/15/15 (Train/Validation/Test).
    - Huấn luyện mô hình Random Forest qua Cross Validation để lựa chọn số cây tối ưu.
    - Đánh giá mô hình với báo cáo phân loại và ma trận nhầm lẫn.
    """)
    
elif page == "📊 Khám phá dữ liệu":
    st.subheader("📊 Khám phá dữ liệu gốc")
    st.write("Bấm nút dưới đây để hiển thị 10 dòng đầu của tập dữ liệu Titanic.")
    
    if st.button("🔍 Hiển thị dữ liệu gốc"):
        st.write(df.head(10))
    
    if st.button("📊 Kiểm tra dữ liệu"):
        st.write("⚠️ **Giá trị thiếu:**")
        st.write(df.isnull().sum()[df.isnull().sum() > 0])
        st.write("📌 **Kiểu dữ liệu:**")
        st.write(df.dtypes)

elif page == "⚙️ Xử lý dữ liệu":
    st.subheader("⚙️ Xử lý dữ liệu")
    st.write("Nhấn nút dưới đây để thực hiện tiền xử lý dữ liệu (loại bỏ cột, mã hóa, điền giá trị thiếu và chuẩn hóa) và log quá trình vào MLflow.")
    
    if st.button("⚙️ Bắt đầu xử lý dữ liệu"):
        df_clean = preprocess_data(df)
        st.success("✅ Dữ liệu đã được xử lý thành công!")
        st.success("✅ Đã log dữ liệu thành công vào MLflow!")
        st.subheader("📌 Các bước xử lý:")
        st.write(""" 
        - Loại bỏ các cột không cần thiết: PassengerId, Name, Ticket, Cabin.
        - Mã hóa biến phân loại: Sex, Embarked.
        - Điền giá trị thiếu: Age (trung vị), Embarked (mode).
        - Chuẩn hóa dữ liệu: Age, Fare với StandardScaler.
        """)
        st.write(df_clean.head(10))

elif page == "🚀 Huấn luyện mô hình":
    st.subheader("🚀 Huấn luyện mô hình")
    # Tiền xử lý dữ liệu
    df_clean = preprocess_data(df)
    # Chia dữ liệu thành Train (70%) và Tạm (30%)
    X_train, X_temp, y_train, y_temp = split_data(df_clean)
    # Chia X_temp thành Validation và Test (mỗi 15%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    st.write("📊 **Chia tập dữ liệu theo tỷ lệ 70/15/15:**")
    st.write(f"🔹 **Tập huấn luyện (Train):** {X_train.shape[0]} mẫu")
    st.write(f"🔹 **Tập validation (Validation):** {X_val.shape[0]} mẫu")
    st.write(f"🔹 **Tập kiểm tra (Test):** {X_test.shape[0]} mẫu")
    
    st.write("Nhấn nút bên dưới để huấn luyện mô hình Random Forest qua Cross Validation với các giá trị `n_estimators` khác nhau.")
    
    if st.button("🚀 Huấn luyện mô hình"):
        model, best_acc, best_n_estimators, param_grid, acc_scores = train_model_with_mlflow(X_train, y_train, X_val, y_val)
        st.success(f"✅ Huấn luyện hoàn thành! Độ chính xác cao nhất: {best_acc:.2f} với {best_n_estimators} cây.")
        st.success("✅ Đã log mô hình huấn luyện thành công vào MLflow!")
        
        st.subheader("🔍 Hiệu suất mô hình")
        plot_model_performance(param_grid, acc_scores)
        
        st.subheader("⚙️ Đánh giá mô hình trên tập kiểm tra")
        evaluate_model(model, X_test, y_test)
