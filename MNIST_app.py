import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # Dùng SVC cho SVM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from PIL import Image
import tempfile

# ========== PHẦN QUAN TRỌNG: LẤY THÔNG TIN TỪ STREAMLIT SECRETS ==========
os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]

mlflow.set_tracking_uri(st.secrets["mlflow"]["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("MNIST")

# ========== CẤU HÌNH STREAMLIT ==========
st.set_page_config(page_title="MNIST App với Streamlit", layout="wide")
st.title("Ứng dụng Phân loại Chữ số MNIST")

# CSS cho tooltip (không thay đổi)
st.markdown("""
    <style>
    .tooltip {
      position: relative;
      display: inline-block;
      cursor: pointer;
      color: #1f77b4;
      font-weight: bold;
    }
    .tooltip .tooltiptext {
      visibility: hidden;
      width: 320px;
      background-color: #f9f9f9;
      color: #333;
      text-align: left;
      border-radius: 6px;
      padding: 8px;
      position: absolute;
      z-index: 1;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      margin-top: 5px;
      opacity: 0;
      transition: opacity 0.3s;
      border: 1px solid #ccc;
      font-size: 0.85em;
      line-height: 1.3;
    }
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Tạo các tab cho ứng dụng
tabs = st.tabs([
    "Thông tin",
    "Tải dữ liệu",
    "Xử lí dữ liệu",
    "Chia dữ liệu",
    "Huấn luyện/Đánh giá",
    "Demo dự đoán",
    "Thông tin Huấn luyện"
])
tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

# ----------------- TAB 1: THÔNG TIN -----------------
with tab_info:
    st.header("Thông tin Ứng dụng")
    st.markdown("""
**Giới thiệu ứng dụng:**
Ứng dụng **Phân loại Chữ số MNIST với MLflow Tracking trên DagsHub** mô phỏng quy trình của một dự án Machine Learning. 
Các chức năng chính gồm:
- **Tải dữ liệu:** Lấy dữ liệu MNIST từ OpenML hoặc upload file CSV (có cột 'label').
- **Xử lí dữ liệu:** Thực hiện normalization, standardization và missing imputation.
- **Chia dữ liệu:** Phân chia dữ liệu thành tập huấn luyện, validation, test.
- **Huấn luyện & Đánh giá:** Huấn luyện mô hình (Decision Tree hoặc SVM) và log các thông số qua MLflow.
- **Demo dự đoán:** Dự đoán mẫu từ tập Test hoặc upload ảnh mới.
- **Thông tin Huấn luyện:** Xem thông tin run và mở giao diện MLflow trên DagsHub.
""", unsafe_allow_html=True)

# ----------------- TAB 2: TẢI DỮ LIỆU -----------------
with tab_load:
    st.header("Tải Dữ liệu")
    st.markdown("""
        <div class="tooltip">
            <span style="font-weight:bold;">OpenML?</span>
            <span class="tooltiptext">
                OpenML là nền tảng mở cho Machine Learning, cung cấp nhiều dataset chất lượng để huấn luyện mô hình.
            </span>
        </div>
        """, unsafe_allow_html=True)
    data_option = st.radio("Chọn nguồn dữ liệu:", ("Tải từ OpenML", "Upload dữ liệu"))
    if data_option == "Tải từ OpenML":
        if st.button("Tải dữ liệu MNIST từ OpenML"):
            with st.spinner("Đang tải dữ liệu từ OpenML..."):
                mnist = openml.datasets.get_dataset(554)
                X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
                st.success("Tải dữ liệu thành công!")
                st.write("Số lượng mẫu:", X.shape[0])
                st.session_state['data'] = (X, y)
    else:
        uploaded_file = st.file_uploader("Upload file CSV", type="csv")
        if uploaded_file is not None:
            with st.spinner("Đang tải dữ liệu từ file..."):
                data = pd.read_csv(uploaded_file)
                st.write("Dữ liệu mẫu:", data.head())
                if "label" in data.columns:
                    X = data.drop("label", axis=1)
                    y = data["label"]
                    st.session_state['data'] = (X, y)
                else:
                    st.error("File cần có cột 'label'.")

# ----------------- TAB 3: XỬ LÍ DỮ LIỆU -----------------
with tab_preprocess:
    st.header("Xử lí Dữ liệu")
    if 'data' in st.session_state:
        X, y = st.session_state['data']
        if "data_original" not in st.session_state:
            st.session_state["data_original"] = (X, y)
        st.subheader("Dữ liệu Gốc")
        st.write(X.head())
        st.markdown("### Các thao tác xử lí dữ liệu")
        
        # Normalization
        with st.container():
            col_norm_btn, col_norm_tip = st.columns([0.95, 0.05])
            with col_norm_btn:
                if st.button("Chuẩn hoá (Normalization)"):
                    with st.spinner("Đang chuẩn hoá dữ liệu..."):
                        X_norm = st.session_state["data_original"][0] / 255.0
                        st.session_state["data_norm"] = (X_norm, y)
                        st.success("Đã chuẩn hoá dữ liệu!")
            with col_norm_tip:
                st.markdown("""
                    <div class="tooltip" style="margin-left:-880px;">
                        <span>?</span>
                        <span class="tooltiptext">
                            Chia giá trị pixel cho 255 để chuyển từ [0, 255] về [0, 1], giúp chuẩn hóa dữ liệu đầu vào cho mô hình.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            if "data_norm" in st.session_state:
                st.write("**Kết quả Chuẩn hoá:**")
                st.write(st.session_state["data_norm"][0].head())
        
        # Standardization
        with st.container():
            col_std_btn, col_std_tip = st.columns([0.95, 0.05])
            with col_std_btn:
                if st.button("Standardization"):
                    with st.spinner("Đang thực hiện Standardization..."):
                        X_std = (st.session_state["data_original"][0] - st.session_state["data_original"][0].mean()) / st.session_state["data_original"][0].std()
                        st.session_state["data_std"] = (X_std, y)
                        st.success("Đã thực hiện Standardization!")
            with col_std_tip:
                st.markdown("""
                    <div class="tooltip" style="margin-left:-880px;">
                        <span>?</span>
                        <span class="tooltiptext">
                            Chuyển dữ liệu sao cho trung bình = 0 và độ lệch chuẩn = 1, giúp cho quá trình huấn luyện mô hình hiệu quả hơn.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            if "data_std" in st.session_state:
                st.write("**Kết quả Standardization:**")
                st.write(st.session_state["data_std"][0].head())
        
        # Missing Imputation
        with st.container():
            col_imp_btn, col_imp_tip = st.columns([0.95, 0.05])
            with col_imp_btn:
                if st.button("Điền giá trị Missing"):
                    with st.spinner("Đang điền giá trị Missing..."):
                        X_filled = st.session_state["data_original"][0].fillna(st.session_state["data_original"][0].median())
                        st.session_state["data_filled"] = (X_filled, y)
                        st.success("Đã điền giá trị missing!")
            with col_imp_tip:
                st.markdown("""
                    <div class="tooltip" style="margin-left:-880px;">
                        <span>?</span>
                        <span class="tooltiptext">
                            Thay thế các giá trị thiếu trong dữ liệu bằng trung vị của từng cột, giúp tránh lỗi khi huấn luyện.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            if "data_filled" in st.session_state:
                st.write("**Kết quả Điền Missing:**")
                st.write(st.session_state["data_filled"][0].head())
    else:
        st.info("Vui lòng tải dữ liệu ở thẻ 'Tải Dữ liệu' trước khi xử lí.")

# ----------------- TAB 4: CHIA DỮ LIỆU -----------------
with tab_split:
    st.header("Chia Tách Dữ liệu")
    if 'data' in st.session_state:
        X, y = st.session_state['data']
        total_samples = X.shape[0]
        st.write(f"Tổng số mẫu: {total_samples}")
        valid_pct = st.slider("Validation set (%)", min_value=0, max_value=50, value=15)
        test_pct = st.slider("Test set (%)", min_value=0, max_value=50, value=15)
        train_pct = 100 - (valid_pct + test_pct)
        
        if train_pct < 0:
            st.warning("Tổng % Validation + Test vượt quá 100%. Hãy điều chỉnh lại.")
        else:
            train_samples = int(total_samples * train_pct / 100)
            valid_samples = int(total_samples * valid_pct / 100)
            test_samples = int(total_samples * test_pct / 100)
            st.write(f"**Tập huấn luyện**: {train_samples} mẫu ({train_pct}%)")
            st.write(f"**Tập validation**: {valid_samples} mẫu ({valid_pct}%)")
            st.write(f"**Tập kiểm tra**: {test_samples} mẫu ({test_pct}%)")
        
        if st.button("Chia dữ liệu"):
            if train_pct < 0:
                st.error("Không thể chia dữ liệu vì tổng % Validation + Test vượt quá 100%.")
            else:
                with st.spinner("Đang chia dữ liệu..."):
                    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42)
                    valid_size = valid_pct / (100 - test_pct) if (100 - test_pct) > 0 else 0
                    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)
                    st.session_state['split_data'] = {
                        "X_train": X_train, "y_train": y_train,
                        "X_valid": X_valid, "y_valid": y_valid,
                        "X_test": X_test, "y_test": y_test
                    }
                    st.success("Dữ liệu đã được chia thành công!")
                    st.write(f"Số mẫu Train: {X_train.shape[0]}")
                    st.write(f"Số mẫu Validation: {X_valid.shape[0]}")
                    st.write(f"Số mẫu Test: {X_test.shape[0]}")
    else:
        st.info("Vui lòng tải và xử lí dữ liệu ở các thẻ trước.")

# ----------------- TAB 5: HUẤN LUYỆN/ĐÁNH GIÁ MÔ HÌNH -----------------
with tab_train_eval:
    st.header("Huấn luyện và Đánh Giá Mô Hình")
    if 'split_data' in st.session_state:
        split_data = st.session_state['split_data']
        col_model, col_model_tooltip = st.columns([0.9, 0.9])
        with col_model:
            model_choice = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"])
        with col_model_tooltip:
            st.markdown("""
                <div class="tooltip" style="margin-top:10px;">
                    <span>?</span>
                    <span class="tooltiptext">
                        <strong>Decision Tree:</strong> Phân lớp dữ liệu theo cấu trúc cây.<br>
                        <strong>SVM:</strong> Tìm biên phân chia tối ưu dựa trên siêu phẳng.
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        params = {}
        if model_choice == "Decision Tree":
            st.subheader("Tham số cho Decision Tree")
            # Tooltip cho criterion với mô tả chi tiết
            col_crit, col_crit_tooltip = st.columns([0.9, 0.9])
            with col_crit:
                params["criterion"] = st.selectbox("criterion", ["gini", "entropy", "log_loss"])
            with col_crit_tooltip:
                st.markdown("""
                    <div class="tooltip" style="margin-top:10px; margin-left:20px;">
                        <span>?</span>
                        <span class="tooltiptext">
                            Chọn hàm đánh giá độ tinh khiết:
                            - <strong>gini</strong>: Dựa trên chỉ số Gini, phổ biến và tính toán nhanh.
                            - <strong>entropy</strong>: Dựa trên entropy, đo lường sự hỗn loạn trong nút.
                            - <strong>log_loss</strong>: Sử dụng log loss, phù hợp với một số bài toán đặc thù.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            # Tooltip cho max_depth
            col_depth, col_depth_tooltip = st.columns([0.9, 0.9])
            with col_depth:
                params["max_depth"] = st.number_input("max_depth", min_value=1, max_value=100, value=10)
            with col_depth_tooltip:
                st.markdown("""
                    <div class="tooltip" style="margin-top:10px; margin-left:20px;">
                        <span>?</span>
                        <span class="tooltiptext">
                            Giới hạn độ sâu của cây để tránh overfitting. Giá trị thấp giúp mô hình đơn giản, cao cho phép mô hình phức tạp hơn.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            # Tooltip cho min_samples_split
            col_split, col_split_tooltip = st.columns([0.9, 0.9])
            with col_split:
                params["min_samples_split"] = st.number_input("min_samples_split", min_value=2, max_value=50, value=5)
            with col_split_tooltip:
                st.markdown("""
                    <div class="tooltip" style="margin-top:10px; margin-left:20px;">
                        <span>?</span>
                        <span class="tooltiptext">
                            Số mẫu tối thiểu cần thiết để chia tách một nút. Giá trị cao hơn giúp hạn chế overfitting bằng cách yêu cầu nhiều mẫu hơn trước khi chia.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.subheader("Tham số cho SVM")
            # Tooltip cho tham số C
            col_C, col_C_tooltip = st.columns([0.9, 0.9])
            with col_C:
                params["C"] = st.number_input("C (Regularization parameter)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
            with col_C_tooltip:
                st.markdown("""
                    <div class="tooltip" style="margin-top:10px; margin-left:20px;">
                        <span>?</span>
                        <span class="tooltiptext">
                            Tham số điều chỉnh mức phạt cho lỗi. Giá trị cao: phân chia dữ liệu chặt chẽ (có thể gây overfitting), giá trị thấp: tạo biên phân cách rộng hơn.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            # Tooltip cho kernel
            col_kernel, col_kernel_tooltip = st.columns([0.9, 0.9])
            with col_kernel:
                params["kernel"] = st.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid"])
            with col_kernel_tooltip:
                st.markdown("""
                    <div class="tooltip" style="margin-top:10px; margin-left:20px;">
                        <span>?</span>
                        <span class="tooltiptext">
                            Chọn hàm nhân để chuyển đổi dữ liệu:
                            - <strong>linear</strong>: Tuyến tính.
                            - <strong>rbf</strong>: Radial Basis Function, hiệu quả với dữ liệu phi tuyến.
                            - <strong>poly</strong>: Đa thức (yêu cầu chỉ định degree).
                            - <strong>sigmoid</strong>: Hàm sigmoid, tương tự hàm kích hoạt trong mạng nơ-ron.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            # Tooltip cho gamma
            col_gamma, col_gamma_tooltip = st.columns([0.9, 0.9])
            with col_gamma:
                params["gamma"] = st.text_input("gamma (mặc định='scale')", value="scale")
            with col_gamma_tooltip:
                st.markdown("""
                    <div class="tooltip" style="margin-top:10px; margin-left:20px;">
                        <span>?</span>
                        <span class="tooltiptext">
                            Hệ số cho kernel, xác định mức độ ảnh hưởng của từng mẫu. Mặc định là 'scale'; có thể điều chỉnh để tối ưu hóa hiệu năng.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            if params["kernel"] == "poly":
                # Tooltip cho degree khi kernel là poly
                col_degree, col_degree_tooltip = st.columns([0.9, 0.9])
                with col_degree:
                    params["degree"] = st.number_input("degree (cho kernel poly)", min_value=1, max_value=10, value=3)
                with col_degree_tooltip:
                    st.markdown("""
                        <div class="tooltip" style="margin-top:10px; margin-left:20px;">
                            <span>?</span>
                            <span class="tooltiptext">
                                Độ bậc của hàm đa thức khi sử dụng kernel poly, xác định độ phức tạp của mối quan hệ phi tuyến.
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
        
        if st.button("Huấn luyện mô hình"):
            with st.spinner("Đang huấn luyện mô hình..."):
                X_train = split_data["X_train"]
                y_train = split_data["y_train"]
                X_valid = split_data["X_valid"]
                y_valid = split_data["y_valid"]
                
                # Kiểm tra missing values và áp dụng imputation nếu cần
                if pd.isnull(X_train).sum().sum() > 0:
                    st.warning("Dữ liệu huấn luyện chứa missing values. Áp dụng imputation (median)...")
                    imputer = SimpleImputer(strategy='median')
                    X_train = imputer.fit_transform(X_train)
                    X_valid = imputer.transform(X_valid)
                    split_data["X_test"] = imputer.transform(split_data["X_test"])
                    st.session_state["imputer"] = imputer
                    st.session_state['split_data'] = split_data
                
                # Tạo model
                if model_choice == "Decision Tree":
                    model = DecisionTreeClassifier(**params)
                else:
                    model = SVC(probability=True, **params)
                
                # Bắt đầu 1 run MLflow
                with mlflow.start_run() as run:
                    run_id = run.info.run_id
                    model.fit(X_train, y_train)
                    
                    # Đánh giá trên tập Validation
                    y_pred_val = model.predict(X_valid)
                    acc_val = accuracy_score(y_valid, y_pred_val)
                    mlflow.log_metric("val_accuracy", acc_val)
                    
                    # Đánh giá trên tập Test
                    X_test = split_data["X_test"]
                    y_test = split_data["y_test"]
                    y_pred_test = model.predict(X_test)
                    acc_test = accuracy_score(y_test, y_pred_test)
                    mlflow.log_metric("test_accuracy", acc_test)
                    
                    mlflow.log_param("model_choice", model_choice)
                    for key, value in params.items():
                        mlflow.log_param(key, value)
                    
                    # Log Confusion Matrix (Validation)
                    cm_val = confusion_matrix(y_valid, y_pred_val)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm_val, annot=True, fmt='d', cmap="Blues", ax=ax)
                    ax.set_title("Confusion Matrix (Validation)")
                    ax.set_xlabel("Dự đoán")
                    ax.set_ylabel("Thực tế")
                    tmp_dir = tempfile.mkdtemp()
                    artifact_path = os.path.join(tmp_dir, "confusion_matrix_val.png")
                    fig.savefig(artifact_path)
                    mlflow.log_artifact(artifact_path, artifact_path="confusion_matrix_val")
                    
                    st.session_state["run_id"] = run_id
                    st.session_state["accuracy_val"] = acc_val
                    st.session_state["accuracy_test"] = acc_test
                    st.session_state["params"] = params
                    
                    st.success("Huấn luyện mô hình thành công!")
                    st.markdown(f"""
                    <div style="background-color: #cce5ff; padding: 10px; border-radius: 5px;">
                        <strong>Accuracy trên validation:</strong> {acc_val:.4f}<br>
                        <strong>Accuracy trên test:</strong> {acc_test:.4f}<br>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state["trained_model"] = model
                    st.pyplot(fig)
                    st.write("Biểu đồ Confusion Matrix hiển thị số dự đoán đúng và sai.")
    else:
        st.info("Vui lòng chia tách dữ liệu ở thẻ 'Chia dữ liệu' trước khi huấn luyện mô hình.")

# ----------------- TAB 6: DEMO DỰ ĐOÁN -----------------
with tab_demo:
    st.header("Demo Dự đoán")
    mode = st.radio("Chọn phương thức dự đoán:", ["Dữ liệu từ Test", "Upload ảnh mới"])
    if mode == "Dữ liệu từ Test":
        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia tách dữ liệu ở thẻ 'Chia dữ liệu'.")
        else:
            split_data = st.session_state['split_data']
            X_test = split_data["X_test"]
            y_test = split_data["y_test"]
            st.write(f"Số mẫu trong tập Test: {X_test.shape[0]}")
            selection_mode = st.radio("Chọn chế độ chọn mẫu:", ["Ngẫu nhiên", "Chọn thủ công"], key="test_selection_mode")
            if selection_mode == "Ngẫu nhiên":
                if st.button("Chọn mẫu ngẫu nhiên", key="btn_random"):
                    idx = np.random.randint(0, X_test.shape[0])
                    sample = X_test.iloc[idx] if isinstance(X_test, pd.DataFrame) else X_test[idx]
                    true_label = y_test.iloc[idx] if isinstance(y_test, pd.Series) else y_test[idx]
                    st.session_state.selected_sample = sample
                    st.session_state.selected_true_label = true_label
                    st.success("Mẫu đã được chọn ngẫu nhiên.")
            else:
                idx_manual = st.slider("Chọn chỉ số mẫu từ tập Test:", min_value=0, max_value=X_test.shape[0]-1, value=0, key="slider_manual")
                if st.button("Chọn mẫu thủ công", key="btn_manual"):
                    sample = X_test.iloc[idx_manual] if isinstance(X_test, pd.DataFrame) else X_test[idx_manual]
                    true_label = y_test.iloc[idx_manual] if isinstance(y_test, pd.Series) else y_test[idx_manual]
                    st.session_state.selected_sample = sample
                    st.session_state.selected_true_label = true_label
                    st.success("Mẫu đã được chọn thủ công.")
            if "selected_sample" in st.session_state:
                sample = st.session_state.selected_sample
                true_label = st.session_state.selected_true_label
                st.write("Nhãn thực:", true_label)
                try:
                    sample_arr = sample.values if hasattr(sample, "values") else sample
                    if sample_arr.size != 784:
                        st.warning(f"Số lượng đặc trưng của mẫu là {sample_arr.size} (yêu cầu 784).")
                        padding = np.zeros(784 - sample_arr.size)
                        sample_arr = np.concatenate([sample_arr, padding])
                    image = sample_arr.reshape(28, 28)
                    st.image(image, caption="Hình ảnh mẫu", width=300)
                except Exception as e:
                    st.error(f"Không thể hiển thị hình ảnh: {e}")
            if st.button("Dự đoán", key="btn_predict_test"):
                with st.spinner("Đang dự đoán..."):
                    if "selected_sample" in st.session_state:
                        sample_input = st.session_state.selected_sample
                        sample_input = sample_input.values if hasattr(sample_input, "values") else sample_input
                        if sample_input.size != 784:
                            st.warning(f"Số lượng đặc trưng của mẫu là {sample_input.size} (yêu cầu 784). Đang padding 0...")
                            padding = np.zeros(784 - sample_input.size)
                            sample_input = np.concatenate([sample_input, padding])
                        if "trained_model" in st.session_state:
                            model = st.session_state["trained_model"]
                            if "imputer" in st.session_state:
                                sample_input = st.session_state["imputer"].transform([sample_input])[0]
                            pred = model.predict([sample_input])
                            st.session_state.prediction = pred[0]
                            st.write("Dự đoán:", pred[0])
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba([sample_input])
                                confidence = np.max(probs)
                                st.write("Độ tin cậy:", confidence)
                        else:
                            st.warning("Vui lòng huấn luyện mô hình trước.")
                    else:
                        st.warning("Vui lòng chọn mẫu trước khi dự đoán.")
            if st.button("Xóa dự đoán", key="btn_clear_test"):
                if "prediction" in st.session_state:
                    del st.session_state["prediction"]
                st.success("Đã xóa kết quả dự đoán.")
                st.stop()
    elif mode == "Upload ảnh mới":
        uploaded_img = st.file_uploader("Upload ảnh (png, jpg, jpeg)", type=["png", "jpg", "jpeg"], key="uploader_new")
        if uploaded_img is not None:
            with st.spinner("Đang xử lí: Upload ảnh..."):
                image = Image.open(uploaded_img).convert("L")
                image = image.resize((28, 28))
                st.image(image, caption="Ảnh đã upload", width=300)
                image_array = np.array(image).flatten()
                image_array = image_array / 255.0
                st.session_state.uploaded_image_array = image_array
        if "uploaded_image_array" in st.session_state:
            if st.button("Dự đoán", key="btn_predict_upload"):
                with st.spinner("Đang dự đoán..."):
                    image_array = st.session_state.uploaded_image_array
                    if image_array.size != 784:
                        st.error(f"Số lượng đặc trưng của ảnh là {image_array.size} (yêu cầu 784).")
                    else:
                        if "trained_model" in st.session_state:
                            model = st.session_state["trained_model"]
                            if "imputer" in st.session_state:
                                image_array = st.session_state["imputer"].transform([image_array])[0]
                            pred = model.predict([image_array])
                            st.session_state.prediction = pred[0]
                            st.write("Dự đoán:", pred[0])
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba([image_array])
                                confidence = np.max(probs)
                                st.write("Độ tin cậy:", confidence)
                        else:
                            st.warning("Vui lòng huấn luyện mô hình ở thẻ 'Huấn luyện/Đánh Giá' trước.")
            if st.button("Xóa dự đoán", key="btn_clear_upload"):
                if "prediction" in st.session_state:
                    del st.session_state["prediction"]
                st.success("Đã xóa kết quả dự đoán.")
                st.stop()

# ----------------- TAB 7: THÔNG TIN HUẤN LUYỆN & MLflow UI -----------------
with tab_log_info:
    st.header("Thông tin Huấn luyện")
    if "run_id" in st.session_state:
        st.markdown("### Thông tin đã log")
        st.write(f"**Run ID:** {st.session_state['run_id']}")
        st.write(f"**Accuracy trên Validation:** {st.session_state['accuracy_val']:.4f}")
        st.write(f"**Accuracy trên Test:** {st.session_state['accuracy_test']:.4f}")
        st.markdown("### Tham số đã sử dụng")
        st.json(st.session_state["params"])
    else:
        st.info("Chưa có thông tin nào được log. Vui lòng huấn luyện mô hình trước.")
    st.markdown("---")
    if st.button("Mở MLflow UI"):
        mlflow_url = "https://dagshub.com/huykibo/streamlit_mlflow.mlflow/#/experiments/0"
        st.markdown(f'**[Click vào đây để mở MLflow UI]({mlflow_url})**')
