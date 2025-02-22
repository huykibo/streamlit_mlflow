import streamlit as st
import mlflow
import mlflow.sklearn
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
import os, tempfile
import streamlit.components.v1 as components

# Cấu hình Tracking URI của MLflow để log về máy chủ (IP: 192.168.1.12, port 5000)
mlflow.set_tracking_uri("http://192.168.1.12:5000")

# Cấu hình trang ứng dụng
st.set_page_config(page_title="MNIST App với Streamlit & MLFlow", layout="wide")
st.title("Ứng dụng MNIST với Streamlit & MLFlow")

# CSS cho tooltip: đặt tooltip bên phải, canh giữa dọc
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
      width: 300px;
      background-color: #f9f9f9;
      color: #333;
      text-align: left;
      border-radius: 6px;
      padding: 8px;
      position: absolute;
      z-index: 1;
      top: 50%;
      left: 105%;
      transform: translateY(-50%);
      margin-left: 8px;
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

# Tạo các tab cho giao diện, đã thêm tab "Log Info" để hiển thị các log đã được lưu và thêm nút xóa log
tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info, tab_mlflow = st.tabs(
    ["Thông tin", "Tải dữ liệu", "Xử lí dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Log Info", "MLflow UI"]
)

# --- Tab 1: Thông tin ---
with tab_info:
    st.header("Thông tin Ứng dụng")
    st.write("""
    Ứng dụng này thực hiện các thao tác:
    - Tải dữ liệu MNIST từ OpenML (hoặc cho phép người dùng upload dữ liệu).
    - Xử lí dữ liệu theo các thao tác do người dùng tùy chỉnh.
    - Chia tách dữ liệu thành Train, Validation, và Test theo tỷ lệ điều chỉnh bằng thanh kéo.
    - Huấn luyện và đánh giá mô hình phân lớp (Decision Tree, SVM) với MLFlow ghi lại các tham số và kết quả.
    - Demo dự đoán trên một mẫu dữ liệu từ tập Test hoặc upload ảnh mới để dự đoán.
    - Xem giao diện MLflow UI và các log được lưu trong MLflow.
    """)

# --- Tab 2: Tải dữ liệu ---
with tab_load:
    st.header("Tải Dữ liệu")
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
                    st.error("File upload cần có cột 'label' để làm nhãn.")

# --- Tab 3: Xử lí dữ liệu ---
with tab_preprocess:
    st.header("Xử lí Dữ liệu")
    if 'data' in st.session_state:
        X, y = st.session_state['data']
        if "data_original" not in st.session_state:
            st.session_state["data_original"] = (X, y)
        st.subheader("Dữ liệu Gốc")
        st.write(X.head())
        
        st.markdown("### Các thao tác xử lí dữ liệu")
        
        # ====== Normalization ======
        with st.container():
            col_norm_btn, col_norm_tip = st.columns([0.95, 0.05])
            with col_norm_btn:
                if st.button("Chuẩn hoá (Normalization)"):
                    with st.spinner("Đang xử lí: Chuẩn hoá dữ liệu..."):
                        X_norm = st.session_state["data_original"][0] / 255.0
                        st.session_state["data_norm"] = (X_norm, y)
                        st.success("Đã chuẩn hoá dữ liệu!")
            with col_norm_tip:
                st.markdown(
                    """
                    <div class="tooltip" style="margin-left:-630px;">?
                        <span class="tooltiptext">
                            <strong>Normalization</strong>: 
                            Chuyển đổi giá trị pixel từ [0, 255] về [0, 1] bằng cách chia cho 255.
                            Giúp giảm kích thước dữ liệu và hỗ trợ quá trình huấn luyện.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            if "data_norm" in st.session_state:
                st.write("**Kết quả Chuẩn hoá:**")
                st.write(st.session_state["data_norm"][0].head())
                
        # ====== Standardization ======
        with st.container():
            col_std_btn, col_std_tip = st.columns([0.95, 0.05])
            with col_std_btn:
                if st.button("Standardization"):
                    with st.spinner("Đang xử lí: Standardization..."):
                        X_std = (st.session_state["data_original"][0] - st.session_state["data_original"][0].mean()) / st.session_state["data_original"][0].std()
                        st.session_state["data_std"] = (X_std, y)
                        st.success("Đã thực hiện Standardization!")
            with col_std_tip:
                st.markdown(
                    """
                    <div class="tooltip" style="margin-left:-630px;">?
                        <span class="tooltiptext">
                            <strong>Standardization</strong>: 
                            Chuẩn hoá dữ liệu bằng cách trừ đi giá trị trung bình và chia cho độ lệch chuẩn của từng cột.
                            Giúp dữ liệu có phân phối chuẩn, cải thiện hiệu suất của nhiều thuật toán.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            if "data_std" in st.session_state:
                st.write("**Kết quả Standardization:**")
                st.write(st.session_state["data_std"][0].head())
                
        # ====== Missing Imputation ======
        with st.container():
            col_imp_btn, col_imp_tip = st.columns([0.95, 0.05])
            with col_imp_btn:
                if st.button("Điền giá trị Missing"):
                    with st.spinner("Đang xử lí: Điền giá trị Missing..."):
                        X_filled = st.session_state["data_original"][0].fillna(st.session_state["data_original"][0].median())
                        st.session_state["data_filled"] = (X_filled, y)
                        st.success("Đã điền giá trị missing!")
            with col_imp_tip:
                st.markdown(
                    """
                    <div class="tooltip" style="margin-left:-630px;">?
                        <span class="tooltiptext">
                            <strong>Missing Imputation</strong>: 
                            Thay thế các giá trị thiếu (missing values) bằng giá trị trung vị của cột tương ứng.
                            Điều này đảm bảo dữ liệu đầy đủ cho quá trình huấn luyện mô hình.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            if "data_filled" in st.session_state:
                st.write("**Kết quả Điền Missing:**")
                st.write(st.session_state["data_filled"][0].head())
    else:
        st.info("Vui lòng tải dữ liệu ở thẻ 'Tải Dữ liệu' trước khi xử lí.")

# --- Tab 4: Chia dữ liệu ---
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
                with st.spinner("Đang xử lí: Chia dữ liệu..."):
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

# --- Tab 5: Huấn luyện/Đánh giá Mô hình ---
with tab_train_eval:
    st.header("Huấn luyện và Đánh giá Mô hình")
    if 'split_data' in st.session_state:
        split_data = st.session_state['split_data']

        # Khung chọn loại mô hình với tooltip mô tả chi tiết
        col_model, col_model_tooltip = st.columns([0.9, 0.1])
        with col_model:
            model_choice = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"])
        with col_model_tooltip:
            st.markdown(
                '''
                <div class="tooltip" style="margin-top:10px;">?
                    <span class="tooltiptext">
                        <strong>Decision Tree</strong>: Phân lớp dựa trên cấu trúc cây, trong đó mỗi nút là quyết định dựa trên giá trị của thuộc tính.<br><br>
                        <strong>SVM (Support Vector Machine)</strong>: Sử dụng siêu phẳng tối ưu để phân chia dữ liệu thành các lớp, tối đa hóa khoảng cách giữa các lớp.
                    </span>
                </div>
                ''',
                unsafe_allow_html=True
            )

        params = {}
        if model_choice == "Decision Tree":
            st.subheader("Tham số cho Decision Tree")

            # Criterion
            col_crit, col_crit_tooltip = st.columns([0.9, 0.1])
            with col_crit:
                params["criterion"] = st.selectbox("criterion", ["gini", "entropy", "log_loss"])
            with col_crit_tooltip:
                st.markdown(
                    '''
                    <div class="tooltip" style="margin-top:10px;">?
                        <span class="tooltiptext">
                            <strong>criterion</strong>: Hàm đánh giá độ tinh khiết của nút khi chia tách dữ liệu.<br>
                            - <em>gini</em>: Sử dụng chỉ số Gini để đo impurity.<br>
                            - <em>entropy</em>: Dựa vào entropy để đánh giá chất lượng của split.<br>
                            - <em>log_loss</em>: Dùng log loss nếu dữ liệu phù hợp.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            # max_depth
            col_depth, col_depth_tooltip = st.columns([0.9, 0.1])
            with col_depth:
                params["max_depth"] = st.number_input("max_depth", min_value=1, max_value=100, value=10)
            with col_depth_tooltip:
                st.markdown(
                    '''
                    <div class="tooltip" style="margin-top:10px;">?
                        <span class="tooltiptext">
                            <strong>max_depth</strong>: Giới hạn độ sâu của cây nhằm kiểm soát độ phức tạp và tránh overfitting.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            # min_samples_split
            col_split, col_split_tooltip = st.columns([0.9, 0.1])
            with col_split:
                params["min_samples_split"] = st.number_input("min_samples_split", min_value=2, max_value=50, value=5)
            with col_split_tooltip:
                st.markdown(
                    '''
                    <div class="tooltip" style="margin-top:10px;">?
                        <span class="tooltiptext">
                            <strong>min_samples_split</strong>: Số mẫu tối thiểu cần có để chia một nút.
                            Giúp hạn chế việc tạo ra các nút quá nhỏ, tránh overfitting.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        else:
            st.subheader("Tham số cho SVM")

            # C
            col_C, col_C_tooltip = st.columns([0.9, 0.1])
            with col_C:
                params["C"] = st.number_input("C (Regularization parameter)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
            with col_C_tooltip:
                st.markdown(
                    '''
                    <div class="tooltip" style="margin-top:10px;">?
                        <span class="tooltiptext">
                            <strong>C</strong>: Tham số điều chỉnh mức độ phạt cho lỗi.
                            Giá trị lớn làm giảm regularization (có thể overfit), giá trị nhỏ tăng regularization (có thể underfit).
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            # kernel
            col_kernel, col_kernel_tooltip = st.columns([0.9, 0.1])
            with col_kernel:
                params["kernel"] = st.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid"])
            with col_kernel_tooltip:
                st.markdown(
                    '''
                    <div class="tooltip" style="margin-top:10px;">?
                        <span class="tooltiptext">
                            <strong>kernel</strong>: Hàm nhân chuyển đổi dữ liệu sang không gian cao chiều.<br>
                            - <em>linear</em>: Dữ liệu tuyến tính.<br>
                            - <em>rbf</em>: Radial Basis Function, phổ biến cho dữ liệu phi tuyến.<br>
                            - <em>poly</em>: Hàm đa thức, dùng khi dữ liệu có quan hệ đa thức.<br>
                            - <em>sigmoid</em>: Hàm sigmoid, tương tự như hàm kích hoạt trong mạng nơ-ron.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            # gamma
            col_gamma, col_gamma_tooltip = st.columns([0.9, 0.1])
            with col_gamma:
                params["gamma"] = st.text_input("gamma (mặc định='scale')", value="scale")
            with col_gamma_tooltip:
                st.markdown(
                    '''
                    <div class="tooltip" style="margin-top:10px;">?
                        <span class="tooltiptext">
                            <strong>gamma</strong>: Hệ số cho kernel (áp dụng với rbf, poly, sigmoid).
                            Kiểm soát phạm vi ảnh hưởng của mỗi mẫu: giá trị nhỏ mở rộng phạm vi, giá trị lớn thu hẹp phạm vi.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            # degree (chỉ dùng với kernel = poly)
            if params["kernel"] == "poly":
                col_degree, col_degree_tooltip = st.columns([0.9, 0.1])
                with col_degree:
                    params["degree"] = st.number_input("degree (cho kernel poly)", min_value=1, max_value=10, value=3)
                with col_degree_tooltip:
                    st.markdown(
                        '''
                        <div class="tooltip" style="margin-top:10px;">?
                            <span class="tooltiptext">
                                <strong>degree</strong>: Độ bậc của hàm đa thức, chỉ áp dụng khi kernel là "poly".
                                Giá trị càng cao thì hàm nhân càng phức tạp.
                            </span>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

        # Nút huấn luyện
        if st.button("Huấn luyện mô hình"):
            with st.spinner("Đang huấn luyện mô hình..."):
                X_train = split_data["X_train"]
                y_train = split_data["y_train"]
                X_valid = split_data["X_valid"]
                y_valid = split_data["y_valid"]

                # Kiểm tra và impute nếu có missing values
                if pd.isnull(X_train).sum().sum() > 0:
                    st.warning("Dữ liệu huấn luyện chứa missing values. Áp dụng imputation (median)...")
                    imputer = SimpleImputer(strategy='median')
                    X_train = imputer.fit_transform(X_train)
                    X_valid = imputer.transform(X_valid)
                    split_data["X_test"] = imputer.transform(split_data["X_test"])
                    st.session_state["imputer"] = imputer
                    st.session_state['split_data'] = split_data

                with mlflow.start_run() as run:
                    if model_choice == "Decision Tree":
                        model = DecisionTreeClassifier(**params)
                    else:
                        if params["gamma"] not in ["scale", "auto"]:
                            try:
                                params["gamma"] = float(params["gamma"])
                            except:
                                st.warning("Giá trị gamma không hợp lệ, mặc định='scale'.")
                                params["gamma"] = "scale"
                        model = SVC(probability=True, **params)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_valid)
                    acc = accuracy_score(y_valid, y_pred)

                    mlflow.log_param("model", model_choice)
                    for key, value in params.items():
                        mlflow.log_param(key, value)
                    mlflow.log_metric("accuracy", acc)

                    mlflow.sklearn.log_model(model, artifact_path="model")

                    st.success(f"Accuracy trên validation: {acc:.4f}")
                    st.session_state["trained_model"] = model

                    cm = confusion_matrix(y_valid, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                    ax.set_title("Confusion Matrix - Validation")
                    ax.set_xlabel("Dự đoán")
                    ax.set_ylabel("Thực tế")
                    st.pyplot(fig)
                    st.write("""
                    Biểu đồ Confusion Matrix hiển thị số lượng dự đoán đúng (trên đường chéo)
                    và số lượng dự đoán sai, giúp đánh giá hiệu năng của mô hình trên tập Validation.
                    """)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        fig_path = os.path.join(tmp_dir, "confusion_matrix.png")
                        fig.savefig(fig_path)
                        mlflow.log_artifact(fig_path, artifact_path="confusion_matrix")

                    # Hiển thị thông tin tracking MLflow
                    st.write("---")
                    st.write("**MLflow Tracking**")
                    st.write(f"**Run ID**: {run.info.run_id}")
                    st.write(f"**Artifact URI**: {run.info.artifact_uri}")
                    st.write("""
                    Để xem chi tiết các log (parameters, metrics, artifacts),
                    hãy mở giao diện MLflow UI (ví dụ: chạy lệnh `mlflow ui --host 0.0.0.0 --port 5000`
                    và truy cập qua http://192.168.1.12:5000).
                    """)
                    
                    # Đánh giá trên tập Test:
                    X_test = st.session_state['split_data']["X_test"]
                    y_test = st.session_state['split_data']["y_test"]
                    if "imputer" in st.session_state:
                        X_test = st.session_state["imputer"].transform(X_test)
                    y_pred_test = st.session_state["trained_model"].predict(X_test)
                    acc_test = accuracy_score(y_test, y_pred_test)
                    st.write(f"**Accuracy trên tập Test:** {acc_test:.4f}")
                    fig_test, ax_test = plt.subplots()
                    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap="Greens", ax=ax_test)
                    ax_test.set_title("Confusion Matrix - Test")
                    ax_test.set_xlabel("Dự đoán")
                    ax_test.set_ylabel("Thực tế")
                    st.pyplot(fig_test)
    else:
        st.info("Vui lòng chia tách dữ liệu ở thẻ 'Chia dữ liệu' trước khi huấn luyện mô hình.")

# --- Tab 6: Demo dự đoán ---
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
                idx_manual = st.slider("Chọn chỉ số mẫu từ tập Test:",
                                       min_value=0, max_value=X_test.shape[0]-1, value=0, key="slider_manual")
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
                with st.spinner("Đang dự đoán ..."):
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
                        else:
                            st.warning("Vui lòng huấn luyện mô hình trước.")
                    else:
                        st.warning("Vui lòng chọn mẫu trước khi dự đoán.")
            
            if st.button("Xóa dự đoán", key="btn_clear_test"):
                for key in ["selected_sample", "selected_true_label", "prediction"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Đã xóa kết quả dự đoán và mẫu đã chọn.")
                st.experimental_rerun()
    
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
                with st.spinner("Đang dự đoán ..."):
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
                        else:
                            st.warning("Vui lòng huấn luyện mô hình ở thẻ 'Huấn luyện/Đánh giá' trước.")
            if st.button("Xóa dự đoán", key="btn_clear_upload"):
                for key in ["uploaded_image_array", "prediction"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Đã xóa kết quả dự đoán và ảnh đã upload.")
                st.experimental_rerun()

# --- Tab 7: Log Info ---
with tab_log_info:
    st.header("Thông tin Log từ MLflow")
    try:
        # Lấy experiment mặc định (Default)
        experiment = mlflow.get_experiment_by_name("Default")
        experiment_id = experiment.experiment_id if experiment is not None else "0"
        runs_df = mlflow.search_runs(experiment_ids=[experiment_id])
        if runs_df.empty:
            st.info("Chưa có log nào được ghi lại.")
        else:
            st.dataframe(runs_df)
    except Exception as e:
        st.error(f"Không thể tải log từ MLflow: {e}")
    
    if st.button("Xóa log"):
        try:
            if not runs_df.empty:
                for run_id in runs_df["run_id"]:
                    mlflow.delete_run(run_id)
                st.success("Đã xóa log thành công!")
                st.experimental_rerun()
            else:
                st.info("Không có log nào để xóa.")
        except Exception as e:
            st.error(f"Lỗi khi xóa log: {e}")

# --- Tab 8: MLflow UI ---
with tab_mlflow:
    st.header("MLflow Tracking UI")
    st.write("Dưới đây là giao diện MLflow UI, bạn có thể xem chi tiết các log, parameters, metrics và artifacts của các run:")
    components.iframe("http://192.168.1.12:5000", height=800)
