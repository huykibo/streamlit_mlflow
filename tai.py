import pandas as pd
import os
from sklearn.datasets import fetch_openml

# Tải dữ liệu MNIST
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data, mnist.target.astype(int)

# Gộp dữ liệu vào DataFrame
df = pd.DataFrame(X)
df['label'] = y  # Thêm cột nhãn

# Đường dẫn lưu file
save_path = r"C:\py\mlflow\mnist.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Tạo thư mục nếu chưa có

# Lưu thành CSV
df.to_csv(save_path, index=False)
print(f"Đã lưu MNIST thành CSV tại {save_path}!")
