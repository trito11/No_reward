import torch
import torch.nn.functional as F
import numpy as np
# Example usage of F.mse_loss
arr = np.arange(192).reshape(6, 8,2,2)
selected_rows = np.random.choice(arr.shape[0], size=2, replace=False)

# Chọn 5 cột ngẫu nhiên cho từng hàng
selected_cols = np.random.randint(0, arr.shape[1], size=(2, 2))
selected_ev = np.random.randint(0, arr.shape[1], size=(2, 2))

# Lấy giá trị tương ứng
random_values = arr[selected_rows[:, None]][selected_cols,selected_ev,:]

print("\nHàng đã chọn:", selected_rows)
print("Cột đã chọn:\n", selected_cols)
print("\nGiá trị ngẫu nhiên:\n", random_values)
