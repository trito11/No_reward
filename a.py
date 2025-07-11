import torch
print(torch.cuda.is_available())  # Kiểm tra CUDA có hoạt động không
print(torch.cuda.device_count())  # Số lượng GPU khả dụng
print(torch.cuda.get_device_name(0))  # Tên GPU đầu tiên
print(torch.version.cuda)  # Phiên bản CUDA mà PyTorch sử dụng
print(torch.backends.cudnn.version())  # Phiên bản cuDNN
