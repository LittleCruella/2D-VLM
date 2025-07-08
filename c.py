import torch

# Kiểm tra xem CUDA có sẵn không
cuda_available = torch.cuda.is_available()
print(f"CUDA có sẵn: {cuda_available}")

if cuda_available:
    # Số lượng GPU có sẵn
    num_gpus = torch.cuda.device_count()
    print(f"Số lượng GPU có sẵn: {num_gpus}")

    # Tên của GPU đầu tiên (nếu có)
    if num_gpus > 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Tên GPU: {gpu_name}")
        
        # Phiên bản CUDA mà PyTorch đang sử dụng
        cuda_version_pytorch = torch.version.cuda
        print(f"Phiên bản CUDA PyTorch đang sử dụng: {cuda_version_pytorch}")

        # Thông tin chi tiết về GPU đầu tiên
        gpu_properties = torch.cuda.get_device_properties(0)
        print(f"Tổng bộ nhớ GPU: {gpu_properties.total_memory / (1024**3):.2f} GB")
        print(f"Khả năng tính toán (Compute Capability): {gpu_properties.major}.{gpu_properties.minor}")
        
else:
    print("CUDA không có sẵn trên hệ thống này. PyTorch sẽ chạy trên CPU.")