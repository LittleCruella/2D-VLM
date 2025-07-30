from transformers import AutoModel, AutoConfig

def check_model_config(model_name_or_path):
    # Load model configuration
    config = AutoConfig.from_pretrained(model_name_or_path)
    
    # Print the configuration
    print("Model Configuration:")
    print(config)

# Example usage
model_name_or_path = "output/Med3DVLM-Qwen-2.5-1.5B-finetune/model_with_lora.bin"  # Thay bằng đường dẫn hoặc tên mô hình của bạn
check_model_config(model_name_or_path)