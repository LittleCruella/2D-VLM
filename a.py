import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# === Cấu hình đường dẫn ===
csv_path = "data/results.csv"
image_folder = "data/flickr30k_images"

# === Đọc danh sách file ảnh trong thư mục ===
image_files = set(os.listdir(image_folder))

# === Danh sách lưu kết quả hợp lệ ===
filtered_rows = []

# === Xử lý từng dòng bằng tay ===
with open(csv_path, "r", encoding="utf-8") as f:
    next(f)  # bỏ dòng tiêu đề
    for line in f:
        # Xử lý dòng: xóa dấu xuống dòng, dấu phẩy cuối nếu có
        line = line.strip().rstrip(',')
        parts = line.split('|')
        
        if len(parts) < 3:
            continue  # dòng lỗi, bỏ qua

        image_name = parts[0].strip()
        try:
            comment_number = int(parts[1].strip())
        except ValueError:
            continue
        comment = '|'.join(parts[2:]).strip()  # nối lại nếu comment chứa dấu |

        # Kiểm tra điều kiện
        if comment_number == 4 and image_name in image_files:
            filtered_rows.append({
                'image_name': image_name,
                'comment_number': comment_number,
                'comment': comment
            })

# === Chuyển sang DataFrame và lưu ra file ===
df_filtered = pd.DataFrame(filtered_rows)
df_filtered.to_csv("filtered_results.csv", index=False)

print(f"✅ Hoàn tất! Số dòng được lọc: {len(df_filtered)}")

image_prefix = "flickr30k_images/"

# Bước 1: Chia dữ liệu 8:1:1
train_val, test = train_test_split(filtered_rows, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=1/9, random_state=42)  # (1/9) * 0.9 = 0.1

# Bước 2: Hàm chuyển thành JSON format
def to_json_format(data):
    return [
        {
            "image": image_prefix + item["image_name"],
            "text": item["comment"]
        }
        for item in data
    ]

# Bước 3: Tạo dict kết quả
output_json = {
    "train": to_json_format(train),
    "validation": to_json_format(val),
    "test": to_json_format(test)
}

# Bước 4: Lưu ra file JSON
with open("data/2D_Cap/2D_Cap.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=4)

# In kết quả
print("✅ JSON đã tạo thành công!")
print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")