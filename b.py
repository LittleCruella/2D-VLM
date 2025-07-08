import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from collections import defaultdict

# === Cấu hình đường dẫn ===
csv_path = "data/results.csv"
image_folder = "data/flickr30k_images"
image_prefix = "data/flickr30k_images/"

# === Đọc danh sách file ảnh trong thư mục ===
image_files = set(os.listdir(image_folder))

# === Gom tất cả comment theo image_name ===
image_comments = defaultdict(list)

# === Đọc và xử lý từng dòng ===
with open(csv_path, "r", encoding="utf-8") as f:
    next(f)  # Bỏ dòng tiêu đề
    for line in f:
        line = line.strip().rstrip(',')
        parts = line.split('|')

        if len(parts) < 3:
            continue

        image_name = parts[0].strip()
        try:
            comment_number = int(parts[1].strip())
        except ValueError:
            continue
        comment = '|'.join(parts[2:]).strip()

        if image_name in image_files and 0 <= comment_number <= 4:
            image_comments[image_name].append((comment_number, comment))

# === Gộp comment cho từng ảnh ===
filtered_rows = []
for image_name, comments in image_comments.items():
    if len(comments) < 5:
        continue  # đảm bảo đủ 5 comment (0-4)
    comments.sort(key=lambda x: x[0])
    full_comment = " ".join([c[1] for c in comments])
    filtered_rows.append({
        "image_name": image_name,
        "comment": full_comment
    })

# === Chia train/val/test: 8:1:1 ===
train_val, test = train_test_split(filtered_rows, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=1/9, random_state=42)  # 10% validation

# === Hàm chuyển sang JSON format ===
def to_json_format(data):
    return [
        {
            "image": image_prefix + item["image_name"],
            "text": item["comment"]
        }
        for item in data
    ]

# === Tạo dict JSON xuất ra ===
output_json = {
    "train": to_json_format(train),
    "validation": to_json_format(val),
    "test": to_json_format(test)
}

# === Ghi ra file JSON ===
os.makedirs("data/2D_Cap", exist_ok=True)
with open("data/2D_Cap/2D_Cap.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=4)

# === In kết quả ===
print("✅ JSON đã tạo thành công!")
print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
