import json
import os
import random

def convert_txt_to_json(input_txt_file, output_json_file="data.json",
                         train_ratio=0.8, validation_ratio=0.1, test_ratio=0.05,
                         image_prefix="test/non-radiology/images/"):
    """
    Chuyển đổi file .txt (dạng image_name\tcaption) thành file JSON
    với các tập 'train', 'validation', và 'test' dựa trên tỷ lệ đã cho.
    Phần dữ liệu còn lại không được phân bổ sẽ bị loại bỏ.

    Args:
        input_txt_file (str): Đường dẫn đến file .txt đầu vào.
        output_json_file (str): Đường dẫn đến file JSON đầu ra.
        train_ratio (float): Tỷ lệ dữ liệu cho tập huấn luyện (0.0 đến 1.0).
        validation_ratio (float): Tỷ lệ dữ liệu cho tập kiểm định (0.0 đến 1.0).
        test_ratio (float): Tỷ lệ dữ liệu cho tập kiểm tra (0.0 đến 1.0).
        image_prefix (str): Tiền tố để thêm vào tên ảnh (ví dụ: "flickr30k_images/").
                             Giúp khớp với định dạng JSON mong muốn.
    """
    if not (0.0 <= train_ratio <= 1.0 and
            0.0 <= validation_ratio <= 1.0 and
            0.0 <= test_ratio <= 1.0):
        raise ValueError("Tỷ lệ cho Train, Validation, Test phải nằm trong khoảng từ 0.0 đến 1.0.")

    all_data = []
    try:
        with open(input_txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue # Bỏ qua các dòng trống

                parts = line.split('\t', 1)
                if len(parts) == 2:
                    image_id = parts[0].strip()
                    caption = parts[1].strip()

                    # Thêm tiền tố và đuôi .jpg cho đường dẫn ảnh
                    full_image_path = image_prefix + image_id + ".jpg"
                    all_data.append({"image": full_image_path, "text": caption})
                else:
                    print(f"Cảnh báo: Bỏ qua dòng bị lỗi định dạng: {line}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file đầu vào '{input_txt_file}'.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file: {e}")
        return

    # Xáo trộn dữ liệu để đảm bảo tính ngẫu nhiên khi chia
    random.shuffle(all_data)

    total_items = len(all_data)

    # Tính số lượng mục cho mỗi tập
    num_train = int(total_items * train_ratio)
    num_validation = int(total_items * validation_ratio)
    num_test = int(total_items * test_ratio)

    # Chia dữ liệu thành các tập
    # Lưu ý: Các lát cắt (slicing) này đảm bảo không vượt quá kích thước của all_data
    # và các phần tử được lấy theo thứ tự từ danh sách đã xáo trộn.
    train_set = all_data[:num_train]
    validation_set = all_data[num_train : num_train + num_validation]
    test_set = all_data[num_train + num_validation : num_train + num_validation + num_test]

    output_data = {
        "train": train_set,
        "validation": validation_set,
        "test": test_set
    }

    try:
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"Đã tạo file '{output_json_file}' thành công với:")
        print(f"  Số lượng mục Train: {len(train_set)}")
        print(f"  Số lượng mục Validation: {len(validation_set)}")
        print(f"  Số lượng mục Test: {len(test_set)}")
        print(f"  Tổng số mục đã được xử lý: {len(train_set) + len(validation_set) + len(test_set)}")
        print(f"  Số mục không được đưa vào bất kỳ tập nào: {total_items - (len(train_set) + len(validation_set) + len(test_set))}")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi ghi file JSON: {e}")



convert_txt_to_json(
    input_txt_file='data/data/train/radiology/captions.txt', # Replace with your actual .txt file name
    output_json_file='data/2D_Cap/2D_Cap.json',
    train_ratio=0.0005,  # Adjust ratios as needed
    validation_ratio=0.0005,
    test_ratio=0.005,
    image_prefix="data/train/radiology/images/" # Change this to your actual image directory prefix
)