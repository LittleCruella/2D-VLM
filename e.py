import json
import os
import random

def convert_txt_to_json(train_txt_file, 
                        val_txt_file,
                        test_txt_file,
                        output_json_file,
                        train_image_prefix, 
                        val_image_prefix, 
                        test_image_prefix):
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


    all_data_train, all_data_val, all_data_test = [], [], []
    try:
        with open(train_txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue # Bỏ qua các dòng trống

                parts = line.split('\t', 1)
                if len(parts) == 2:
                    image_id = parts[0].strip()
                    caption = parts[1].strip()

                    # Thêm tiền tố và đuôi .jpg cho đường dẫn ảnh
                    full_image_path = os.path.join(train_image_prefix, image_id + ".jpg")
                    full_image_path = "data/" + full_image_path
                    temp_image_path = f"data/{full_image_path}"
                    if os.path.exists(temp_image_path):
                        all_data_train.append({"image": full_image_path, "text": caption})
                    else:
                        print(f"Cảnh báo: Không tìm thấy ảnh: {temp_image_path}, bỏ qua.")
                        continue

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file đầu vào '{input_txt_file}'.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file: {e}")
        return

    try:
        with open(val_txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue # Bỏ qua các dòng trống

                parts = line.split('\t', 1)
                if len(parts) == 2:
                    image_id = parts[0].strip()
                    caption = parts[1].strip()

                    # Thêm tiền tố và đuôi .jpg cho đường dẫn ảnh
                    full_image_path = os.path.join(val_image_prefix, image_id + ".jpg")
                    full_image_path = "data/" + full_image_path
                    temp_image_path = f"data/{full_image_path}"
                    if os.path.exists(temp_image_path):
                        all_data_val.append({"image": full_image_path, "text": caption})
                    else:
                        print(f"Cảnh báo: Không tìm thấy ảnh: {temp_image_path}, bỏ qua.")
                        continue

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file đầu vào '{input_txt_file}'.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file: {e}")
        return
    
    try:
        with open(test_txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue # Bỏ qua các dòng trống

                parts = line.split('\t', 1)
                if len(parts) == 2:
                    image_id = parts[0].strip()
                    caption = parts[1].strip()

                    # Thêm tiền tố và đuôi .jpg cho đường dẫn ảnh
                    full_image_path = os.path.join(test_image_prefix, image_id + ".jpg")
                    full_image_path = "data/" + full_image_path
                    temp_image_path = f"data/{full_image_path}"
                    if os.path.exists(temp_image_path):
                        all_data_test.append({"image": full_image_path, "text": caption})
                    else:
                        print(f"Cảnh báo: Không tìm thấy ảnh: {temp_image_path}, bỏ qua.")
                        continue

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file đầu vào '{input_txt_file}'.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file: {e}")
        return
    # Xáo trộn dữ liệu để đảm bảo tính ngẫu nhiên khi chia
    random.shuffle(all_data_train)
    random.shuffle(all_data_val)
    random.shuffle(all_data_test)

    total_items = len(all_data_train) + len(all_data_val) + len(all_data_test)


    # Chia dữ liệu thành các tập
    # Lưu ý: Các lát cắt (slicing) này đảm bảo không vượt quá kích thước của all_data
    # và các phần tử được lấy theo thứ tự từ danh sách đã xáo trộn.
    train_set = all_data_train
    validation_set = all_data_val
    test_set = all_data_test

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
    train_txt_file="data/data/train/radiology/captions.txt", 
    val_txt_file="data/data/validation/radiology/captions.txt", 
    test_txt_file="data/data/test/radiology/captions.txt", 
    output_json_file="data/2D_Cap/2D_Cap.json",
    train_image_prefix="train/radiology/images/", 
    val_image_prefix="validation/radiology/images/", 
    test_image_prefix="test/radiology/images/"
)