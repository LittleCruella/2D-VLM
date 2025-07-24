import os
import json
import torch
import random
import numpy as np
# import monai.transforms as mtf
import SimpleITK as sitk
import pandas as pd
import monai.transforms as mtf
from monai.data import set_track_meta
from torch.utils.data import Dataset, ConcatDataset
from src.dataset.prompt_templates import Caption_templates
import torchvision.transforms as T # Renamed to T for brevity
from PIL import Image


class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, "r", encoding='utf-8') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates
        if len(self.args.input_image) >= 2:
            target_image_size = (self.args.input_image[0], self.args.input_image[1])
        else:
            raise ValueError("input_image in ModelArguments must specify at least H and W dimensions (e.g., (256, 256) or (256, 256, 128)).")

        # Transforms for training
        train_transform = T.Compose(
            [
                T.Resize(target_image_size), # Resize to fixed size
                T.RandomApply([T.RandomRotation(degrees=90)], p=0.5), # Equivalent to RandRotate90 for 2D
                T.RandomHorizontalFlip(p=0.10), # Equivalent to RandFlip spatial_axis=1
                T.RandomVerticalFlip(p=0.10), # Equivalent to RandFlip spatial_axis=0
                T.ToTensor(), # Convert to tensor (H, W) -> (C, H, W)
            ]
        )

        # Transforms for validation/test
        val_transform = T.Compose(
            [
                T.Resize(target_image_size), # Resize to fixed size
                T.ToTensor(), # Convert to tensor (H, W) -> (C, H, W)
            ]
        )

        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform
            self.data_list = self.data_list[:test_size]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                # print(f"Processing index: {idx}")
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                # Read image using SimpleITK
                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image) # H, W, C
                if image.ndim == 3:
                    image = image.mean(-1)
                # print(f"Image shape before transform: {image.shape}")
                # Convert numpy array to PIL Image before applying transforms
                # image = Image.fromarray(image[0])  # Assuming image is 3D and selecting the first channel

  
                # Chuyển đổi sang kiểu uint8 nếu cần thiết
                # if image.dtype != np.uint8:
                #     image_min = image.min()
                #     image_max = image.max()
                #     if image_max > image_min:
                #         image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                #     else:
                #         image = np.zeros_like(image, dtype=np.uint8)

                image_pil = Image.fromarray(image)
                # print(f"Image shape after conversion to PIL: {image_pil.size}")
                # Apply the transformations
                image_tensor = self.transform(image_pil)

                # print(f"Image shape after transform: {image.shape}")

                # Process the caption text
                raw_text = data["text"]
                answer = raw_text

                prompt_question = random.choice(self.caption_prompts)
                question = self.image_tokens + prompt_question

                # Tokenize the question and answer
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                if valid_len < len(label):
                    label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image_tensor,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "Caption",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")
        
        if len(self.args.input_image) >= 2:
            target_image_size = (self.args.input_image[0], self.args.input_image[1])
        else:
            raise ValueError("input_image in ModelArguments must specify at least H and W dimensions (e.g., (256, 256) or (256, 256, 128)).")

        # Transforms for training
        train_transform = T.Compose(
            [
                # First, ensure image is a PIL Image or compatible for torchvision transforms
                # For medical images from SimpleITK, it's often a numpy array.
                # We'll convert to PIL Image during __getitem__ before these transforms.
                T.Resize(target_image_size), # Resize to fixed size
                T.RandomApply([T.RandomRotation(degrees=90)], p=0.5), # Equivalent to RandRotate90 for 2D
                T.RandomHorizontalFlip(p=0.10), # Equivalent to RandFlip spatial_axis=1
                T.RandomVerticalFlip(p=0.10), # Equivalent to RandFlip spatial_axis=0
                # For intensity transforms, torchvision doesn't have direct equivalents for RandScaleIntensity/RandShiftIntensity.
                # You might need custom transforms or libraries like Albumentations for these.
                # For simplicity here, we'll omit them or use basic color jitter if applicable (not for grayscale medical images).
                # If these are crucial, consider implementing them as custom torch.nn.Module or using Albumentations.
                # For now, let's add a placeholder for intensity adjustment if needed.
                # T.ColorJitter(brightness=0.1, contrast=0.1) # Not suitable for medical images often
                T.ToTensor(), # Convert to tensor (H, W) -> (C, H, W) where C is 1 for grayscale
                # Normalization is often applied after ToTensor. Add if needed.
                # T.Normalize(mean=[0.5], std=[0.5])
            ]
        )

        # Transforms for validation/test
        val_transform = T.Compose(
            [
                T.Resize(target_image_size), # Resize to fixed size
                T.ToTensor(), # Convert to tensor (H, W) -> (C, H, W)
                # T.Normalize(mean=[0.5], std=[0.5])
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["image"])

                # image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image)
                # image = np.expand_dims(image, axis=0)
                # if image.dtype != np.uint8:
                #     image_min = image.min()
                #     image_max = image.max()
                #     if image_max > image_min:
                #         image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                #     else: # Handle cases where image is constant (e.g., all zeros)
                #         image = np.zeros_like(image, dtype=np.uint8)

                if image.ndim == 3:
                    image = image.mean(-1)  # Convert to grayscale if needed

                image_pil = Image.fromarray(image)
                # Apply the transformations
                image_tensor = self.transform(image_pil)

                # if self.close_ended:
                #     question = data["Question"]
                #     choices = "Choices: A. {} B. {} C. {} D. {}".format(
                #         data["Choice A"],
                #         data["Choice B"],
                #         data["Choice C"],
                #         data["Choice D"],
                #     )
                #     question = question + " " + choices
                #     answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                # else:
                #     question = data["Question"]
                #     answer = str(data["Answer"])
                question = data["question"]
                answer = str(data["answer"])
                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image_tensor,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    # "answer_choice": data["Answer Choice"],
                    "question_type": "VQA",
                }

                # if self.args.seg_enable:
                #     ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQAYNDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_yn_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_yn_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_yn_data_test_path)
        else:
            print("The mode is not desired ! ")

        # Transforms for training
        train_transform = T.Compose(
            [
                # First, ensure image is a PIL Image or compatible for torchvision transforms
                # For medical images from SimpleITK, it's often a numpy array.
                # We'll convert to PIL Image during __getitem__ before these transforms.
                T.Resize(target_image_size), # Resize to fixed size
                T.RandomApply([T.RandomRotation(degrees=90)], p=0.5), # Equivalent to RandRotate90 for 2D
                T.RandomHorizontalFlip(p=0.10), # Equivalent to RandFlip spatial_axis=1
                T.RandomVerticalFlip(p=0.10), # Equivalent to RandFlip spatial_axis=0
                # For intensity transforms, torchvision doesn't have direct equivalents for RandScaleIntensity/RandShiftIntensity.
                # You might need custom transforms or libraries like Albumentations for these.
                # For simplicity here, we'll omit them or use basic color jitter if applicable (not for grayscale medical images).
                # If these are crucial, consider implementing them as custom torch.nn.Module or using Albumentations.
                # For now, let's add a placeholder for intensity adjustment if needed.
                # T.ColorJitter(brightness=0.1, contrast=0.1) # Not suitable for medical images often
                T.ToTensor(), # Convert to tensor (H, W) -> (C, H, W) where C is 1 for grayscale
                # Normalization is often applied after ToTensor. Add if needed.
                # T.Normalize(mean=[0.5], std=[0.5])
            ]
        )

        # Transforms for validation/test
        val_transform = T.Compose(
            [
                T.Resize(target_image_size), # Resize to fixed size
                T.ToTensor(), # Convert to tensor (H, W) -> (C, H, W)
                # T.Normalize(mean=[0.5], std=[0.5])
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                # image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image)
                image = np.expand_dims(image, axis=0)
                image = self.transform(image)

                question = data["Question"]
                answer = str(data["Answer"])

                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": data["Answer Choice"],
                    "question_type": data["Question Type"],
                }
                # if self.args.seg_enable:
                #     ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            # VQADataset(args, tokenizer, close_ended=False, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TextYNDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextYNDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
            VQAYNDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]