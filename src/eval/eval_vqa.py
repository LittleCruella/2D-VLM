import os
import sys
# Đường dẫn tương đối cho các module
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
parent_dir = os.path.abspath(os.path.join(src_dir, '..'))
sys.path.append(parent_dir)


import argparse
import csv
import random

# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.llm.qwen import VLMQwenForCausalLM
from src.dataset.mllm_dataset import VQADataset

bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/Med3DVLM-Qwen-2.5-1.5B-diff-2"
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--vqa_data_test_path", type=str, default="data/vqa_rad/test_other.csv"
    )
    parser.add_argument("--close_ended", action="store_true", default=False)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/eval_vqa/",
    )

    parser.add_argument("--proj_out_num", type=int, default=256)
    parser.add_argument(
        "--input_image", type=tuple, default=(256, 256)
    )  
    return parser.parse_args(args)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = VLMQwenForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", trust_remote_code=True
    )
    model = model.to(device=device)

    test_dataset = VQADataset(
        args, tokenizer=tokenizer, close_ended=args.close_ended, mode="test"
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_name = args.model_name_or_path.split("/")[-1]

    if args.close_ended:
        print("Evaluating close-ended VQA...")
        output_path = os.path.join(args.output_dir, f"{model_name}_eval_close_vqa.csv")
        with open(output_path, mode="w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                [
                    "Question",
                    "Answer",
                    "Pred",
                    "Correct",
                ]
            )
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                answer = sample["answer"]

                image = sample["image"].to(device=device)

                input_id = tokenizer(question, return_tensors="pt")["input_ids"].to(
                    device=device
                )

                with torch.inference_mode():
                    generation = model.generate(
                        image,
                        input_id,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        top_p=args.top_p,
                        temperature=args.temperature,
                    )
                generated_texts = tokenizer.batch_decode(
                    generation, skip_special_tokens=True
                )
                # print(f"Generated text: {generated_texts}")
                # print(f"Answer: {answer}")
                if answer[0] in generated_texts[0].lower():
                    correct = 1
                else:
                    correct = 0

                writer.writerow(
                    [
                        question[0],
                        answer[0],
                        generated_texts[0],
                        correct,
                    ]
                )
    else:
        print("Evaluating open-ended VQA...")
        output_path = os.path.join(args.output_dir, f"{model_name}_eval_open_vqa.csv")
        with open(output_path, mode="w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                [
                    "Question",
                    "Answer",
                    "Pred",
                    "bleu",
                    "rouge1",
                    "meteor",
                    "bert_f1",
                ]
            )
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                answer = sample["answer"]

                image = sample["image"].to(device=device)
                input_id = tokenizer(question, return_tensors="pt")["input_ids"].to(
                    device=device
                )

                with torch.inference_mode():
                    generation = model.generate(
                        image,
                        input_id,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        top_p=args.top_p,
                        temperature=args.temperature,
                    )
                generated_texts = tokenizer.batch_decode(
                    generation, skip_special_tokens=True
                )

                result = dict()
                # print(f"Generated text: {generated_texts}")
                # print(f"Answer: {answer}")
                decoded_preds, decoded_labels = postprocess_text(
                    generated_texts, answer
                )
                if not decoded_preds or all(len(pred) == 0 for pred in decoded_preds):
                    print("Error: Predictions are empty.")
                    continue
                # print(f"Decoded preds: {decoded_preds}")
                # print(f"Decoded labels: {decoded_labels}")
                bleu_score = bleu.compute(
                    predictions=decoded_preds, references=decoded_labels, max_order=1
                )
                result["bleu"] = bleu_score["bleu"]

                rouge_score = rouge.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    rouge_types=["rouge1"],
                )
                result["rouge1"] = rouge_score["rouge1"]

                meteor_score = meteor.compute(
                    predictions=decoded_preds, references=decoded_labels
                )
                result["meteor"] = meteor_score["meteor"]

                bert_score = bertscore.compute(
                    predictions=decoded_preds, references=decoded_labels, lang="en"
                )
                result["bert_f1"] = sum(bert_score["f1"]) / len(bert_score["f1"])
                print(f"Result: {result}")
                print(f"Question: {question[0]}")
                print(f"Answer: {answer[0]}")
                print(f"Generated text: {generated_texts[0]}")
                writer.writerow(
                    [
                        question[0],
                        answer[0],
                        generated_texts[0],
                        result["bleu"],
                        result["rouge1"],
                        result["meteor"],
                        result["bert_f1"],
                    ]
                )

    if args.close_ended:
        with open(output_path, mode="r") as infile:
            reader = csv.DictReader(infile)
            total = 0
            correct = 0
            for row in reader:
                total += 1
                if row["Correct"] == "1":
                    correct += 1
        print(f"Accuracy: {correct / total:.4f}")
    else:
        with open(output_path, mode="r") as infile:
            reader = csv.DictReader(infile)
            scores = {"bleu": [], "rouge1": [], "meteor": [], "bert_f1": []}
            for row in reader:
                for metric in scores.keys():
                    scores[metric].append(float(row[metric]))
        print("\nAverage scores:")
        for metric in scores.keys():
            avg = sum(scores[metric]) / len(scores[metric]) if scores[metric] else 0
            print(f"{metric}: {avg:.4f}")


if __name__ == "__main__":
    main()