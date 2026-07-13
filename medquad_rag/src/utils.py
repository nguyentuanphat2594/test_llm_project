"""
utils.py
--------
Các hàm tiện ích dùng chung (đọc/ghi dữ liệu, validate sample, chia
train/val/test, load+format dataset cho SFTTrainer) -- được tái sử dụng bởi
build_train_dataset.py, train.py VÀ hpo_train.py.
"""

import json
import os
import random


def load_dataset(path):
    """Đọc file JSON (list các sample question/answer)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_sample(sample):
    """Kiểm tra 1 sample có đủ 'question' và 'answer' hay không."""
    return (
        isinstance(sample, dict)
        and sample.get("question")
        and sample.get("answer")
    )


def save_jsonl(data, path):
    """Ghi danh sách dict ra file .jsonl (mỗi dòng 1 JSON object)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def split_dataset(samples, train_ratio, val_ratio, test_ratio, seed):
    """
    Xáo trộn (deterministic, theo seed) rồi chia danh sách sample thành 3
    tập train/val/test theo tỷ lệ cho trước.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio phải = 1.0, hiện đang = {total_ratio}"
        )

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train:n_train + n_val]
    test_samples = shuffled[n_train + n_val:]

    return train_samples, val_samples, test_samples


def format_chat_dataset(tokenizer, path, required=True):
    """
    Đọc 1 file .jsonl dạng {"messages": [...]} (train.jsonl / val.jsonl) và
    áp dụng chat template của tokenizer -> dataset có cột "text" sẵn sàng
    đưa vào SFTTrainer (dataset_text_field="text").

    Dùng chung cho pipeline/train.py VÀ pipeline/hpo_train.py để tránh lệch
    logic format giữa 2 nơi.

    Args:
        required: nếu True mà file không tồn tại -> raise lỗi rõ ràng.
                  Nếu False -> trả về None (dùng cho val.jsonl khi không bắt
                  buộc phải có, ví dụ script train.py chạy nhanh không HPO).

    Returns:
        datasets.Dataset (đã có cột "text") hoặc None.
    """
    from datasets import load_dataset as hf_load_dataset

    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Không tìm thấy {path}. "
                f"Hãy chạy `python -m pipeline.build_train_dataset` trước để tạo file này."
            )
        return None

    dataset = hf_load_dataset("json", data_files=str(path), split="train")

    if len(dataset) == 0:
        return None

    def format_sample(sample):
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_sample)
    return dataset