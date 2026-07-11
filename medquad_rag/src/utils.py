"""
utils.py
--------
Các hàm tiện ích dùng chung (đọc/ghi dữ liệu, validate sample, chia
train/val/test) — tách ra từ build_train_dataset.py để pipeline/evaluate.py
cũng có thể tái sử dụng khi cần đọc trực tiếp medquad.json thay vì chỉ dùng
vài câu hỏi mẫu.
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

    Args:
        samples: list các sample (đã validate)
        train_ratio, val_ratio, test_ratio: tỷ lệ chia, phải cộng lại ~1.0
        seed: random seed để chia lần nào cũng ra kết quả giống nhau

    Returns:
        (train_samples, val_samples, test_samples)
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
    # test lấy phần còn lại, tránh rơi rớt sample do làm tròn số
    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train:n_train + n_val]
    test_samples = shuffled[n_train + n_val:]

    return train_samples, val_samples, test_samples
