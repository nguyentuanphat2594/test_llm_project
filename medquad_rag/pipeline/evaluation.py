"""
evaluation.py
--------------
Đánh giá nhanh model sau khi train bằng ROUGE/BLEU/Perplexity + xuất CSV kết
quả trên tập test (test.jsonl -- {question, contexts, ground_truth}, xem
pipeline/build_train_dataset.py). Dùng chung cho cả pipeline/train.py VÀ
pipeline/hpo_train.py (bước Final Training) để không lệch logic đánh giá
giữa 2 nơi.

Khác với 1 số project dùng template alpaca/chatml viết tay: file này SINH
CÂU TRẢ LỜI bằng build_prompt() (src/prompt_template.py) -- tôn trọng đúng
cờ USE_RAG/contexts giống hệt lúc build_train_dataset.py và chat.py, tránh
lệch prompt giữa lúc train/chat và lúc đánh giá.

Đây chỉ là bước đánh giá NHANH (so khớp từ vựng, không cần load thêm model
nào khác) -- chạy TRƯỚC bước LLM Judge (RAGAs, xem pipeline/evaluate.py) để
có con số tham khảo ngay sau khi train xong.
"""

import json
import math
import os
from typing import Optional

import evaluate
import nltk
import pandas as pd
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from src.prompt_template import build_prompt

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

_rouge_metric = evaluate.load("rouge")
_bleu_metric = evaluate.load("bleu")


def compute_perplexity(eval_loss: Optional[float]) -> Optional[float]:
    if eval_loss is None:
        return None
    return math.exp(eval_loss) if eval_loss < 100 else float("inf")


def load_raw_test_samples(test_file) -> Optional[Dataset]:
    """
    Đọc test.jsonl (sinh bởi build_train_dataset.py, mỗi dòng
    {question, contexts, ground_truth}) -> Dataset dùng cho
    save_predictions_csv(). Trả về None nếu file không tồn tại/rỗng
    (giống format_chat_dataset(required=False) trong src/utils.py).
    """
    if not os.path.exists(test_file):
        return None

    rows = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append({
                "question": row["question"],
                "contexts": row.get("contexts") or [],
                "ground_truth": row["ground_truth"],
            })

    return Dataset.from_list(rows) if rows else None


def save_predictions_csv(
    model,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    output_path: str,
    max_new_tokens: int = 300,
    max_samples: int = 100,
) -> pd.DataFrame:
    """
    Chạy inference trên tập test (build_prompt() + apply_chat_template(),
    giống hệt cách pipeline/chat.py sinh câu trả lời) rồi lưu kết quả +
    ROUGE/BLEU ra CSV.

    File CSV: question, reference, prediction, rouge1, rouge2, rougeL, bleu

    Args:
        model: model đã train (PEFT-wrapped)
        tokenizer: tokenizer tương ứng
        dataset: Dataset {question, contexts, ground_truth}
                 (xem load_raw_test_samples())
        output_path: đường dẫn CSV đầu ra
        max_new_tokens: số token tối đa sinh ra mỗi câu trả lời
        max_samples: giới hạn số mẫu inference (generate tuần tự khá chậm,
                     không nên chạy hết toàn bộ test set theo mặc định)

    Returns:
        pandas.DataFrame chứa toàn bộ kết quả (cũng đã ghi ra output_path).
    """
    model.eval()
    results = []
    num_samples = min(len(dataset), max_samples)

    print(f"[EVAL] Bắt đầu inference {num_samples} mẫu...")

    for i in range(num_samples):
        example = dataset[i]
        question = example["question"]
        contexts = example.get("contexts") or []
        reference = example["ground_truth"]

        messages = build_prompt(contexts, question)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        rouge_scores = _rouge_metric.compute(
            predictions=[prediction],
            references=[reference],
            use_stemmer=True,
        )

        pred_tokens = nltk.word_tokenize(prediction) if prediction else []
        ref_tokens = [nltk.word_tokenize(reference)]
        try:
            bleu_score = _bleu_metric.compute(
                predictions=[pred_tokens],
                references=[ref_tokens],
            )["bleu"]
        except (ZeroDivisionError, ValueError):
            # BLEU đòi khớp cả n-gram tới 4-gram liên tiếp -- rất dễ ra 0
            # với câu trả lời bị paraphrase, không phải lỗi.
            bleu_score = 0.0

        results.append({
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "rouge1": round(rouge_scores["rouge1"], 4),
            "rouge2": round(rouge_scores["rouge2"], 4),
            "rougeL": round(rouge_scores["rougeL"], 4),
            "bleu": round(bleu_score, 4),
        })

        if (i + 1) % 10 == 0:
            print(f"[EVAL] Đã inference {i + 1}/{num_samples} mẫu...")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"[KẾT QUẢ] Đã lưu {len(results)} mẫu → {output_path}")
    print(f"  ROUGE-1 (avg): {df['rouge1'].mean():.4f}")
    print(f"  ROUGE-2 (avg): {df['rouge2'].mean():.4f}")
    print(f"  ROUGE-L (avg): {df['rougeL'].mean():.4f}")
    print(f"  BLEU    (avg): {df['bleu'].mean():.4f}")
    print(f"{'=' * 60}")

    return df


def append_summary_row(summary: dict, summary_csv) -> None:
    """
    Ghi thêm (append) 1 dòng tổng hợp vào SUMMARY_CSV -- dễ so sánh giữa các
    lần train (model, hyperparameters, rouge1/2/L, bleu, perplexity,
    val_loss...). Tự tạo file + header nếu chưa tồn tại.
    """
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    df_row = pd.DataFrame([summary])
    header = not os.path.exists(summary_csv)
    df_row.to_csv(summary_csv, mode="a", header=header, index=False, encoding="utf-8")
    print(f"Đã lưu (append) dòng tổng hợp -> {summary_csv}")