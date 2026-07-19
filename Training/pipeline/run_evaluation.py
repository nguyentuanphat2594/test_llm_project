"""
Chạy RIÊNG bước inference + ROUGE/BLEU trên model ĐÃ TRAIN SẴN, KHÔNG
train lại. Dùng khi đã có checkpoint và chỉ muốn:
- Đổi max_samples (test full thay vì demo 100 mẫu).
- Chạy lại vì lần trước bị ngắt giữa chừng (resume=True).
- Bật/tắt RAG (USE_RAG trong src/config.py) mà không cần train lại model.

Tự động đọc tập test (test.jsonl), load model + adapter LoRA, và nếu
USE_RAG=True sẽ tự nối rag_bridge để retrieve context thật cho từng câu
hỏi trước khi generate. Kết quả lưu vào evaluation_results.csv (đường dẫn
theo PREDICTIONS_CSV trong config), sẵn sàng cho bước LLM Judge sau đó.

Cách gọi từ notebook:
    run_evaluation.main()                              # full tập test
    run_evaluation.main(max_samples=500)                # 500 mẫu
    run_evaluation.main(resume=True)                    # nối tiếp CSV cũ
    run_evaluation.main(rag_similarity_threshold=0.70)   # đổi ngưỡng RAG
"""

import argparse
import json
import os

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    MAX_NEW_TOKENS_TRAIN_GEN,
    PREDICTIONS_CSV,
    PROMPT_STYLE,
    SYSTEM_PROMPT,
    TEST_FILE,
    USE_RAG,
)
from src.evaluation import save_predictions_csv

USE_GPU = torch.cuda.is_available()


def load_trained_model():
    """Load base model + adapter LoRA đã train sẵn từ ADAPTER_DIR (giống
    cách pipeline/chat.py load, KHÔNG gắn LoRA mới / KHÔNG train)."""
    if not os.path.exists(ADAPTER_DIR):
        raise FileNotFoundError(
            f"Không tìm thấy {ADAPTER_DIR}. Hãy chạy `python -m pipeline.train` "
            f"trước để có model đã train."
        )

    print(f"Đang load model đã train từ {ADAPTER_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if USE_GPU else torch.float32,
        device_map="auto" if USE_GPU else {"": "cpu"},
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    print("Load xong.")
    return model, tokenizer


def load_raw_test_for_export(path):
    """Đọc thẳng test.jsonl -> Dataset {question, answer} (map ground_truth
    -> answer nếu cần), giống hệt logic trong pipeline/train.py."""
    if not os.path.exists(path):
        return None

    raw_samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_samples.append({
                "question": row["question"],
                "answer": row.get("ground_truth", row.get("answer", "")),
            })

    return Dataset.from_list(raw_samples) if raw_samples else None


def main(max_samples: int = None, resume: bool = False, rag_similarity_threshold: float = None):
    """
    Args:
        max_samples: số mẫu test dùng để tính ROUGE/BLEU.
            - None (mặc định) -> chạy FULL tập test.
            - Số cụ thể -> chỉ chạy đúng số đó.
        resume: True -> đọc CSV cũ tại PREDICTIONS_CSV (nếu có), bỏ qua các
            câu đã có sẵn prediction, chỉ generate tiếp phần còn thiếu.
            CHỈ bật khi chắc chắn CSV cũ là của ĐÚNG model hiện tại.
        rag_similarity_threshold: ngưỡng % tương đồng (0..1, vd 0.70) để
            CHẤP NHẬN context RAG. None -> dùng SIMILARITY_THRESHOLD mặc
            định của RAG project. Chỉ áp dụng đúng ý nghĩa khi
            RETRIEVAL_MODE của RAG project là "cosine" (xem rag_bridge.py).
    """
    print("Đang tải tập test (thô)...")
    test_raw = load_raw_test_for_export(TEST_FILE)
    if test_raw is None:
        raise FileNotFoundError(
            f"Không tìm thấy/rỗng {TEST_FILE}. Hãy chạy "
            f"`python -m pipeline.build_train_dataset` trước."
        )

    if max_samples is None:
        # Chỉ parse argparse khi cần (không đụng khi gọi trực tiếp từ
        # notebook với max_samples đã truyền sẵn) -- tránh lỗi "-f kernel.json"
        # của Jupyter/Colab/Kaggle.
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--max_samples",
            type=int,
            default=len(test_raw),  # mặc định: FULL tập test
            help="Số mẫu test dùng để tính ROUGE/BLEU (mặc định: toàn bộ tập test).",
        )
        args = parser.parse_known_args()[0]
        max_samples = args.max_samples

    model, tokenizer = load_trained_model()

    print(f"Số câu hỏi test: {len(test_raw)} | Sẽ chạy: {min(len(test_raw), max_samples)}")

    retrieve_context_fn = None
    if USE_RAG:
        print("USE_RAG=True -> sẽ retrieve context THẬT + lọc theo % tương đồng (qua rag_bridge).")
        from src.rag_bridge import get_context_with_similarity
        retrieve_context_fn = get_context_with_similarity
    else:
        print("USE_RAG=False -> chạy Q&A thuần, không có ngữ cảnh (như cũ).")

    save_predictions_csv(
        model=model,
        tokenizer=tokenizer,
        dataset=test_raw,
        output_path=str(PREDICTIONS_CSV),
        system_prompt=SYSTEM_PROMPT,
        prompt_style=PROMPT_STYLE,
        max_new_tokens=MAX_NEW_TOKENS_TRAIN_GEN,
        max_samples=max_samples,
        resume=resume,
        retrieve_context_fn=retrieve_context_fn,
        rag_similarity_threshold=rag_similarity_threshold,
    )
    print(f"Đã lưu CSV dự đoán -> {PREDICTIONS_CSV}")

    pred_df = pd.read_csv(PREDICTIONS_CSV)
    print(f"ROUGE-1 (avg): {pred_df['rouge1'].mean():.4f}")
    print(f"ROUGE-2 (avg): {pred_df['rouge2'].mean():.4f}")
    print(f"ROUGE-L (avg): {pred_df['rougeL'].mean():.4f}")
    print(f"BLEU    (avg): {pred_df['bleu'].mean():.4f}")
    print(
        "Bước tiếp theo: chạy `python -m pipeline.evaluate` để đưa CSV này "
        "qua LLM Judge (RAGAs + Prometheus/API)."
    )


if __name__ == "__main__":
    main()