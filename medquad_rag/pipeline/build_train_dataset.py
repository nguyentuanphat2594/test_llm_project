"""
build_train_dataset.py
-----------------------
Build 3 tập train/val/test từ medquad.json:
  - train.jsonl : dùng để fine-tune (pipeline/train.py)
  - val.jsonl   : theo dõi loss trong lúc train, không dùng để cập nhật
                  trọng số (pipeline/train.py)
  - test.jsonl  : GIỮ NGUYÊN, không đụng tới lúc train -- dùng để chấm điểm
                  cuối cùng bằng RAGAs (pipeline/evaluate.py). Model chưa
                  từng thấy các câu hỏi này lúc train.

train.jsonl / val.jsonl lưu ở dạng "messages" chuẩn (chat format) sẵn sàng
đưa vào SFTTrainer. test.jsonl lưu ở dạng "thô" (question / contexts /
ground_truth) vì evaluate.py cần tự sinh câu trả lời bằng model rồi mới
build prompt, không dùng answer có sẵn.

Mặc định KHÔNG dùng RAG (xem src/config.py -> USE_RAG): train/test thuần
Q&A, chưa có đoạn ngữ cảnh nào. Khi nối vector DB thật, bật lại bằng
MEDQUAD_USE_RAG=1.

Cách chạy:
    python -m pipeline.build_train_dataset
"""

from src.config import (
    INPUT_FILE,
    TEST_FILE,
    TRAIN_FILE,
    TRAIN_RATIO,
    TRAIN_SAMPLE_LIMIT,
    TEST_RATIO,
    USE_RAG,
    VAL_FILE,
    VAL_RATIO,
    SPLIT_SEED,
)
from src.prompt_template import build_prompt
from src.utils import load_dataset, save_jsonl, split_dataset, validate_sample


def to_chat_sample(sample):
    """Chuyển 1 sample question/answer thành messages hoàn chỉnh để train (SFT)."""
    question = sample["question"].strip()
    answer = sample["answer"].strip()

    # Nếu USE_RAG=True: dùng chính "answer" gốc trong MedQuAD làm chunk giả
    # lập (chưa có vector DB thật). Nếu USE_RAG=False (mặc định hiện tại):
    # không đưa ngữ cảnh nào, model học trả lời bằng kiến thức đã fine-tune.
    chunks = [answer] if USE_RAG else None
    messages = build_prompt(chunks=chunks, question=question)

    # build_prompt() chỉ trả về [system, user] (chưa có câu trả lời).
    # Thêm message "assistant" để hoàn chỉnh 1 sample train.
    messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def to_eval_sample(sample):
    """Chuyển 1 sample question/answer thành format THÔ để evaluate.py dùng
    (không phải messages, vì evaluate.py cần tự sinh câu trả lời bằng model
    trước khi build prompt, và cần contexts/ground_truth riêng cho RAGAs)."""
    question = sample["question"].strip()
    answer = sample["answer"].strip()

    return {
        "question": question,
        # contexts để trống nếu không dùng RAG -- RAGAs sẽ bỏ qua các metric
        # cần context (faithfulness/context_precision/context_recall) trong
        # trường hợp này, xem pipeline/evaluate.py.
        "contexts": [answer] if USE_RAG else [],
        "ground_truth": answer,
    }


def main():
    dataset = load_dataset(INPUT_FILE)

    valid_samples = [s for s in dataset[:TRAIN_SAMPLE_LIMIT] if validate_sample(s)]
    skipped = len(dataset[:TRAIN_SAMPLE_LIMIT]) - len(valid_samples)

    train_raw, val_raw, test_raw = split_dataset(
        valid_samples,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SPLIT_SEED,
    )

    train_out = [to_chat_sample(s) for s in train_raw]
    val_out = [to_chat_sample(s) for s in val_raw]
    test_out = [to_eval_sample(s) for s in test_raw]

    save_jsonl(train_out, TRAIN_FILE)
    save_jsonl(val_out, VAL_FILE)
    save_jsonl(test_out, TEST_FILE)

    print("=" * 40)
    print(f"Total samples (raw)   : {len(dataset)}")
    print(f"Sample limit áp dụng  : {TRAIN_SAMPLE_LIMIT}")
    print(f"Hợp lệ (validated)    : {len(valid_samples)}")
    print(f"Bỏ qua (thiếu Q/A)    : {skipped}")
    print(f"USE_RAG               : {USE_RAG}")
    print("-" * 40)
    print(f"Train : {len(train_out)} -> {TRAIN_FILE}")
    print(f"Val   : {len(val_out)} -> {VAL_FILE}")
    print(f"Test  : {len(test_out)} -> {TEST_FILE}")


if __name__ == "__main__":
    main()
