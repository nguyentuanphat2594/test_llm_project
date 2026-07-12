"""
evaluate.py
-----------
Đánh giá chất lượng chatbot RAG bằng RAGAs.

QUAN TRỌNG - GIÁM KHẢO PHẢI TÁCH RỜI MODEL ĐANG ĐƯỢC ĐÁNH GIÁ:
Trước đây file này dùng chính model vừa fine-tune (base + adapter LoRA) làm
luôn LLM Judge -- giống 1 học sinh tự chấm bài thi của mình: model có xu
hướng tự đánh giá cao câu trả lời của chính nó (self-preference bias), kết
quả không đáng tin.

Giờ tách làm 2 model độc lập:
  - Model BỊ ĐÁNH GIÁ: base model + adapter LoRA vừa train (pipeline/train.py)
    -> chỉ dùng để SINH câu trả lời.
  - Model GIÁM KHẢO: Prometheus 2 (prometheus-eval/prometheus-7b-v2.0) -- model
    được train CHUYÊN để chấm điểm LLM khác, KHÔNG dính dáng gì tới model vừa
    train ở trên. Xem src/config.py (JUDGE_MODEL_NAME).

LƯU Ý:
Prometheus 2 vẫn nhỏ hơn nhiều so với GPT-4/Claude nên kết quả chỉ mang tính
tham khảo, nhưng đáng tin hơn hẳn việc để model tự chấm bài mình.

Cài đặt cần thiết:
    pip install -r requirements.txt

Cách chạy:
    python -m pipeline.evaluate
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from src.config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    JUDGE_LOAD_IN_4BIT,
    JUDGE_MODEL_NAME,
    MAX_NEW_TOKENS_TRAIN_GEN,
    TEST_FILE,
    USE_RAG,
)
from src.prompt_template import build_prompt

USE_GPU = torch.cuda.is_available()


# ============================================================
# 1a. LOAD MODEL BỊ ĐÁNH GIÁ (base model + adapter LoRA vừa train)
#     -> chỉ dùng để sinh câu trả lời, KHÔNG dùng làm giám khảo.
# ============================================================

def load_model_under_test():
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if USE_GPU else torch.float32,
        device_map="auto" if USE_GPU else {"": "cpu"},
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    return model, tokenizer


# ============================================================
# 1b. LOAD MODEL GIÁM KHẢO (Prometheus 2) — hoàn toàn tách biệt,
#     không load adapter LoRA, không liên quan tới model vừa train.
# ============================================================

def load_judge_model():
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)

    if USE_GPU and JUDGE_LOAD_IN_4BIT:
        # Prometheus 2 là model 7B -> quantize 4-bit để vừa GPU free tier
        # (Colab T4 / Kaggle T4-P100, ~15-16GB VRAM).
        print(f"Load {JUDGE_MODEL_NAME} ở chế độ 4-bit (giám khảo, tách biệt model đang train)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif USE_GPU:
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        print(
            "CẢNH BÁO: không có GPU -> chạy Prometheus 2 (7B) trên CPU sẽ RẤT chậm. "
            "Cân nhắc chạy evaluate.py trên máy/Colab/Kaggle có GPU."
        )
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

    # Bọc model thành 1 "pipeline" text-generation, để LangChain/RAGAs gọi được
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,  # tắt random để LLM Judge chấm điểm ổn định hơn
    )

    return HuggingFacePipeline(pipeline=gen_pipeline)


# ============================================================
# 2. TẢI DỮ LIỆU ĐÁNH GIÁ TỪ test.jsonl
#    (tập TEST -- model chưa từng thấy lúc train, xem
#    pipeline/build_train_dataset.py)
# ============================================================

def load_eval_samples():
    """
    Đọc test.jsonl (sinh bởi pipeline/build_train_dataset.py). Mỗi dòng có
    dạng {question, contexts, ground_truth}. contexts rỗng nếu USE_RAG=False.
    """
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Không tìm thấy {TEST_FILE}. "
            f"Hãy chạy `python -m pipeline.build_train_dataset` trước để tạo file này."
        )

    samples = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        raise ValueError(
            f"{TEST_FILE} rỗng -- tập test quá nhỏ so với dataset gốc. "
            f"Tăng TRAIN_SAMPLE_LIMIT hoặc TEST_RATIO trong src/config.py rồi build lại."
        )

    return samples


def generate_answers(samples, model, tokenizer):
    """Dùng model bị đánh giá để sinh câu trả lời thật cho từng câu hỏi test."""

    for sample in samples:
        messages = build_prompt(sample["contexts"], sample["question"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_TRAIN_GEN,
                do_sample=False,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        sample["answer"] = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return samples


# ============================================================
# 3. CHẠY RAGAs
# ============================================================

def select_metrics():
    """
    context_precision/context_recall/faithfulness cần "contexts" thật (đo
    độ bám ngữ cảnh) -- vô nghĩa khi USE_RAG=False (không có contexts nào).
    Chỉ answer_relevancy (so khớp câu hỏi <-> câu trả lời, không cần context)
    dùng được trong cả 2 chế độ.
    """
    if USE_RAG:
        return [faithfulness, answer_relevancy, context_precision, context_recall]
    print(
        "USE_RAG=False -> bỏ qua faithfulness/context_precision/context_recall "
        "(cần contexts thật, hiện không có). Chỉ chấm answer_relevancy."
    )
    return [answer_relevancy]


def main():
    print("Đang load model bị đánh giá (base + adapter LoRA vừa train)...")
    model, tokenizer = load_model_under_test()

    print("Đang tải dữ liệu đánh giá từ test.jsonl...")
    samples = load_eval_samples()
    print(f"Số câu hỏi test: {len(samples)}")
    samples = generate_answers(samples, model, tokenizer)

    # Giải phóng model bị đánh giá trước khi load giám khảo (7B, tốn VRAM)
    del model
    if USE_GPU:
        torch.cuda.empty_cache()

    print(f"Đang load model giám khảo ({JUDGE_MODEL_NAME}, tách biệt model vừa train)...")
    llm_judge = load_judge_model()

    dataset = Dataset.from_list(samples)

    print("Đang load embedding model (cho Answer Relevance)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Đang chấm điểm bằng RAGAs (có thể chậm vì dùng model local)...")
    result = evaluate(
        dataset,
        metrics=select_metrics(),
        llm=llm_judge,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=7200,       # tăng timeout mỗi job lên 120 phút (model local chậm)
            max_workers=1,     # chạy tuần tự thật sự, đúng bản chất model local trên 1 GPU
        ),
    )

    print("=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 50)
    print(result)


if __name__ == "__main__":
    main()
